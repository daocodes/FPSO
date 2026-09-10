"""Does the search converge, and is that specific to one optimizer?

The study's central measurement is that re-running an arm under a different seed
moves realised Sharpe more than any mechanism it tests. Two explanations produce
that observation and they have opposite implications:

**(a) the objective is flat** — many structurally different portfolios really are
near-optimal, so disagreement is the problem being honest about itself; or
**(b) the search stops early** — there is an identifiable answer and the runs
are landing short of it in different places.

They are separated by *headroom*. Give the same problem far more budget: if (a),
the extra budget buys nothing and the best-of-many at standard budget already
matches it. If (b), the long run is reliably better and the standard runs sit a
measurable distance below.

Why the blend sweep matters more than the answer
------------------------------------------------
A result about FPSO alone is a case report. ``pso_blend`` interpolates between two
standard algorithms — 0 is pure firefly, 1 is pure particle swarm, and the study
runs 0.5 — so sweeping it turns one measurement into three, on optimizers that
predate this work and are used throughout the portfolio literature. If all three
land short, the finding is about population metaheuristics on this problem rather
than about one hybrid.

This is not a substitute for a genuine GA or a canonical PSO implementation: all
three configurations share this study's repair operator and initialisation, so
they are three points on one axis rather than three independent algorithms. State
that limitation rather than letting a reader assume otherwise.

Usage::

    python -m fpso.experiments.convergence_probe --dates 12 --restarts 200
    python -m fpso.experiments.convergence_probe --blends 0.0 0.5 1.0 --quick
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fpso.config.loader import load_experiment_config
from fpso.data.moments import Moments, build_moment_estimator
from fpso.data.panel import ReturnPanel
from fpso.data.universe import build_universe_provider
from fpso.optimizer.fpso import FPSOOptimizer

DEFAULT_BLENDS = (0.0, 0.5, 1.0)
BLEND_NAMES = {0.0: "pure firefly", 0.5: "hybrid (as run)", 1.0: "pure PSO"}
REFERENCE_PARTICLES = 100
REFERENCE_ITERATIONS = 300
"""Budget for the reference run: roughly 3x the particles and 5x the iterations
of the study's 30 x 60.

Scaled asymmetrically on purpose. The firefly step compares every particle with
every brighter peer, so cost grows quadratically in the swarm and only linearly
in iterations; multiplying both by ten would spend ~17x the compute of this
setting to answer the same question. This is the budget the study's own landscape
probe used, which also keeps the two comparable.
"""


@dataclass(frozen=True)
class ProbeResult:
    """Headroom at one date for one blend setting."""

    date: pd.Timestamp
    blend: float
    best_standard: float
    """Best objective over `n_restarts` runs at the study's budget."""
    median_standard: float
    """What a *typical* run achieves — the quantity the study actually reports."""
    best_long: float
    """Best objective over `n_long` runs at the reference budget."""
    headroom: float
    """Relative gain of the long run over the best standard restart."""
    typical_shortfall: float
    """Relative gap between a typical standard run and the long-run best."""
    long_run_wins: bool
    """True when no standard restart matched the long run."""


def _relative_gap(better: float, worse: float) -> float:
    """Fractional improvement, guarded against a near-zero reference.

    The objective is an annualised return net of risk and can legitimately be
    small or negative, at which point a ratio is meaningless rather than merely
    imprecise. Returning NaN keeps such dates out of the averages instead of
    letting one of them dominate.
    """
    scale = abs(worse)
    if scale < 1e-9 or not np.isfinite(scale) or not np.isfinite(better):
        return float("nan")
    return (better - worse) / scale


def probe_date(
    moments: Moments,
    weights_prev: np.ndarray,
    params,
    blend: float,
    *,
    date: pd.Timestamp,
    n_restarts: int,
    n_long: int,
    seed: int,
) -> ProbeResult:
    """Restart the search many times at one date and measure the headroom."""
    from dataclasses import replace

    standard = replace(params, pso_blend=blend)
    long_run = replace(
        standard,
        num_particles=REFERENCE_PARTICLES,
        max_iter=REFERENCE_ITERATIONS,
    )

    # One generator for the whole date so restarts are independent of each other
    # but the whole probe is reproducible from `seed`.
    rng = np.random.default_rng(seed)

    def objectives(config, count: int) -> np.ndarray:
        optimizer = FPSOOptimizer(config)
        return np.array([
            optimizer.solve(moments, weights_prev, rng=rng).objective_value
            for _ in range(count)
        ])

    standard_values = objectives(standard, n_restarts)
    long_values = objectives(long_run, n_long)

    best_standard = float(standard_values.max())
    median_standard = float(np.median(standard_values))
    best_long = float(long_values.max())

    return ProbeResult(
        date=date,
        blend=blend,
        best_standard=best_standard,
        median_standard=median_standard,
        best_long=best_long,
        headroom=_relative_gap(best_long, best_standard),
        typical_shortfall=_relative_gap(best_long, median_standard),
        long_run_wins=best_long > best_standard,
    )


def run_probe(
    config_path: Path,
    cache_dir: Path,
    *,
    n_dates: int,
    n_restarts: int,
    n_long: int,
    blends: tuple[float, ...],
    seed: int = 0,
) -> pd.DataFrame:
    """Sweep blend settings over evenly spaced rebalance dates."""
    config = load_experiment_config(config_path)
    panel = ReturnPanel.from_parquet(cache_dir)

    in_window = panel.dates[
        (panel.dates >= pd.Timestamp(config.schedule.start))
        & (panel.dates <= pd.Timestamp(config.schedule.end))
    ]
    # Evenly spaced across the window rather than random, so the probe covers
    # calm and stressed periods alike without depending on a draw.
    picks = in_window[np.linspace(0, len(in_window) - 1, n_dates).astype(int)]

    universe_provider = build_universe_provider(config.data, panel)
    # The plain sample estimator the unconditional arm uses. The probe is about
    # the *search*, so the moment mechanism is deliberately held fixed rather than
    # inherited from whichever arm the config happens to describe.
    estimator = build_moment_estimator("sample", config.schedule.min_observations)
    lookback = config.schedule.estimation_window_days

    rows = []
    for index, date in enumerate(picks):
        universe = universe_provider.at(date)
        window = panel.returns_between(
            date - pd.Timedelta(days=lookback), date, assets=universe
        )
        moments = estimator.estimate(window)
        # An equal-weight incumbent: the turnover term needs a reference, and any
        # fixed one keeps the comparison across blends like-for-like.
        weights_prev = np.full(len(moments.assets), 1.0 / len(moments.assets))

        for blend in blends:
            rows.append(asdict(probe_date(
                moments, weights_prev, config.base_params, blend,
                date=date, n_restarts=n_restarts, n_long=n_long,
                seed=seed + 1000 * index,
            )))
            print(f"  {date.date()}  blend={blend:<4} "
                  f"headroom={rows[-1]['headroom']:+.1%} "
                  f"typical shortfall={rows[-1]['typical_shortfall']:+.1%}")
    return pd.DataFrame(rows)


def summarize(probe: pd.DataFrame) -> pd.DataFrame:
    """Per-blend summary: does more budget help, and by how much?

    **Report the medians.** These are ratios against an objective that varies by
    an order of magnitude across the window — 0.03 at 2019-05-28 against 0.53 at
    2024-12-31 — so a mean over dates is dominated by whichever month had the
    smallest denominator. At 2019-05-28 a typical pure-firefly run reached 0.0067
    against a reference of 0.031, a genuine 4.6x shortfall that is correctly
    reported as 358% and that single-handedly moves that arm's *mean* shortfall
    from 35% to 88%.

    That date is not an artifact and is not excluded: the relative gap is real,
    and dropping the hardest month would flatter the search. It is a reason to
    read the median, which is why the median columns come first. `worst_headroom`
    is reported for the same reason in the other direction — pure firefly has two
    dates where the best of many standard restarts *beat* the reference run, and a
    mean alone would hide that.
    """
    usable = probe[np.isfinite(probe["headroom"])]
    summary = usable.groupby("blend").agg(
        dates=("date", "nunique"),
        reference_wins=("long_run_wins", "sum"),
        median_headroom=("headroom", "median"),
        worst_headroom=("headroom", "min"),
        mean_headroom=("headroom", "mean"),
        median_typical_shortfall=("typical_shortfall", "median"),
        mean_typical_shortfall=("typical_shortfall", "mean"),
    )
    summary.insert(0, "algorithm", [BLEND_NAMES.get(b, f"blend={b}") for b in summary.index])
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--dates", type=int, default=12)
    parser.add_argument("--restarts", type=int, default=200)
    parser.add_argument("--long-runs", type=int, default=25)
    parser.add_argument("--blends", type=float, nargs="*", default=list(DEFAULT_BLENDS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("paper/tables"))
    parser.add_argument("--quick", action="store_true",
                        help="4 dates, 20 restarts, 3 long runs — checks it runs.")
    args = parser.parse_args(argv)

    n_dates, n_restarts, n_long = args.dates, args.restarts, args.long_runs
    if args.quick:
        n_dates, n_restarts, n_long = 4, 20, 3

    probe = run_probe(
        args.config, args.cache_dir,
        n_dates=n_dates, n_restarts=n_restarts, n_long=n_long,
        blends=tuple(args.blends), seed=args.seed,
    )
    summary = summarize(probe)

    args.out.mkdir(parents=True, exist_ok=True)
    probe.to_csv(args.out / "convergence_probe.csv", index=False)
    summary.to_csv(args.out / "convergence_summary.csv")

    print("\n=== Headroom of a reference-budget run over the best of many standard runs ===")
    print(summary.to_string())
    print(f"\nWritten to {args.out}/convergence_probe.csv and convergence_summary.csv")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
