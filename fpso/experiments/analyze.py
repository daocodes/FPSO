"""Turn finished runs into the paper's tables and figures.

    python -m fpso.experiments.analyze --results results --figures paper/figures

Produces:

* ``headline_metrics.csv``      — one row per arm, seed-averaged.
* ``regime_conditional.csv``    — per-arm, per-regime metrics.
* ``arm_comparisons.csv``       — paired differences vs. the static baseline,
  with stationary-bootstrap intervals, the Ledoit-Wolf test, and Holm-adjusted
  p-values.
* ``cost_sensitivity.csv``      — the metric sweep across transaction-cost rates.
* Figures F1-F9 in the figures directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fpso.backtest.results import ArmResult, BacktestResult
from fpso.evaluation import figures as fig
from fpso.evaluation.conviction import conviction_profile, inclusion_frequency
from fpso.evaluation.metrics import metrics_table, summarize, summarize_by_regime
from fpso.evaluation.statistics import (
    StationaryBootstrap,
    holm_adjust,
    ledoit_wolf_sharpe_test,
    max_drawdown_statistic,
    volatility_statistic,
)

BASELINE_ARM = "static"
HEADLINE_ARM = "regime_hmm"


def load_results(results_dir: str | Path) -> dict[str, ArmResult]:
    """Load every ``<arm>/seed=<n>/`` directory under `results_dir`."""
    root = Path(results_dir)
    arms: dict[str, ArmResult] = {}
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = [
            BacktestResult.load(seed_dir)
            for seed_dir in sorted(arm_dir.glob("seed=*"))
        ]
        if runs:
            arms[arm_dir.name] = ArmResult(arm=arm_dir.name, runs=runs)
    if not arms:
        raise FileNotFoundError(
            f"No results under {root.resolve()}. Run the experiments first: "
            "python -m fpso.experiments.run_study"
        )
    return arms


def headline_table(arms: dict[str, ArmResult], cost: float) -> pd.DataFrame:
    """Seed-averaged performance for every arm at one transaction-cost rate."""
    table = metrics_table({name: arm.mean_returns(cost) for name, arm in arms.items()})
    table["mean_turnover"] = pd.Series({name: arm.mean_turnover for name, arm in arms.items()})
    return table.sort_values("sharpe", ascending=False)


def regime_conditional_tables(
    arms: dict[str, ArmResult], cost: float
) -> dict[str, pd.DataFrame]:
    """Per-regime metrics for each arm, attributed by the *acting* regime label.

    All arms are attributed using the headline arm's labels so that the rows
    compare like with like: "how did the static portfolio do during the months
    the detector called a crisis" is the question the paper is asking.
    """
    if HEADLINE_ARM not in arms:
        return {}
    reference_labels = arms[HEADLINE_ARM].runs[0].regime_labels

    return {
        name: summarize_by_regime(arm.mean_returns(cost), reference_labels)
        for name, arm in arms.items()
    }


def comparison_table(
    arms: dict[str, ArmResult],
    cost: float,
    baseline: str = BASELINE_ARM,
    n_resamples: int = 2000,
) -> pd.DataFrame:
    """Paired differences of every arm against the static baseline.

    Every comparison uses matched seeds and the same bootstrap block indices, so
    the intervals are on the *paired* difference rather than on two independent
    estimates. Holm adjustment is applied across the family of comparisons.
    """
    if baseline not in arms:
        raise KeyError(f"Baseline arm '{baseline}' not found in {sorted(arms)}.")

    control = arms[baseline].mean_returns(cost)
    bootstrap = StationaryBootstrap(n_resamples=n_resamples)

    rows, raw_p_values = {}, {}
    for name, arm in arms.items():
        if name == baseline:
            continue
        treatment = arm.mean_returns(cost)

        # Keyed by arm and statistic so each interval has its own fixed stream and
        # does not depend on where the arm falls in the iteration order.
        sharpe = bootstrap.paired_sharpe_difference(
            treatment, control, key=f"{name}|{baseline}|sharpe|{cost}"
        )
        drawdown = bootstrap.paired_statistic(
            treatment, control, max_drawdown_statistic,
            key=f"{name}|{baseline}|drawdown|{cost}",
        )
        volatility = bootstrap.paired_statistic(
            treatment, control, volatility_statistic,
            key=f"{name}|{baseline}|volatility|{cost}",
        )
        lw_difference, lw_p = ledoit_wolf_sharpe_test(treatment, control)

        raw_p_values[name] = sharpe.p_value
        rows[name] = {
            "delta_sharpe": sharpe.statistic,
            "delta_sharpe_lo": sharpe.lower,
            "delta_sharpe_hi": sharpe.upper,
            "delta_sharpe_p": sharpe.p_value,
            "delta_max_drawdown": drawdown.statistic,
            "delta_max_drawdown_p": drawdown.p_value,
            "delta_volatility": volatility.statistic,
            "delta_volatility_p": volatility.p_value,
            "ledoit_wolf_delta_sharpe": lw_difference,
            "ledoit_wolf_p": lw_p,
            "delta_turnover": arm.mean_turnover - arms[baseline].mean_turnover,
        }

    table = pd.DataFrame(rows).T
    table["delta_sharpe_p_holm"] = pd.Series(holm_adjust(raw_p_values))
    return table.sort_values("delta_sharpe", ascending=False)


def cost_sensitivity_table(arms: dict[str, ArmResult]) -> pd.DataFrame:
    """Sharpe, return and drawdown for every arm at every simulated cost rate."""
    rows = []
    for name, arm in arms.items():
        for rate in sorted(arm.runs[0].returns_by_cost):
            metrics = summarize(arm.mean_returns(rate))
            rows.append({"arm": name, "transaction_cost": rate, **metrics.as_dict()})
    return pd.DataFrame(rows).set_index(["arm", "transaction_cost"])


def build_figures(
    arms: dict[str, ArmResult],
    market_returns: pd.Series | None,
    cost: float,
    figure_dir: Path,
    comparisons: pd.DataFrame | None = None,
) -> list[Path]:
    """Render F1-F9 from loaded results."""
    paths: list[Path] = []
    headline = arms.get(HEADLINE_ARM)
    baseline = arms.get(BASELINE_ARM)

    if headline is not None:
        labels = headline.runs[0].regime_labels
        if market_returns is not None:
            paths.append(fig.plot_market_regimes(market_returns, labels, figure_dir))

        if baseline is not None:
            paths.append(
                fig.plot_cumulative_and_drawdown(
                    {"Static FPSO": baseline.mean_returns(cost),
                     "Regime FPSO": headline.mean_returns(cost)},
                    regime_labels=labels,
                    directory=figure_dir,
                )
            )
            conditional = regime_conditional_tables(
                {BASELINE_ARM: baseline, HEADLINE_ARM: headline}, cost
            )
            paths.append(
                fig.plot_regime_conditional_bars(
                    {"Static FPSO": conditional[BASELINE_ARM],
                     "Regime FPSO": conditional[HEADLINE_ARM]},
                    directory=figure_dir,
                )
            )
            paths.append(
                fig.plot_cost_sensitivity(
                    {"Static FPSO": _by_cost(baseline),
                     "Regime FPSO": _by_cost(headline)},
                    directory=figure_dir,
                )
            )

        transition, dwell = _regime_dynamics(labels)
        paths.append(fig.plot_transition_and_dwell(transition, dwell, figure_dir))

    if comparisons is not None and not comparisons.empty:
        paths.append(fig.plot_effect_sizes(comparisons, BASELINE_ARM, figure_dir))

    mechanisms = _mechanism_comparison(arms)
    if mechanisms is not None:
        paths.append(fig.plot_cost_conditional_effects(mechanisms, BASELINE_ARM, figure_dir))

    turnover, effect = _turnover_vs_effect(arms)
    if turnover is not None:
        paths.append(fig.plot_turnover_explains_effect(turnover, effect, figure_dir))

    matched = _turnover_matched_panels(arms)
    if matched:
        paths.append(fig.plot_turnover_matched_comparison(matched, figure_dir))

    seeds, effects = _noise_floor(arms, cost)
    if seeds is not None:
        paths.append(
            fig.plot_noise_floor(seeds, effects, figure_dir, cost_label=f"{cost:.1%}")
        )

    conviction = _conviction(arms)
    if conviction is not None:
        frequency, profile = conviction
        paths.append(fig.plot_conviction(frequency, profile, figure_dir))

    # A schematic, so it depends on no result and is always emitted.
    paths.append(fig.plot_injection_points(figure_dir))
    return paths


def _conviction(arms: dict[str, ArmResult]):
    """Inclusion frequency and its profile for the baseline arm, or None.

    Measured on the unconditional arm so the result describes the optimizer
    rather than any mechanism layered on top of it.
    """
    baseline = arms.get(BASELINE_ARM)
    if baseline is None or len(baseline.runs) < 2:
        return None
    frequency = inclusion_frequency(
        {run.seed: run.target_weights for run in baseline.runs}
    )
    return frequency, conviction_profile(frequency)


# The comparisons the paper reports, drawn against the baseline arm's own seed
# spread in F10. Chosen to span the study: one arm per injection point, the
# decisive control, and the algorithmic ablation.
NOISE_FLOOR_EFFECTS = (
    ("Regime preferences", "regime_hmm"),
    ("Regime beliefs", "moments_hmm"),
    ("Regime expected returns", "tilt_hmm"),
    ("Turnover-matched control", "static_matched_regime_hmm"),
    ("PSO-term ablation", "ablation_no_pso"),
)


def _noise_floor(
    arms: dict[str, ArmResult], cost: float
) -> tuple[pd.Series | None, pd.Series]:
    """Per-seed Sharpe of the baseline arm, and each effect measured against it.

    The seed spread is computed on the baseline alone — deliberately, because the
    claim is about the dispersion of the *instrument*, not about any treatment. A
    reader who doubts an effect should be able to ask what the same arm does when
    only the seed changes, and F10 is that question answered on one axis.
    """
    baseline = arms.get(BASELINE_ARM)
    if baseline is None or len(baseline.runs) < 2:
        return None, pd.Series(dtype=float)

    per_seed = pd.Series({
        run.seed: summarize(run.returns(cost)).sharpe for run in baseline.runs
    }).sort_index()
    reference = summarize(baseline.mean_returns(cost)).sharpe

    effects = {}
    for label, arm_name in NOISE_FLOOR_EFFECTS:
        arm = arms.get(arm_name)
        if arm is None:
            continue
        effects[label] = summarize(arm.mean_returns(cost)).sharpe - reference
    return per_seed, pd.Series(effects)


# Each mechanism paired with the static arm whose turnover penalty was solved to
# match it. See fpso.experiments.match_turnover.
#
# The three entries are the three points at which a market-state signal can enter
# a mean-variance optimizer: its preferences (the objective's weights), its
# beliefs (Sigma), and its forecasts (mu). Together they exhaust the injection
# points, which is what lets the paper make a claim about regime conditioning
# rather than about one implementation of it.
MATCHED_PAIRS = (
    ("Preferences", "regime_hmm", "static_matched_regime_hmm"),
    ("Beliefs", "moments_hmm", "static_matched_moments_hmm"),
    ("Expected returns", "tilt_hmm", "static_matched_tilt_hmm"),
)


def turnover_matched_table(
    arms: dict[str, ArmResult], n_resamples: int = 5000
) -> pd.DataFrame | None:
    """The paper's Table 2: each mechanism against its turnover-matched control.

    This is the study's decisive comparison, so it is emitted as a table rather
    than living only inside figure F9 — every number quoted in the paper should
    be regenerable from `paper/tables/`, and this one previously was not.

    Reported at every cost level, because the argument is about the *shape* of
    the effect across costs, not its level at any single one.
    """
    rows = {}
    bootstrap = StationaryBootstrap(n_resamples=n_resamples)
    for label, treatment_arm, matched_arm in MATCHED_PAIRS:
        if not {treatment_arm, matched_arm} <= set(arms):
            continue
        treat, matched = arms[treatment_arm], arms[matched_arm]
        for cost in sorted(treat.runs[0].returns_by_cost):
            if cost not in matched.runs[0].returns_by_cost:
                continue
            key = f"{treatment_arm}|{matched_arm}|sharpe|{cost}"
            interval = bootstrap.paired_sharpe_difference(
                treat.mean_returns(cost), matched.mean_returns(cost), key=key
            )
            _, lw_p = ledoit_wolf_sharpe_test(
                treat.mean_returns(cost), matched.mean_returns(cost)
            )
            rows[(label, cost)] = {
                "treatment": treatment_arm,
                "matched_control": matched_arm,
                "turnover_treatment": treat.mean_turnover,
                "turnover_control": matched.mean_turnover,
                "regime_effect": interval.statistic,
                "ci_lo": interval.lower,
                "ci_hi": interval.upper,
                "p_bootstrap": interval.p_value,
                "p_ledoit_wolf": lw_p,
            }
    if not rows:
        return None
    table = pd.DataFrame(rows).T
    table.index.names = ["mechanism", "transaction_cost"]
    return table


def _turnover_matched_panels(arms: dict[str, ArmResult]):
    """Cost sweeps for F9, or an empty dict if the matched controls are absent."""
    panels = {}
    for label, treatment, matched in MATCHED_PAIRS:
        if not {treatment, matched, BASELINE_ARM} <= set(arms):
            continue
        panels[label] = {
            "treatment": _by_cost(arms[treatment]),
            "unmatched": _by_cost(arms[BASELINE_ARM]),
            "matched": _by_cost(arms[matched]),
        }
    return panels


# Rebalance-frequency ablations are excluded from F8: turnover per *rebalance* is
# not comparable when one arm rebalances 14 times and the others 168.
FREQUENCY_ABLATIONS = ("ablation_annual",)
TURNOVER_PLOT_COST = 0.015


def _turnover_vs_effect(arms: dict[str, ArmResult]):
    """Mean turnover and net-of-cost effect per arm, for F8."""
    if BASELINE_ARM not in arms:
        return None, None
    baseline = summarize(arms[BASELINE_ARM].mean_returns(TURNOVER_PLOT_COST)).sharpe

    usable = {
        name: arm for name, arm in arms.items()
        if name not in FREQUENCY_ABLATIONS
        and TURNOVER_PLOT_COST in arm.runs[0].returns_by_cost
    }
    if len(usable) < 4:
        return None, None

    turnover = pd.Series({n: a.mean_turnover for n, a in usable.items()})
    effect = pd.Series({
        n: summarize(a.mean_returns(TURNOVER_PLOT_COST)).sharpe - baseline
        for n, a in usable.items()
    })
    return turnover, effect


# The arms F7 contrasts: the baseline, one arm per injection point, and the
# control that separates mechanism 2's regime weighting from its longer window.
MECHANISM_ARMS = (
    BASELINE_ARM,
    "regime_hmm",
    "moments_hmm",
    "moments_longwindow",
    "tilt_hmm",
)


def _mechanism_comparison(arms: dict[str, ArmResult]):
    """Cost sweeps for the arms F7 draws, or None if the matrix lacks them."""
    available = [name for name in MECHANISM_ARMS if name in arms]
    if BASELINE_ARM not in available or len(available) < 2:
        return None
    return {name: _by_cost(arms[name]) for name in available}


def _by_cost(arm: ArmResult) -> dict[float, pd.Series]:
    """Seed-averaged path at each simulated transaction-cost rate."""
    return {rate: arm.mean_returns(rate) for rate in sorted(arm.runs[0].returns_by_cost)}


def _regime_dynamics(labels: pd.Series) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """Empirical rebalance-frequency transition matrix and dwell-time episodes.

    Estimated from the realised acting-label sequence rather than read off the
    fitted HMM, so the figure describes the regimes the strategy actually
    experienced (post-smoothing, post-burn-in).
    """
    import numpy as np

    from fpso.evaluation.style import REGIME_ORDER

    index = {name: i for i, name in enumerate(REGIME_ORDER)}
    counts = np.zeros((len(REGIME_ORDER), len(REGIME_ORDER)))
    for previous, current in zip(labels.iloc[:-1], labels.iloc[1:], strict=True):
        if previous in index and current in index:
            counts[index[previous], index[current]] += 1
    row_totals = counts.sum(axis=1, keepdims=True)
    transition = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)

    dwell: dict[str, list[int]] = {name: [] for name in REGIME_ORDER}
    if len(labels):
        current, run_length = labels.iloc[0], 1
        for label in labels.iloc[1:]:
            if label == current:
                run_length += 1
            else:
                dwell.setdefault(current, []).append(run_length)
                current, run_length = label, 1
        dwell.setdefault(current, []).append(run_length)
    return transition, dwell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--figures", default="paper/figures")
    parser.add_argument("--tables", default="paper/tables")
    parser.add_argument(
        "--cost", type=float, default=0.005,
        help="Transaction-cost rate for the headline tables and figures.",
    )
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    arms = load_results(args.results)
    print(f"Loaded {len(arms)} arms: {', '.join(sorted(arms))}")

    table_dir = Path(args.tables)
    table_dir.mkdir(parents=True, exist_ok=True)

    headline = headline_table(arms, args.cost)
    headline.to_csv(table_dir / "headline_metrics.csv")
    print(f"\n=== Headline metrics (tc={args.cost:.2%}) ===\n{headline.round(4)}")

    comparisons = None
    if BASELINE_ARM in arms and len(arms) > 1:
        comparisons = comparison_table(arms, args.cost, n_resamples=args.bootstrap_resamples)
        comparisons.to_csv(table_dir / "arm_comparisons.csv")
        print(f"\n=== Paired differences vs. '{BASELINE_ARM}' ===\n{comparisons.round(4)}")

    conditional = regime_conditional_tables(arms, args.cost)
    if conditional:
        stacked = pd.concat(conditional, names=["arm"])
        stacked.to_csv(table_dir / "regime_conditional.csv")
        print(f"\n=== Regime-conditional metrics ===\n{stacked.round(4)}")

    cost_sweep = cost_sensitivity_table(arms)
    cost_sweep.to_csv(table_dir / "cost_sensitivity.csv")

    matched = turnover_matched_table(arms, n_resamples=args.bootstrap_resamples)
    if matched is not None:
        matched.to_csv(table_dir / "turnover_matched.csv")
        print(f"\n=== Regime effect vs. turnover-matched control ===\n{matched.round(4)}")

    market = _load_market(args.cache_dir, arms)
    paths = build_figures(arms, market, args.cost, Path(args.figures), comparisons)
    print("\n=== Figures ===")
    for path in paths:
        print(f"  {path}")


def _load_market(cache_dir: str, arms: dict[str, ArmResult]) -> pd.Series | None:
    """Market series over the evaluation window, for F1; None if the cache is absent."""
    from fpso.data.panel import ReturnPanel

    try:
        panel = ReturnPanel.from_parquet(cache_dir)
    except FileNotFoundError:
        print(f"  (no panel cache at {cache_dir}; skipping F1)")
        return None

    any_arm = next(iter(arms.values()))
    span = any_arm.runs[0].returns(min(any_arm.runs[0].returns_by_cost)).index
    return panel.market.loc[span[0]:span[-1]]


if __name__ == "__main__":
    main()
