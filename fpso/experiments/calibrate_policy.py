"""Pre-sample calibration of the regime -> parameter policy, then freeze.

    python -m fpso.experiments.calibrate_policy

"Where did those numbers come from?" is the first question a reviewer asks about
a regime-conditioned parameter map. This script gives an auditable answer:
coordinate search on a **2008-2010 pre-sample holdout**, which spans the global
financial crisis, the recovery and a calm stretch, so all three regimes are
represented. (It opens in 2008 rather than 2005 because the detector needs about
three years of panel history to warm up — see `main` for the arithmetic.)
The result is written to ``configs/regime_policy.frozen.yaml`` with a timestamp
and the git SHA, and is then **frozen** — the 2011-2024 evaluation sample is
never touched here.

Coordinate search rather than a full grid: the map has ~12 free values, so a grid
is not affordable, and coordinate descent from the a-priori starting point makes
the search path itself reportable. Each candidate is scored by mean Sharpe across
a small number of seeds on the holdout, so the objective is the same quantity the
paper reports.

If you would rather make no data-driven choice at all, the a-priori values in
``configs/regime_policy.yaml`` are the alternative — weaker, but unimpeachable.
Whichever file the arms point at is recorded in every result manifest.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from fpso.config.loader import load_experiment_config
from fpso.config.schema import ExperimentConfig, RegimeLabel
from fpso.data.panel import ReturnPanel
from fpso.evaluation.metrics import summarize
from fpso.experiments.runner import load_panel, run_arm

# The candidate values searched per (regime, parameter). Each list brackets the
# a-priori value from configs/regime_policy.yaml, so the search can move in
# either direction but stays inside economically sensible bounds.
SEARCH_GRID: dict[str, dict[str, list[float]]] = {
    "CALM": {
        "alpha": [0.05, 0.10, 0.15],
        "lambda_v": [0.40, 0.60, 0.80],
        "lambda_t": [0.02, 0.03, 0.05],
        "max_assets": [20, 25, 30],
    },
    "TURBULENT": {
        "alpha": [0.25, 0.30, 0.40],
        "gamma": [1.30, 1.60, 2.00],
        "lambda_v": [1.30, 1.60, 2.00],
    },
    "CRISIS": {
        "alpha": [0.30, 0.35, 0.45],
        "lambda_v": [2.00, 3.00, 4.00],
        "lambda_t": [0.000, 0.005, 0.010],
        "max_assets": [8, 12, 15],
        "max_weight": [0.08, 0.10, 0.12],
    },
}


def calibrate(
    config: ExperimentConfig,
    panel: ReturnPanel,
    n_seeds: int = 3,
    verbose: bool = True,
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    """Coordinate-descend the override map on the holdout; return it and the trace."""
    overrides = {
        label: dict(config.regime.overrides.get(label, {}))
        for label in (state.name for state in RegimeLabel)
    }
    trace: list[dict] = []

    best_score = _score(config, panel, overrides, n_seeds)
    if verbose:
        print(f"a-priori starting point: holdout Sharpe = {best_score:.4f}\n")

    for regime, parameters in SEARCH_GRID.items():
        for parameter, candidates in parameters.items():
            incumbent = overrides[regime].get(parameter)
            scores = {}
            for value in candidates:
                trial = {r: dict(v) for r, v in overrides.items()}
                trial[regime][parameter] = value
                if not _is_feasible(config, trial[regime]):
                    # e.g. max_assets=8 with max_weight=0.10 gives 0.8 < 1, so no
                    # fully-invested portfolio exists. Skipping keeps the search
                    # inside the feasible region instead of crashing mid-run.
                    continue
                scores[value] = _score(config, panel, trial, n_seeds)

            if not scores:
                continue
            winner = max(scores, key=scores.get)
            improvement = scores[winner] - best_score
            if scores[winner] > best_score:
                overrides[regime][parameter] = winner
                best_score = scores[winner]

            trace.append({
                "regime": regime,
                "parameter": parameter,
                "incumbent": incumbent,
                "candidates": {str(k): round(v, 5) for k, v in scores.items()},
                "selected": overrides[regime].get(parameter),
                "holdout_sharpe": round(best_score, 5),
            })
            if verbose:
                print(
                    f"  {regime:<10} {parameter:<12} "
                    f"{incumbent} -> {overrides[regime].get(parameter)}  "
                    f"(Sharpe {best_score:.4f}, {improvement:+.4f})"
                )

    return overrides, trace


def _is_feasible(config: ExperimentConfig, regime_overrides: dict[str, float]) -> bool:
    """Whether a candidate regime block still admits a fully-invested portfolio."""
    params = config.base_params.with_overrides(regime_overrides)
    return params.max_assets * params.max_weight >= 1.0


def _score(
    config: ExperimentConfig,
    panel: ReturnPanel,
    overrides: dict[str, dict[str, float]],
    n_seeds: int,
) -> float:
    """Mean Sharpe over the holdout at the mid transaction-cost rate."""
    candidate = replace(
        config,
        seeds=tuple(range(n_seeds)),
        regime=replace(config.regime, overrides=overrides),
    )
    arm = run_arm(candidate, panel, save=False, verbose=False)
    rates = sorted(candidate.transaction_cost_rates)
    cost = rates[min(1, len(rates) - 1)]
    return float(np.mean([summarize(run.returns(cost)).sharpe for run in arm.runs]))


def write_frozen_config(
    overrides: dict[str, dict[str, float]],
    trace: list[dict],
    holdout: tuple[str, str],
    output: Path,
) -> None:
    """Emit the frozen policy file, with provenance in a header comment."""
    header = (
        "# FROZEN regime -> parameter policy.\n"
        "#\n"
        "# Produced by `python -m fpso.experiments.calibrate_policy` via coordinate\n"
        f"# search on the {holdout[0]} .. {holdout[1]} pre-sample holdout. The\n"
        "# 2011-2024 evaluation sample was NOT used to select any value below.\n"
        "#\n"
        f"# frozen_at : {datetime.now(UTC).isoformat()}\n"
        f"# git_sha   : {_git_sha()}\n"
        "#\n"
        "# Do not hand-edit. Re-run the calibration and commit the new file, so the\n"
        "# freeze stays auditable.\n"
    )
    payload = {
        "extends": "base.yaml",
        "name": "regime_policy_frozen",
        "regime": {"detector": "hmm", "overrides": overrides},
    }
    output.write_text(header + yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
    output.with_suffix(".trace.json").write_text(json.dumps(trace, indent=2))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/regime_policy.yaml")
    parser.add_argument("--output", default="configs/regime_policy.frozen.yaml")
    # The holdout opens in 2008, not 2005, because the detector needs warm-up: the
    # 252-day realized-vol and drawdown features plus 252 observations of expanding
    # standardization plus 252 observations to identify the HMM means the first
    # usable label arrives roughly three years after the panel starts. Calibrating
    # from 2005 would have spent most of the window in burn-in with the base
    # parameters, so the search would have been comparing identical configurations.
    # 2008-2010 still spans the crisis, the recovery and a calm stretch, and still
    # ends before the 2011 evaluation window opens.
    parser.add_argument("--holdout-start", default="2008-01-01")
    parser.add_argument("--holdout-end", default="2010-12-31")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--particles", type=int, default=20,
        help="Reduced swarm size: calibration ranks configurations, it does not "
             "produce reported numbers.",
    )
    parser.add_argument("--max-iter", type=int, default=30)
    args = parser.parse_args()

    base = load_experiment_config(args.config)
    config = replace(
        base,
        name="calibration",
        base_params=replace(base.base_params, num_particles=args.particles, max_iter=args.max_iter),
        schedule=replace(base.schedule, start=args.holdout_start, end=args.holdout_end),
        # Burn-in must fit inside the holdout, or the policy never fires and the
        # search would compare identical configurations.
        regime=replace(base.regime, burn_in_months=6, refit_months=12),
    )

    print(f"Calibrating on holdout {args.holdout_start} .. {args.holdout_end}")
    panel = load_panel(config)
    overrides, trace = calibrate(config, panel, n_seeds=args.seeds)

    output = Path(args.output)
    write_frozen_config(overrides, trace, (args.holdout_start, args.holdout_end), output)
    print(f"\nFrozen policy written to {output}")
    print(f"Search trace written to {output.with_suffix('.trace.json')}")
    print("\nTo evaluate the calibrated variant, point the regime arms at this file:")
    print(f"  extends: ../{output.name}")


if __name__ == "__main__":
    main()
