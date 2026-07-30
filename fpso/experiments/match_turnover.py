"""Solve for the turnover-matched control arm.

    python -m fpso.experiments.match_turnover --target-arm regime_hmm

The study's central finding is that net-of-cost performance tracks realized
turnover across every arm. The obvious objection is that regime conditioning is
*supposed* to change how much you trade, so a turnover reduction is the mechanism
working rather than a confound. This script builds the arm that settles it: plain
static FPSO with its turnover penalty `lambda_t` raised until it trades exactly as
much as the regime arm does.

If the matched control then matches or beats the regime arm, the regime signal
adds nothing beyond the turnover it happens to induce — and the paper's
recommendation (report turnover-matched comparisons) is demonstrated rather than
asserted.

**What is being matched, and why that is legitimate.** The target is *turnover*,
a design covariate, chosen from the treatment arm's realized trading and never
from its Sharpe. That is the same logic as matching a control group on age: it
removes a nuisance difference so the remaining comparison is about the treatment.
Calibrating on the *outcome* would be circular; calibrating on turnover is not.
Like the shuffled and oracle arms this is a control, not a deployable strategy,
and should be described as one wherever it appears.

Turnover is monotonically decreasing in `lambda_t` — a heavier penalty on
||w - w_prev||_1 buys less trading — so a bisection on the bracket is enough.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from fpso.backtest.results import BacktestResult
from fpso.config.loader import load_experiment_config
from fpso.config.schema import ExperimentConfig
from fpso.data.panel import ReturnPanel
from fpso.experiments.runner import load_panel, run_arm

DEFAULT_BRACKET = (0.0, 0.5)
"""lambda_t search range. The upper end is far past any sensible value, so the
bracket is widened rather than the search failing."""


def target_turnover(arm: str, results_dir: str | Path) -> float:
    """Mean one-way turnover per rebalance of a completed arm."""
    seeds = sorted(Path(results_dir).joinpath(arm).glob("seed=*"))
    if not seeds:
        raise FileNotFoundError(
            f"No results for arm '{arm}' under {results_dir}. Run it first."
        )
    values = []
    for seed_dir in seeds:
        result = BacktestResult.load(seed_dir)
        turnover = result.turnover_by_cost[min(result.turnover_by_cost)]
        values.append(float(turnover.mean()))
    return float(np.mean(values))


def measure_turnover(
    config: ExperimentConfig, panel: ReturnPanel, lambda_t: float, n_seeds: int
) -> float:
    """Mean turnover of `config` when its turnover penalty is set to `lambda_t`."""
    candidate = replace(
        config,
        seeds=tuple(range(n_seeds)),
        base_params=replace(config.base_params, lambda_t=lambda_t),
    )
    arm = run_arm(candidate, panel, save=False, verbose=False)
    return float(np.mean([
        float(run.turnover_by_cost[min(run.turnover_by_cost)].mean()) for run in arm.runs
    ]))


def solve_lambda_t(
    config: ExperimentConfig,
    panel: ReturnPanel,
    target: float,
    n_seeds: int = 5,
    tolerance: float = 0.004,
    max_iterations: int = 9,
    verbose: bool = True,
) -> tuple[float, float, list[dict]]:
    """Bisect `lambda_t` until mean turnover is within `tolerance` of `target`.

    Returns the solved penalty, the turnover it achieves, and the search trace.
    """
    low, high = DEFAULT_BRACKET
    trace: list[dict] = []

    turnover_at_high = measure_turnover(config, panel, high, n_seeds)
    if turnover_at_high > target:
        raise ValueError(
            f"Even lambda_t={high} leaves turnover at {turnover_at_high:.4f}, above "
            f"the target {target:.4f}. Widen DEFAULT_BRACKET."
        )

    best = (high, turnover_at_high)
    for iteration in range(max_iterations):
        midpoint = 0.5 * (low + high)
        turnover = measure_turnover(config, panel, midpoint, n_seeds)
        trace.append({
            "iteration": iteration,
            "lambda_t": round(midpoint, 6),
            "turnover": round(turnover, 5),
            "target": round(target, 5),
        })
        if verbose:
            print(
                f"  iter {iteration}: lambda_t={midpoint:.5f} -> turnover={turnover:.4f} "
                f"(target {target:.4f}, gap {turnover - target:+.4f})",
                flush=True,
            )

        if abs(turnover - target) < abs(best[1] - target):
            best = (midpoint, turnover)
        if abs(turnover - target) < tolerance:
            return midpoint, turnover, trace

        # Higher penalty => less trading, so overshooting downward means backing off.
        if turnover > target:
            low = midpoint
        else:
            high = midpoint

    if verbose:
        print(f"  bracket exhausted; using closest point lambda_t={best[0]:.5f}")
    return best[0], best[1], trace


def write_matched_config(
    lambda_t: float,
    achieved: float,
    target: float,
    target_arm: str,
    output: Path,
) -> None:
    """Emit the matched control's arm config with its provenance in the header."""
    output.write_text(
        f"""# TURNOVER-MATCHED CONTROL for `{target_arm}`.
#
# Plain static FPSO with its turnover penalty raised until it trades as much as
# `{target_arm}` does. Solved by bisection on lambda_t against that arm's realized
# turnover; see `python -m fpso.experiments.match_turnover`.
#
#   target turnover   : {target:.4f}  (from {target_arm})
#   achieved turnover : {achieved:.4f}
#   solved lambda_t   : {lambda_t:.5f}  (baseline 0.01)
#   solved_at         : {datetime.now(UTC).isoformat()}
#
# This is a CONTROL, not a strategy. It is matched on turnover — a design
# covariate taken from the treatment arm's trading, never from its performance —
# so that the remaining difference between the two arms is attributable to the
# regime signal alone. Describe it as a control wherever it appears.
extends: ../base.yaml
name: {output.stem}
base_params:
  lambda_t: {lambda_t:.5f}
regime:
  detector: none
  overrides: {{}}
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-arm", default="regime_hmm")
    parser.add_argument("--results", default="results")
    parser.add_argument("--config", default="configs/arms/static.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=0.004)
    args = parser.parse_args()

    target = target_turnover(args.target_arm, args.results)
    base = load_experiment_config(args.config)
    config = replace(base, name=f"match_{args.target_arm}", results_dir=args.results)

    print(f"Matching static FPSO turnover to '{args.target_arm}' ({target:.4f})")
    panel = load_panel(config)
    lambda_t, achieved, _ = solve_lambda_t(
        config, panel, target, n_seeds=args.seeds, tolerance=args.tolerance
    )

    output = Path(args.output or f"configs/arms/static_matched_{args.target_arm}.yaml")
    write_matched_config(lambda_t, achieved, target, args.target_arm, output)
    print(f"\nSolved lambda_t = {lambda_t:.5f} -> turnover {achieved:.4f}")
    print(f"Wrote {output}\n\nRun it with:")
    print(f"  python -m fpso.experiments.run_study --arms {output.stem}")


if __name__ == "__main__":
    main()
