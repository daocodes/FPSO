"""Does conditioning shrinkage on the optimizer's own uncertainty beat a constant?

Pipeline, per seed:

1. At each rebalance run the swarm **once** to choose a support, and record the
   ex-ante uncertainty features. The support is chosen on the unshrunk objective,
   so it does not depend on the shrinkage intensity — which is what lets one
   search price the whole grid.
2. Solve the exact weights for every intensity, carrying each intensity's own
   holdings forward, giving one portfolio path per intensity.
3. Assemble those into a :class:`ShrinkageGrid` and run the policies: learned,
   walk-forward best constant, oracle, and the shuffled control.
4. Re-run each policy's choices **sequentially**, so the evaluated path is exact
   rather than stitched.

On the approximation in step 3: the grid records what a *constant*-intensity path
realised at each date, but a switching policy arrives at each date holding
something different, so its turnover term differs. The grid is therefore an
approximation used for *training*; every number reported comes from step 4, which
is exact. Training on an approximate signal is ordinary; evaluating on one is not.

Usage:
    python -m fpso.experiments.run_adaptive --config configs/base.yaml --seeds 5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpso.adaptive.features import FEATURE_NAMES, build_features
from fpso.adaptive.shrinkage import (
    RidgeShrinkagePolicy,
    ShrinkageGrid,
    oracle_shrinkage,
    shuffled_shrinkage,
    walk_forward_best_constant,
    walk_forward_learned,
)
from fpso.backtest.schedule import build_schedule
from fpso.config.loader import load_experiment_config
from fpso.data.moments import SampleMoments
from fpso.data.universe import build_universe_provider
from fpso.experiments.runner import load_panel
from fpso.optimizer.fpso_cw import FPSOCWOptimizer

DEFAULT_INTENSITIES = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
TRADING_DAYS = 252


@dataclass
class SeedRun:
    """One seed's grid: features, per-intensity weights, per-intensity outcomes."""

    dates: pd.DatetimeIndex
    features: np.ndarray
    weights: dict[float, pd.DataFrame]
    outcomes: np.ndarray
    supports: list[np.ndarray]
    moments: list
    prev_index: list


def _one_seed(config, panel, seed: int, intensities) -> SeedRun:
    universe = build_universe_provider(config.data, panel)
    estimator = SampleMoments(min_observations=config.schedule.min_observations)
    optimizer = FPSOCWOptimizer(config.base_params)
    dates = build_schedule(config.schedule.frequency).dates(panel.dates)
    dates = dates[(dates >= pd.Timestamp(config.schedule.start))
                  & (dates <= pd.Timestamp(config.schedule.end))]

    columns = panel.returns.columns
    paths = {d: pd.DataFrame(0.0, index=dates, columns=columns) for d in intensities}
    previous = {d: np.zeros(len(columns)) for d in intensities}
    feature_rows, kept, supports, moment_list = [], [], [], []
    rng = np.random.default_rng(seed)

    for as_of in dates:
        assets = universe.at(as_of)
        window = panel.returns_between(
            as_of - pd.Timedelta(days=config.schedule.estimation_window_days), as_of, assets
        )
        try:
            moments = estimator.estimate(window)
        except ValueError:
            for d in intensities:
                paths[d].loc[as_of] = previous[d]
            continue

        position = {asset: i for i, asset in enumerate(columns)}
        local = [position[a] for a in moments.assets]
        prev_local = {d: previous[d][local] for d in intensities}

        search, _ = optimizer.solve_over_shrinkage(
            moments, prev_local[intensities[0]], rng, intensities[:1]
        )
        support = np.flatnonzero(search.weights > 1e-8)
        feature_rows.append(
            build_features(search.population, search.population_fitness, moments).to_array()
        )
        kept.append(as_of)
        supports.append(support)
        moment_list.append(moments)

        for d in intensities:
            w_local, _ = optimizer.solve_weights(support, moments, prev_local[d], d)
            row = np.zeros(len(columns))
            row[local] = w_local
            paths[d].loc[as_of] = row
            previous[d] = row

    kept_index = pd.DatetimeIndex(kept)
    outcomes = np.column_stack([
        _per_rebalance_returns(paths[d], panel.returns, kept_index) for d in intensities
    ])
    return SeedRun(kept_index, np.array(feature_rows), paths, outcomes, supports,
                   moment_list, list(range(len(kept_index))))


def _per_rebalance_returns(
    weights: pd.DataFrame, returns: pd.DataFrame, dates: pd.DatetimeIndex
) -> np.ndarray:
    """Gross return realised over each holding period by that intensity's path."""
    out = []
    for i, start in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else returns.index[-1]
        segment = returns.loc[(returns.index > start) & (returns.index <= end)]
        if segment.empty:
            out.append(0.0)
            continue
        held = weights.loc[start].to_numpy(dtype=float)
        value = 1.0
        for _, day in segment.iterrows():
            r = np.nan_to_num(day.to_numpy(dtype=float))
            value *= 1 + float(held @ r)
            if held.sum() > 0:
                held = held * (1 + r)
                held = held / held.sum()
        out.append(value - 1.0)
    return np.array(out)


def _sequential_backtest(run: SeedRun, chosen: np.ndarray, config, panel, cost: float):
    """Re-solve the path under a switching policy, so turnover is exact."""
    optimizer = FPSOCWOptimizer(config.base_params)
    columns = panel.returns.columns
    position = {asset: i for i, asset in enumerate(columns)}
    weights = pd.DataFrame(0.0, index=run.dates, columns=columns)
    previous = np.zeros(len(columns))
    for i, as_of in enumerate(run.dates):
        moments = run.moments[i]
        local = [position[a] for a in moments.assets]
        w_local, _ = optimizer.solve_weights(
            run.supports[i], moments, previous[local], float(chosen[i])
        )
        row = np.zeros(len(columns))
        row[local] = w_local
        weights.loc[as_of] = row
        previous = row
    return _equity(weights, panel.returns, cost)


def _equity(weights: pd.DataFrame, returns: pd.DataFrame, cost: float) -> pd.Series:
    dates = returns.index[returns.index >= weights.index[0]]
    rebalances, held, out = set(weights.index), np.zeros(weights.shape[1]), []
    for day in dates:
        charge = 0.0
        if day in rebalances:
            target = weights.loc[day].to_numpy(dtype=float)
            # vectorbt charges each leg, so a one-way turnover of x costs 2*x*rate.
            charge = 2.0 * 0.5 * float(np.abs(target - held).sum()) * cost
            held = target
        r = np.nan_to_num(returns.loc[day].to_numpy(dtype=float))
        out.append((1 + float(held @ r)) * (1 - charge) - 1)
        if held.sum() > 0:
            held = held * (1 + r)
            held = held / held.sum()
    return pd.Series(out, index=dates)


def _sharpe(series: pd.Series) -> float:
    annual = (1 + series).prod() ** (TRADING_DAYS / len(series)) - 1
    volatility = series.std() * np.sqrt(TRADING_DAYS)
    return float(annual / volatility) if volatility > 0 else float("nan")


def _turnover(weights: pd.DataFrame) -> float:
    a = weights.to_numpy()
    return float(np.mean([0.5 * np.abs(a[i] - a[i - 1]).sum() for i in range(1, len(a))]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=36)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--costs", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.015])
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    panel = load_panel(config)
    intensities = DEFAULT_INTENSITIES

    runs = []
    for seed in range(args.seeds):
        runs.append(_one_seed(config, panel, seed, intensities))
        print(f"  seed {seed} searched", flush=True)

    deltas = np.array(intensities, dtype=float)
    policies: dict[str, list[np.ndarray]] = {
        "learned": [], "best_constant": [], "oracle": [], "shuffled": []
    }
    coefficients = []
    for run in runs:
        grid = ShrinkageGrid(deltas=deltas, features=run.features, outcomes=run.outcomes)
        learned = walk_forward_learned(grid, ridge=args.ridge, min_train=args.min_train)
        policies["learned"].append(learned)
        policies["best_constant"].append(
            walk_forward_best_constant(grid, min_train=args.min_train))
        policies["oracle"].append(oracle_shrinkage(grid))
        policies["shuffled"].append(shuffled_shrinkage(learned, np.random.default_rng(0)))
        coefficients.append(
            RidgeShrinkagePolicy(ridge=args.ridge)
            .fit(grid.features, grid.best_delta_per_date())
            .coefficients
        )

    rows = []
    for name in ("learned", "best_constant", "oracle", "shuffled"):
        for cost in args.costs:
            series = [_sequential_backtest(run, chosen, config, panel, cost)
                      for run, chosen in zip(runs, policies[name], strict=True)]
            rows.append({"policy": name, "cost": cost,
                         "sharpe": float(np.mean([_sharpe(s) for s in series]))})
    for d in intensities:
        for cost in args.costs:
            series = [_equity(run.weights[d], panel.returns, cost) for run in runs]
            rows.append({"policy": f"constant_{d}", "cost": cost,
                         "sharpe": float(np.mean([_sharpe(s) for s in series]))})

    table = pd.DataFrame(rows).pivot(index="policy", columns="cost", values="sharpe")
    pd.set_option("display.width", 200)
    print("\nSharpe by policy and transaction cost\n")
    print(table.round(4).to_string())

    print("\nMean chosen intensity (post burn-in):")
    for name in ("learned", "best_constant", "oracle"):
        vals = np.concatenate([c[args.min_train:] for c in policies[name]])
        print(f"  {name:<14} {vals.mean():.3f}   sd {vals.std():.3f}")

    print("\nRidge coefficients (standardised, full-sample fit, for reporting only):")
    mean_coef = np.mean(coefficients, axis=0)
    for feature, value in sorted(zip(FEATURE_NAMES, mean_coef, strict=True),
                                 key=lambda kv: -abs(kv[1])):
        print(f"  {feature:<24} {value:+.4f}")


if __name__ == "__main__":
    main()
