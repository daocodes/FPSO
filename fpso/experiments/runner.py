"""Orchestration: run one arm across its seeds, or the whole matrix.

The panel is loaded once and shared, so a full matrix run costs one Parquet read
rather than 7 x 30 of them. Seeds within an arm are independent, so they
parallelize over processes; arms are run in sequence to keep memory flat and the
progress output readable.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from fpso.backtest.engine import RollingBacktestEngine
from fpso.backtest.results import ArmResult, BacktestResult
from fpso.config.loader import load_experiment_config
from fpso.config.schema import ExperimentConfig
from fpso.data.panel import ReturnPanel
from fpso.data.source import build_source


def load_panel(config: ExperimentConfig) -> ReturnPanel:
    """Load the return panel described by `config.data`."""
    return build_source(config.data).load()


def run_arm(
    config: ExperimentConfig,
    panel: ReturnPanel,
    workers: int = 1,
    save: bool = True,
    verbose: bool = True,
) -> ArmResult:
    """Run every seed of one arm and optionally persist each result."""
    engine = RollingBacktestEngine(config, panel)
    arm = ArmResult(arm=config.name)

    if workers > 1:
        runs = _run_parallel(engine, config.seeds, workers, verbose)
    else:
        runs = []
        for seed in config.seeds:
            runs.append(engine.run(seed))
            if verbose:
                _report(config, runs[-1])

    arm.runs = sorted(runs, key=lambda run: run.seed)
    if save:
        for run in arm.runs:
            run.save(config.results_dir)
    return arm


def _run_parallel(engine, seeds, workers: int, verbose: bool) -> list[BacktestResult]:
    """Fan seeds out over processes.

    The engine is picklable (it holds only frozen config plus DataFrames), so it
    is sent to each worker once per task rather than rebuilt from the config —
    which matters because building it recomputes the whole regime feature panel.
    """
    results: list[BacktestResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(engine.run, seed): seed for seed in seeds}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if verbose:
                _report(engine.config, result)
    return results


def _report(config: ExperimentConfig, result: BacktestResult) -> None:
    """One progress line per completed seed."""
    from fpso.evaluation.metrics import summarize

    cheapest = min(result.returns_by_cost)
    metrics = summarize(result.returns(cheapest))
    diagnostics = result.diagnostics
    regime_share = (
        diagnostics["regime_applied"].mean() if "regime_applied" in diagnostics else 0.0
    )
    print(
        f"  {config.name:<18} seed={result.seed:<3} "
        f"sharpe={metrics.sharpe:6.3f}  ann={metrics.annual_return:7.2%}  "
        f"mdd={metrics.max_drawdown:7.2%}  "
        f"regime-active={regime_share:5.1%}  "
        f"rebalances={len(diagnostics)}",
        flush=True,
    )


def run_matrix(
    arms: list[str | Path | ExperimentConfig],
    workers: int | None = None,
    save: bool = True,
    verbose: bool = True,
) -> dict[str, ArmResult]:
    """Run several arms against a shared panel, returning results by arm name.

    Accepts config paths (the normal case, where the YAML on disk is the source
    of truth) or already-built configs (used by ``--quick`` and ``--seeds``).
    """
    configs = [
        arm if isinstance(arm, ExperimentConfig) else load_experiment_config(arm)
        for arm in arms
    ]
    _assert_shared_panel(configs)

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    panel = load_panel(configs[0])
    if verbose:
        print(
            f"Panel: {panel.returns.shape[0]} days x {panel.returns.shape[1]} assets "
            f"({panel.dates[0].date()} .. {panel.dates[-1].date()})",
            flush=True,
        )

    results: dict[str, ArmResult] = {}
    for config in configs:
        if verbose:
            print(f"\n=== {config.name} ({len(config.seeds)} seeds) ===", flush=True)
        results[config.name] = run_arm(config, panel, workers, save, verbose)
    return results


def _assert_shared_panel(configs: list[ExperimentConfig]) -> None:
    """Refuse to compare arms that were not run on the same data.

    A silent panel mismatch would make every cross-arm difference meaningless,
    and it is exactly the kind of error that survives review.
    """
    signatures = {
        (c.data.source, c.data.cache_dir, c.data.panel_start, c.data.panel_end,
         c.data.universe_size, c.data.universe_rule)
        for c in configs
    }
    if len(signatures) > 1:
        raise ValueError(
            "Arms in one matrix must share an identical data configuration; "
            f"found {len(signatures)} distinct ones: {sorted(signatures)}"
        )
