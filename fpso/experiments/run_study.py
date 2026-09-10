"""Entry point for the full experiment matrix.

    python -m fpso.experiments.run_study                    # all arms, all seeds
    python -m fpso.experiments.run_study --arms static regime_hmm
    python -m fpso.experiments.run_study --seeds 3 --quick  # fast smoke run

``--quick`` shrinks the search budget and the seed count so the whole pipeline
can be exercised in a couple of minutes. It is for checking that things run, not
for producing numbers.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from fpso.config.loader import load_experiment_config
from fpso.experiments.runner import run_matrix

ARM_DIR = Path("configs/arms")

def discover_arms(arm_dir: Path = ARM_DIR) -> list[str]:
    """Every arm config on disk, sorted.

    Derived from the directory rather than hardcoded. A hardcoded list was used
    here previously and went stale: the mechanism-2 (`moments_*`), pinned, and
    turnover-matched arms were added afterwards and were silently excluded, so
    the documented "run everything" command reproduced 11 of 20 arms and omitted
    the ones carrying the study's headline result. Discovery cannot drift from
    the configs that define the matrix.
    """
    return sorted(p.stem for p in arm_dir.glob("*.yaml"))


def main(argv: list[str] | None = None) -> None:
    """Run the matrix. `argv` defaults to the process arguments; passing it
    explicitly lets callers such as
    :mod:`fpso.experiments.verify_reproduction` drive the same entry point the
    CLI uses, rather than a parallel code path that could drift from it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--config-dir", default=str(ARM_DIR))
    parser.add_argument("--results", default=None, help="Override results_dir.")
    parser.add_argument("--seeds", type=int, default=None, help="Use only the first N seeds.")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small swarm and short schedule, for verifying the pipeline runs.",
    )
    args = parser.parse_args(argv)

    arms = args.arms if args.arms else discover_arms(Path(args.config_dir))
    if not arms:
        raise FileNotFoundError(f"No arm configs found in {args.config_dir}")
    paths = [Path(args.config_dir) / f"{arm}.yaml" for arm in arms]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing arm config(s): {missing}")

    overrides = _build_overrides(args)
    if overrides:
        paths = _materialize_overrides(paths, overrides)

    started = time.perf_counter()
    results = run_matrix(paths, workers=args.workers)
    elapsed = time.perf_counter() - started

    total_runs = sum(len(arm.runs) for arm in results.values())
    print(
        f"\nCompleted {total_runs} runs across {len(results)} arms in "
        f"{elapsed / 60:.1f} min. Analyze with:\n"
        "  python -m fpso.experiments.analyze"
    )


def _build_overrides(args) -> dict:
    """Collect the CLI flags that patch every arm config."""
    overrides = {}
    if args.results:
        overrides["results_dir"] = args.results
    if args.seeds:
        overrides["n_seeds"] = args.seeds
    if args.quick:
        overrides.setdefault("n_seeds", args.seeds or 2)
        overrides["quick"] = True
    return overrides


def _materialize_overrides(paths: list[Path], overrides: dict) -> list:
    """Apply CLI overrides by rewriting the loaded configs in memory.

    ``run_matrix`` takes paths so that the on-disk config is the source of truth
    for a normal run. The small shim here keeps that property while still letting
    ``--quick`` and ``--seeds`` work, by passing pre-loaded configs through a
    trivial wrapper that ``load_experiment_config`` accepts.
    """
    configs = []
    for path in paths:
        config = load_experiment_config(path)
        if "results_dir" in overrides:
            config = replace(config, results_dir=overrides["results_dir"])
        if "n_seeds" in overrides:
            config = replace(config, seeds=tuple(config.seeds[: overrides["n_seeds"]]))
        if overrides.get("quick"):
            # Burn-in and refit cadence shrink alongside the schedule; otherwise a
            # four-year smoke run would spend its entire length in burn-in and the
            # regime mechanism would never actually fire.
            config = replace(
                config,
                base_params=replace(config.base_params, num_particles=12, max_iter=15),
                schedule=replace(config.schedule, start="2021-01-01", end="2024-12-31"),
                regime=replace(config.regime, burn_in_months=6, refit_months=6),
                transaction_cost_rates=(0.0, 0.005),
            )
        configs.append(config)
    return configs


if __name__ == "__main__":
    main()
