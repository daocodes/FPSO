"""Re-execute stored runs and prove they reproduce, bit for bit.

A manifest records the code that *claimed* to produce a result. This checks the
claim by running the arm again and diffing the decisions.

The distinction matters here more than usual. The study's first result matrix was
stamped ``-dirty`` on 450 of 719 runs, because ``git status --porcelain`` reports
the whole working tree and regenerating a figure is enough to trip it. Read
literally, that marker says the provenance of most of the study is unknown. Read
correctly, it says almost nothing at all — and the only way to tell the two
readings apart is to re-run the code and compare.

Bit-identical target weights are the right acceptance test rather than matching
Sharpe ratios. Performance is a lossy summary of a decision path: two different
weight schedules can post the same Sharpe, and a comparison at that level would
pass while the optimizer had silently changed. The weights *are* the decision.

Usage::

    python -m fpso.experiments.verify_reproduction                  # spot check
    python -m fpso.experiments.verify_reproduction --arms static tilt_hmm
    python -m fpso.experiments.verify_reproduction --seeds 3 --all-arms

Exit status is 0 when every checked run reproduces and 1 otherwise, so this can
gate a release.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fpso.backtest.results import source_hash

DEFAULT_RESULTS = Path("results")
DEFAULT_ARMS = ("static",)
WEIGHTS_FILE = "target_weights.parquet"


@dataclass(frozen=True)
class ReproductionCheck:
    """Outcome of re-running one (arm, seed)."""

    arm: str
    seed: int
    reproduced: bool
    max_abs_difference: float
    n_differing: int
    stored_source_hash: str | None
    note: str = ""

    @property
    def status(self) -> str:
        if self.reproduced:
            return "OK"
        return "MISMATCH" if not self.note else "SKIP"


def compare_weights(stored: Path, fresh: Path) -> tuple[bool, float, int]:
    """Diff two target-weight schedules on their union of columns.

    Aligning on the union with a zero fill rather than requiring identical
    columns means a genuine difference is reported as a differing *weight*
    instead of a column-set error, which is easier to act on. A name absent from
    one run is a zero holding in that run, which is exactly what the fill means.
    """
    left = pd.read_parquet(stored)
    right = pd.read_parquet(fresh)
    if not left.index.equals(right.index):
        return False, float("inf"), max(len(left), len(right))

    left, right = left.align(right, join="outer", axis=1, fill_value=0.0)
    difference = np.abs(left.to_numpy() - right.to_numpy())
    return bool(np.array_equal(left.to_numpy(), right.to_numpy())), float(
        difference.max()
    ), int((difference > 0).sum())


def _stored_source_hash(run_dir: Path) -> str | None:
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        return None
    return json.loads(manifest.read_text()).get("source_hash")


def verify(
    arms: list[str],
    seeds: int,
    results_dir: Path,
    fresh_dir: Path,
    config_dir: Path,
) -> list[ReproductionCheck]:
    """Re-run each arm into `fresh_dir` and compare against `results_dir`."""
    from fpso.experiments.run_study import main as run_study_main

    run_study_main(
        [
            "--arms", *arms,
            "--seeds", str(seeds),
            "--results", str(fresh_dir),
            "--config-dir", str(config_dir),
        ]
    )

    checks: list[ReproductionCheck] = []
    for arm in arms:
        for seed in range(seeds):
            stored = results_dir / arm / f"seed={seed}" / WEIGHTS_FILE
            fresh = fresh_dir / arm / f"seed={seed}" / WEIGHTS_FILE
            if not stored.exists():
                checks.append(
                    ReproductionCheck(arm, seed, False, float("nan"), 0, None,
                                      note="no stored run")
                )
                continue
            if not fresh.exists():  # pragma: no cover - run_study would have raised
                checks.append(
                    ReproductionCheck(arm, seed, False, float("nan"), 0, None,
                                      note="re-run produced nothing")
                )
                continue
            same, largest, count = compare_weights(stored, fresh)
            checks.append(
                ReproductionCheck(
                    arm, seed, same, largest, count,
                    _stored_source_hash(stored.parent),
                )
            )
    return checks


def report(checks: list[ReproductionCheck], current_hash: str) -> bool:
    """Print a per-run table and return True when everything reproduced."""
    print(f"\ncurrent source_hash: {current_hash[:16]}\n")
    print(f"{'arm':<28} {'seed':>4} {'status':>9} {'max diff':>10} {'cells':>7}  stored hash")
    print("-" * 84)
    for check in sorted(checks, key=lambda c: (c.arm, c.seed)):
        stored = check.stored_source_hash
        marker = "-" if stored is None else stored[:16]
        if stored is not None and stored != current_hash:
            marker += " (differs)"
        print(
            f"{check.arm:<28} {check.seed:>4} {check.status:>9} "
            f"{check.max_abs_difference:>10.2e} {check.n_differing:>7}  {marker}"
        )

    ok = [c for c in checks if c.reproduced]
    print(f"\n{len(ok)}/{len(checks)} runs reproduced bit-identically.")
    if len(ok) != len(checks):
        print("Decisions changed. Do not claim reproducibility until this is resolved.")
    return len(ok) == len(checks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=list(DEFAULT_ARMS))
    parser.add_argument("--all-arms", action="store_true",
                        help="Check every arm present in the results directory.")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--config-dir", type=Path, default=Path("configs/arms"))
    parser.add_argument("--fresh-dir", type=Path, default=Path("results_reproduction"))
    args = parser.parse_args(argv)

    arms = args.arms
    if args.all_arms:
        arms = sorted(p.name for p in args.results.iterdir() if p.is_dir())

    checks = verify(arms, args.seeds, args.results, args.fresh_dir, args.config_dir)
    return 0 if report(checks, source_hash()) else 1


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())
