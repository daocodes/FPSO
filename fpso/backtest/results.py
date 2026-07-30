"""Result containers and the reproducibility manifest written beside them.

Every persisted result set carries a manifest recording the config, the git SHA,
the interpreter and the versions of the packages that can change numerical
output. A reviewer holding the artifact can therefore tell whether a figure came
from the code they are reading.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fpso.config.schema import ExperimentConfig


@dataclass(frozen=True)
class RebalanceRecord:
    """Everything decided at a single rebalance date."""

    as_of: pd.Timestamp
    regime_label: str
    regime_posterior: tuple[float, ...]
    regime_applied: bool
    is_burn_in: bool
    n_assets: int
    n_active: int
    objective_value: float
    solve_seconds: float
    params: dict[str, Any]

    def to_row(self) -> dict[str, Any]:
        """Flatten into one row of the per-rebalance diagnostics table."""
        row = {
            "as_of": self.as_of,
            "regime_label": self.regime_label,
            "regime_applied": self.regime_applied,
            "is_burn_in": self.is_burn_in,
            "n_assets": self.n_assets,
            "n_active": self.n_active,
            "objective_value": self.objective_value,
            "solve_seconds": self.solve_seconds,
        }
        for state, probability in enumerate(self.regime_posterior):
            row[f"posterior_{state}"] = probability
        row.update({f"param_{k}": v for k, v in self.params.items()})
        return row


@dataclass
class BacktestResult:
    """One (arm, seed) run: weights, diagnostics, and one equity path per cost rate."""

    arm: str
    seed: int
    target_weights: pd.DataFrame
    rebalances: list[RebalanceRecord]
    returns_by_cost: dict[float, pd.Series]
    equity_by_cost: dict[float, pd.Series]
    turnover_by_cost: dict[float, pd.Series]
    config: ExperimentConfig | None = None
    _diagnostics: pd.DataFrame | None = None
    """Populated on reload, when the original RebalanceRecord objects are gone."""

    @property
    def diagnostics(self) -> pd.DataFrame:
        """Per-rebalance table: regimes, parameters, solve times, cardinality."""
        if self._diagnostics is not None:
            return self._diagnostics
        return pd.DataFrame([record.to_row() for record in self.rebalances]).set_index(
            "as_of"
        )

    @property
    def regime_labels(self) -> pd.Series:
        """Acting regime label at each rebalance, for the regime-conditional tables."""
        diagnostics = self.diagnostics
        return diagnostics["regime_label"].rename("regime")

    def returns(self, transaction_cost_rate: float = 0.0) -> pd.Series:
        """Daily net returns at one cost rate."""
        if transaction_cost_rate not in self.returns_by_cost:
            raise KeyError(
                f"No path simulated at tc={transaction_cost_rate}; "
                f"available: {sorted(self.returns_by_cost)}"
            )
        return self.returns_by_cost[transaction_cost_rate]

    def save(self, directory: str | Path) -> Path:
        """Persist to ``<directory>/<arm>/seed=<seed>/`` and return that path."""
        path = Path(directory) / self.arm / f"seed={self.seed}"
        path.mkdir(parents=True, exist_ok=True)

        self.target_weights.rename(columns=str).to_parquet(path / "target_weights.parquet")
        self.diagnostics.to_parquet(path / "rebalances.parquet")
        pd.DataFrame(self.returns_by_cost).to_parquet(path / "returns_by_cost.parquet")
        pd.DataFrame(self.equity_by_cost).to_parquet(path / "equity_by_cost.parquet")
        pd.DataFrame(self.turnover_by_cost).to_parquet(path / "turnover_by_cost.parquet")

        if self.config is not None:
            manifest = build_manifest(self.config, arm=self.arm, seed=self.seed)
            (path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        return path

    @classmethod
    def load(cls, directory: str | Path) -> BacktestResult:
        """Reload a result set previously written by :meth:`save`."""
        path = Path(directory)
        returns = pd.read_parquet(path / "returns_by_cost.parquet")
        equity = pd.read_parquet(path / "equity_by_cost.parquet")
        turnover = pd.read_parquet(path / "turnover_by_cost.parquet")
        diagnostics = pd.read_parquet(path / "rebalances.parquet")

        # Parquet requires string column names, so PERMNOs come back as strings.
        weights = pd.read_parquet(path / "target_weights.parquet").rename(columns=int)

        return cls(
            arm=path.parent.name,
            seed=int(path.name.split("=")[1]),
            target_weights=weights,
            rebalances=[],
            returns_by_cost={float(c): returns[c] for c in returns.columns},
            equity_by_cost={float(c): equity[c] for c in equity.columns},
            turnover_by_cost={float(c): turnover[c].dropna() for c in turnover.columns},
            _diagnostics=diagnostics,
        )


@dataclass
class ArmResult:
    """All seeds for one arm, plus the seed-averaged path the paper reports."""

    arm: str
    runs: list[BacktestResult] = field(default_factory=list)

    def mean_returns(self, transaction_cost_rate: float = 0.0) -> pd.Series:
        """Cross-seed average daily return path.

        Averaging returns (not equity) across seeds keeps the result interpretable
        as "the expected path of the strategy" rather than as a portfolio of 30
        independently-funded strategies.
        """
        paths = [run.returns(transaction_cost_rate) for run in self.runs]
        return pd.concat(paths, axis=1).mean(axis=1).rename(self.arm)

    def per_seed_returns(self, transaction_cost_rate: float = 0.0) -> pd.DataFrame:
        """Daily returns with one column per seed, for the paired statistics."""
        frame = pd.concat(
            {run.seed: run.returns(transaction_cost_rate) for run in self.runs}, axis=1
        )
        frame.columns.name = "seed"
        return frame

    @property
    def mean_turnover(self) -> float:
        """Average one-way turnover per rebalance across seeds (gross of costs)."""
        values = [
            float(run.turnover_by_cost[min(run.turnover_by_cost)].mean())
            for run in self.runs
        ]
        return float(np.mean(values)) if values else 0.0


def build_manifest(config: ExperimentConfig, arm: str, seed: int) -> dict[str, Any]:
    """Assemble the reproducibility record stored next to every result set."""
    return {
        "arm": arm,
        "seed": seed,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "config": _to_plain(config),
    }


def _git_sha() -> str:
    """Current commit, or a marker when the tree is not a git checkout."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    """Versions of the packages that can move numerical results."""
    from importlib.metadata import PackageNotFoundError, version

    names = ["numpy", "pandas", "scipy", "scikit-learn", "hmmlearn", "vectorbt"]
    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - optional at runtime
            versions[name] = "not installed"
    return versions


def _to_plain(value: Any) -> Any:
    """Recursively convert dataclasses and tuples into JSON-serialisable data."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _to_plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value
