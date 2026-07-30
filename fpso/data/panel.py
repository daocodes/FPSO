"""The in-memory return panel that every downstream component reads from.

A :class:`ReturnPanel` is a wide daily-return frame (dates x PERMNOs) plus the
market-index series used for regime features and the market-cap frame used for
point-in-time universe selection. It is deliberately a dumb container: the only
methods it exposes are *causal slices*, which is what lets the lookahead test in
``tests/unit/test_lookahead.py`` reason about the whole pipeline at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RETURNS_FILE = "returns.parquet"
MARKET_FILE = "market.parquet"
MARKET_CAP_FILE = "market_cap.parquet"


def _to_numpy_floats(frame: pd.DataFrame | pd.Series):
    """Cast to plain numpy float64, turning any pandas NA into np.nan."""
    return frame.astype("float64")


@dataclass(frozen=True)
class ReturnPanel:
    """Daily returns, a market proxy, and market caps over a common date index."""

    returns: pd.DataFrame
    """Daily simple returns; index = trading dates, columns = PERMNOs."""
    market: pd.Series
    """CRSP value-weighted market return, indexed by the same trading dates."""
    market_cap: pd.DataFrame
    """Daily market capitalisation ($000s), same shape/alignment as `returns`."""

    def __post_init__(self) -> None:
        if not self.returns.index.is_monotonic_increasing:
            raise ValueError("ReturnPanel.returns must be sorted by date.")
        if self.returns.index.has_duplicates:
            raise ValueError("ReturnPanel.returns has duplicate dates.")

        # The WRDS driver hands back pandas' nullable Float64 extension dtype,
        # which numba cannot compile — so vectorbt's simulation kernel fails with
        # an opaque typing error deep in the call stack. Normalising here means
        # every panel presents plain numpy float64 regardless of its source, and
        # that failure mode cannot recur.
        object.__setattr__(self, "returns", _to_numpy_floats(self.returns))
        object.__setattr__(self, "market_cap", _to_numpy_floats(self.market_cap))
        object.__setattr__(self, "market", _to_numpy_floats(self.market))

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.returns.index)

    def returns_between(
        self, start: pd.Timestamp, end: pd.Timestamp, assets: pd.Index | None = None
    ) -> pd.DataFrame:
        """Returns on the closed interval ``[start, end]``, optionally sub-set to `assets`.

        This is the only way the backtest engine reads history, so "no lookahead"
        reduces to "`end` is never past the decision date".
        """
        window = self.returns.loc[start:end]
        if assets is not None:
            window = window.reindex(columns=assets)
        return window

    def market_before(self, as_of: pd.Timestamp) -> pd.Series:
        """Market returns strictly up to and including `as_of`."""
        return self.market.loc[:as_of]

    def market_cap_at(self, as_of: pd.Timestamp) -> pd.Series:
        """Latest observed market cap per asset on or before `as_of`."""
        history = self.market_cap.loc[:as_of]
        if history.empty:
            return pd.Series(dtype=float)
        return history.ffill().iloc[-1].dropna()

    def to_parquet(self, directory: str | Path) -> None:
        """Persist the panel as three Parquet files under `directory`."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        # Parquet requires string column names; PERMNOs are integers.
        self.returns.rename(columns=str).to_parquet(path / RETURNS_FILE)
        self.market.to_frame("market").to_parquet(path / MARKET_FILE)
        self.market_cap.rename(columns=str).to_parquet(path / MARKET_CAP_FILE)

    @classmethod
    def from_parquet(cls, directory: str | Path) -> ReturnPanel:
        """Load a panel previously written by :meth:`to_parquet`."""
        path = Path(directory)
        missing = [f for f in (RETURNS_FILE, MARKET_FILE, MARKET_CAP_FILE)
                   if not (path / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Panel cache at {path} is missing {missing}. "
                "Build it first with: python -m fpso.experiments.build_cache"
            )
        returns = pd.read_parquet(path / RETURNS_FILE)
        market = pd.read_parquet(path / MARKET_FILE)["market"]
        market_cap = pd.read_parquet(path / MARKET_CAP_FILE)
        return cls(
            returns=returns.rename(columns=int).sort_index(),
            market=market.sort_index(),
            market_cap=market_cap.rename(columns=int).sort_index(),
        )
