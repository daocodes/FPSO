"""Panel providers: cached CRSP (default) and a deterministic synthetic market.

Splitting acquisition behind :class:`PanelSource` is what makes the study
reproducible for a reviewer without WRDS credentials: the CRSP pull happens once
into a Parquet cache, and every experiment thereafter reads that cache. The
synthetic source lets the unit-test suite and the end-to-end smoke run execute
offline with no network and no credentials at all.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from fpso.config.schema import DataConfig
from fpso.data.panel import ReturnPanel

TRADING_DAYS_PER_YEAR = 252


class PanelSource(ABC):
    """Anything that can produce a :class:`ReturnPanel` for the study period."""

    @abstractmethod
    def load(self) -> ReturnPanel:
        """Return the full panel. Implementations should be cheap to call twice."""


class ParquetPanelSource(PanelSource):
    """Reads the one-time CRSP pull from ``cache_dir``."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)

    def load(self) -> ReturnPanel:
        return ReturnPanel.from_parquet(self.cache_dir)


class SyntheticPanelSource(PanelSource):
    """A regime-switching market simulated from a fixed seed.

    Used by the test suite and by ``--source synthetic`` smoke runs. The
    generative process is an explicit 3-state Markov chain over volatility and
    drift, so a detector that works at all should recover something close to the
    true state sequence — which makes this a useful sanity harness for the regime
    subsystem, not just a stand-in for missing data.
    """

    # (annualised drift, annualised vol) per latent state, ordered CALM -> CRISIS.
    STATE_PARAMS = ((0.12, 0.11), (0.02, 0.20), (-0.25, 0.42))
    TRANSITION = np.array(
        [
            [0.980, 0.018, 0.002],
            [0.040, 0.940, 0.020],
            [0.010, 0.090, 0.900],
        ]
    )

    def __init__(
        self,
        start: str,
        end: str,
        n_assets: int = 60,
        seed: int = 20260101,
    ):
        self.start = start
        self.end = end
        self.n_assets = n_assets
        self.seed = seed

    def load(self) -> ReturnPanel:
        rng = np.random.default_rng(self.seed)
        dates = pd.bdate_range(self.start, self.end)
        states = self._simulate_states(rng, len(dates))

        drift, vol = np.array(self.STATE_PARAMS).T
        daily_drift = drift[states] / TRADING_DAYS_PER_YEAR
        daily_vol = vol[states] / np.sqrt(TRADING_DAYS_PER_YEAR)

        market = daily_drift + daily_vol * rng.standard_normal(len(dates))

        # One market factor plus idiosyncratic noise; betas and idio vols are
        # fixed across the sample so any regime effect comes from the market.
        betas = rng.uniform(0.5, 1.6, self.n_assets)
        idio_vol = rng.uniform(0.10, 0.35, self.n_assets) / np.sqrt(TRADING_DAYS_PER_YEAR)
        idio = rng.standard_normal((len(dates), self.n_assets)) * idio_vol
        asset_returns = np.outer(market, betas) + idio

        permnos = list(range(10001, 10001 + self.n_assets))
        returns = pd.DataFrame(asset_returns, index=dates, columns=permnos)
        prices = (1.0 + returns).cumprod()
        shares = pd.Series(rng.uniform(50_000, 5_000_000, self.n_assets), index=permnos)

        return ReturnPanel(
            returns=returns,
            market=pd.Series(market, index=dates, name="market"),
            market_cap=prices.mul(shares, axis=1),
        )

    def _simulate_states(self, rng: np.random.Generator, n_days: int) -> np.ndarray:
        """Draw a latent state path from the fixed transition matrix."""
        states = np.zeros(n_days, dtype=int)
        for t in range(1, n_days):
            states[t] = rng.choice(3, p=self.TRANSITION[states[t - 1]])
        return states


class WRDSPanelSource(PanelSource):
    """One-time CRSP pull. Used by ``fpso.experiments.build_cache``, not by backtests.

    Three deliberate choices here are visible in the paper's data section:

    * The universe is the union of the top-`universe_pool` names by market cap
      sampled annually, so that point-in-time selection can happen downstream
      without re-querying WRDS.
    * Delisting returns from ``crsp.dsedelist`` are compounded into the final
      trading day of a delisted name, so exiting the sample costs what it
      actually cost. Filling those days with 0.0 would bias returns upward.
    * The market proxy is CRSP's own value-weighted index (``crsp.dsi.vwretd``),
      which is one query and is the standard choice in the literature.
    """

    def __init__(self, config: DataConfig, universe_pool: int = 500):
        self.config = config
        self.universe_pool = universe_pool

    def load(self) -> ReturnPanel:
        connection = self._connect()
        try:
            permnos = self._universe_union(connection)
            daily = self._fetch_daily(connection, permnos)
            delisting = self._fetch_delisting_returns(connection, permnos)
            market = self._fetch_market(connection)
        finally:
            connection.close()

        daily = self._apply_delisting_returns(daily, delisting)
        returns = daily.pivot(index="date", columns="permno", values="ret").sort_index()
        market_cap = (
            daily.assign(cap=lambda d: d["prc"].abs() * d["shrout"])
            .pivot(index="date", columns="permno", values="cap")
            .sort_index()
        )
        market = market.reindex(returns.index).ffill()
        return ReturnPanel(returns=returns, market=market, market_cap=market_cap)

    def _connect(self):
        """Open a WRDS session using credentials from the environment."""
        import wrds  # imported lazily so offline runs never need the dependency
        from dotenv import load_dotenv

        load_dotenv()
        username = os.getenv("WRDS_USERNAME", "").strip()
        password = os.getenv("WRDS_PASSWORD", "").strip()
        if not username or not password:
            raise ValueError("Set WRDS_USERNAME and WRDS_PASSWORD (see .env.example).")
        return wrds.Connection(wrds_username=username, wrds_password=password)

    def _universe_union(self, connection) -> list[int]:
        """PERMNOs that were ever top-`universe_pool` by market cap during the sample.

        Sampling once per calendar year keeps the query cheap while guaranteeing
        that any name a point-in-time rule could pick at a monthly rebalance is
        present in the cache.
        """
        years = range(int(self.config.panel_start[:4]), int(self.config.panel_end[:4]) + 1)
        query = f"""
            WITH sample_dates AS (
                SELECT MAX(date) AS date
                FROM crsp.dsf
                WHERE date <= DATE '{{as_of}}'
                  AND date >= DATE '{{as_of}}' - INTERVAL '10 days'
            )
            SELECT d.permno
            FROM crsp.dsf AS d
            JOIN sample_dates s ON d.date = s.date
            JOIN crsp.msenames AS n
              ON d.permno = n.permno AND n.namedt <= d.date AND d.date <= n.nameendt
            WHERE d.prc IS NOT NULL AND d.shrout IS NOT NULL
              AND n.shrcd IN (10, 11)          -- ordinary U.S. common shares
              AND n.exchcd IN (1, 2, 3)        -- NYSE / AMEX / NASDAQ
            ORDER BY ABS(d.prc) * d.shrout DESC
            LIMIT {self.universe_pool}
        """
        permnos: set[int] = set()
        for year in years:
            as_of = min(f"{year}-12-31", self.config.panel_end)
            frame = connection.raw_sql(query.format(as_of=as_of))
            permnos.update(frame["permno"].astype(int).tolist())
            print(f"  universe {as_of}: cumulative {len(permnos)} PERMNOs", flush=True)
        return sorted(permnos)

    def _fetch_daily(self, connection, permnos: list[int]) -> pd.DataFrame:
        """Daily returns, prices and shares outstanding for the universe union."""
        permno_list = ",".join(str(p) for p in permnos)
        query = f"""
            SELECT date, permno, ret, prc, shrout
            FROM crsp.dsf
            WHERE permno IN ({permno_list})
              AND date BETWEEN DATE '{self.config.panel_start}'
                           AND DATE '{self.config.panel_end}'
              AND ret IS NOT NULL
        """
        frame = connection.raw_sql(query, date_cols=["date"])
        frame["permno"] = frame["permno"].astype(int)
        return frame

    def _fetch_delisting_returns(self, connection, permnos: list[int]) -> pd.DataFrame:
        """Delisting returns (B7): what a holder actually realised on the way out."""
        if not self.config.use_delisting_returns:
            return pd.DataFrame(columns=["date", "permno", "dlret"])
        permno_list = ",".join(str(p) for p in permnos)
        query = f"""
            SELECT dlstdt AS date, permno, dlret
            FROM crsp.dsedelist
            WHERE permno IN ({permno_list})
              AND dlret IS NOT NULL
              AND dlstdt BETWEEN DATE '{self.config.panel_start}'
                             AND DATE '{self.config.panel_end}'
        """
        frame = connection.raw_sql(query, date_cols=["date"])
        frame["permno"] = frame["permno"].astype(int)
        return frame

    def _fetch_market(self, connection) -> pd.Series:
        """CRSP value-weighted market return, used for regime features."""
        query = f"""
            SELECT date, vwretd
            FROM crsp.dsi
            WHERE date BETWEEN DATE '{self.config.panel_start}'
                           AND DATE '{self.config.panel_end}'
        """
        frame = connection.raw_sql(query, date_cols=["date"])
        return frame.set_index("date")["vwretd"].sort_index().rename("market")

    @staticmethod
    def _apply_delisting_returns(
        daily: pd.DataFrame, delisting: pd.DataFrame
    ) -> pd.DataFrame:
        """Compound each delisting return into that name's last observed return."""
        if delisting.empty:
            return daily
        merged = daily.merge(delisting, on=["date", "permno"], how="left")
        has_dlret = merged["dlret"].notna()
        merged.loc[has_dlret, "ret"] = (
            (1.0 + merged.loc[has_dlret, "ret"]) * (1.0 + merged.loc[has_dlret, "dlret"])
        ) - 1.0
        return merged.drop(columns=["dlret"])


def build_source(config: DataConfig) -> PanelSource:
    """Instantiate the panel source named by `config.source`."""
    if config.source == "parquet":
        return ParquetPanelSource(config.cache_dir)
    if config.source == "synthetic":
        return SyntheticPanelSource(config.panel_start, config.panel_end)
    if config.source == "wrds":
        return WRDSPanelSource(config)
    raise ValueError(
        f"Unknown data.source '{config.source}'; expected parquet, synthetic or wrds."
    )
