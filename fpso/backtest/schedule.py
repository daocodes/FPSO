"""Rebalance schedules.

The pre-refactor backtest rebalanced once per year: 14 decisions over 2010-2024.
A regime switch that can fire 14 times in fifteen years is not a regime switch —
COVID would have been a single observation. :class:`MonthlySchedule` is the
paper's stated cadence and gives ~168 decisions; :class:`AnnualSchedule` is kept
so the change can be reported as a measured rebalance-frequency ablation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class RebalanceSchedule(ABC):
    """Selects the trading dates on which the portfolio is re-optimized."""

    name: str = "schedule"

    @abstractmethod
    def dates(self, trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Rebalance dates drawn from `trading_days`, in ascending order."""

    @staticmethod
    def _period_ends(trading_days: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
        """Last actual trading day within each calendar period.

        Anchoring on observed trading days rather than calendar month-ends means
        a rebalance never lands on a holiday and never silently shifts forward
        into data the decision was not entitled to see.
        """
        series = pd.Series(trading_days, index=trading_days)
        return pd.DatetimeIndex(series.resample(freq).last().dropna().values)


class MonthlySchedule(RebalanceSchedule):
    """Rebalance on the last trading day of each month (default)."""

    name = "monthly"

    def dates(self, trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return self._period_ends(trading_days, "ME")


class AnnualSchedule(RebalanceSchedule):
    """Rebalance on the last trading day of each year (legacy cadence)."""

    name = "annual"

    def dates(self, trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return self._period_ends(trading_days, "YE")


def build_schedule(frequency: str) -> RebalanceSchedule:
    """Instantiate the schedule named by `ScheduleConfig.frequency`."""
    schedules = {"monthly": MonthlySchedule, "annual": AnnualSchedule}
    if frequency not in schedules:
        raise ValueError(
            f"Unknown schedule.frequency '{frequency}'; expected monthly or annual."
        )
    return schedules[frequency]()
