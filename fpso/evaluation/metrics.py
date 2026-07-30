"""Performance metrics computed from a daily net-return path.

vectorbt owns the *simulation*; this module owns the *reporting*, so the same
metric definitions apply to simulated paths, to seed-averaged paths, and to the
regime-conditional sub-samples where no ``Portfolio`` object exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class PerformanceMetrics:
    """The summary statistics reported for every arm."""

    n_days: int
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def summarize(returns: pd.Series | np.ndarray) -> PerformanceMetrics:
    """Compute the full metric set for one daily return path.

    Annual return is geometric (CAGR), not the arithmetic mean scaled by 252, so
    it is consistent with the terminal value of the equity curve.
    """
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 2:
        return PerformanceMetrics(len(r), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = len(r) / TRADING_DAYS_PER_YEAR
    annual_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0

    daily_std = float(r.std(ddof=1))
    annual_volatility = daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (
        float(np.sqrt(TRADING_DAYS_PER_YEAR) * r.mean() / daily_std) if daily_std > 0 else 0.0
    )

    downside = r[r < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (
        float(np.sqrt(TRADING_DAYS_PER_YEAR) * r.mean() / downside_std)
        if downside_std > 0
        else 0.0
    )

    max_drawdown = float((equity / equity.cummax() - 1.0).min())
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0

    return PerformanceMetrics(
        n_days=len(r),
        total_return=total_return,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        calmar=calmar,
    )


def summarize_by_regime(
    returns: pd.Series, regime_labels: pd.Series
) -> pd.DataFrame:
    """Metrics computed separately within each detected regime.

    `regime_labels` is indexed by rebalance date; it is forward-filled onto the
    daily grid so each day is attributed to the regime that was *acting* when the
    portfolio held that day was chosen. This is the table where the risk story
    lives even when the pooled Sharpe difference is modest.
    """
    daily_labels = (
        regime_labels.reindex(regime_labels.index.union(returns.index))
        .ffill()
        .reindex(returns.index)
        # Shift by one day: a target chosen on rebalance date t is only held from
        # t+1, so date t's return was earned by the *previous* portfolio and
        # belongs to the previous regime.
        .shift(1)
    )

    rows = {}
    for label, group in returns.groupby(daily_labels):
        if len(group) >= 2:
            rows[label] = summarize(group).as_dict()
    frame = pd.DataFrame(rows).T
    frame.index.name = "regime"
    return frame


def metrics_table(paths: dict[str, pd.Series]) -> pd.DataFrame:
    """Stack `summarize` across named return paths into one comparison table."""
    return pd.DataFrame(
        {name: summarize(path).as_dict() for name, path in paths.items()}
    ).T
