"""Portfolio accounting, delegated to vectorbt.

The engine decides *what to hold*; this module decides *what that was worth*.
Everything below the target-weight matrix — order sizing, cash, fees, drift
between rebalances, the equity curve — is handed to
``vectorbt.Portfolio.from_orders`` with ``size_type='targetpercent'``.

That replaces a hand-rolled ledger whose transaction-cost model charged
``tc_rate * turnover`` against the first day's return of each window. That
approximation ignores drift (weights are only correct on the rebalance day) and
mis-times the cost. vectorbt simulates the actual orders, so the cost is charged
per trade at the traded notional and the reported turnover is realised rather
than assumed.

Timing convention: a target computed on rebalance date *t* is submitted against
*t*'s close, so the position earns returns from *t+1* onward. The decision
therefore never touches a return it also earns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class LedgerResult:
    """One simulated equity path at one transaction-cost rate."""

    transaction_cost_rate: float
    returns: pd.Series
    """Daily portfolio returns net of fees."""
    equity: pd.Series
    """Portfolio value, starting at 1.0."""
    turnover: pd.Series
    """One-way turnover as a fraction of portfolio value, indexed by trade date."""

    @property
    def mean_turnover(self) -> float:
        """Average one-way turnover per rebalance."""
        return float(self.turnover.mean()) if len(self.turnover) else 0.0


class VectorBTLedger:
    """Simulates a target-weight schedule into an equity path via vectorbt."""

    INITIAL_CASH = 1.0

    def __init__(self, returns: pd.DataFrame):
        """Args: `returns` is the daily return panel over the evaluation window."""
        self.returns = returns.sort_index()
        self.prices = self._synthetic_prices(self.returns)

    @staticmethod
    def _synthetic_prices(returns: pd.DataFrame) -> pd.DataFrame:
        """Compound returns into a price series vectorbt can trade against.

        Missing returns are treated as zero *for price construction only*. That is
        safe because the universe rule never targets a non-zero weight in a name
        with missing history, so the fabricated flat segments carry no position and
        contribute nothing to PnL. A name that delists mid-month keeps a frozen
        price (its delisting return is already compounded into its final
        observation by the data layer) and is exited at the next rebalance.
        """
        return (1.0 + returns.fillna(0.0)).cumprod()

    def simulate(
        self, target_weights: pd.DataFrame, transaction_cost_rate: float
    ) -> LedgerResult:
        """Run the weight schedule through vectorbt at one cost rate."""
        size = self._to_order_size(target_weights)
        portfolio = vbt.Portfolio.from_orders(
            close=self.prices,
            size=size,
            size_type="targetpercent",
            group_by=True,
            cash_sharing=True,
            # 'auto' executes sells before buys within a rebalance, which is what
            # makes a fully-invested target reachable without phantom leverage.
            call_seq="auto",
            fees=transaction_cost_rate,
            init_cash=self.INITIAL_CASH,
            freq="1D",
        )
        equity = portfolio.value()
        return LedgerResult(
            transaction_cost_rate=transaction_cost_rate,
            returns=portfolio.returns().rename("returns"),
            equity=equity.rename("equity"),
            turnover=self._one_way_turnover(portfolio, equity),
        )

    def _to_order_size(self, target_weights: pd.DataFrame) -> pd.DataFrame:
        """Expand rebalance-date targets onto the daily grid.

        Rows that are not rebalance dates are NaN, which vectorbt reads as "no
        order" — so the portfolio drifts with the market between decisions
        instead of being silently rebalanced daily.
        """
        aligned = target_weights.reindex(columns=self.prices.columns).fillna(0.0)
        size = pd.DataFrame(np.nan, index=self.prices.index, columns=self.prices.columns)
        common = size.index.intersection(aligned.index)
        size.loc[common] = aligned.loc[common].to_numpy()
        return size

    @staticmethod
    def _one_way_turnover(portfolio: vbt.Portfolio, equity: pd.Series) -> pd.Series:
        """Traded notional per rebalance, halved, as a fraction of portfolio value.

        Halving converts vectorbt's gross traded value (which counts both the sale
        and the purchase legs of a swap) into the one-way convention used
        throughout the portfolio literature.
        """
        orders = portfolio.orders.records_readable
        if orders.empty:
            return pd.Series(dtype=float, name="turnover")

        traded = (orders["Size"] * orders["Price"]).abs()
        by_date = traded.groupby(orders["Timestamp"]).sum()
        value_at_trade = equity.reindex(by_date.index)
        return (0.5 * by_date / value_at_trade).rename("turnover")
