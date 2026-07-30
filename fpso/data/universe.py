"""Point-in-time universe selection.

The legacy backtest took ``get_sp500_constituents(...)[:50]``. PERMNOs are
assigned roughly in CRSP listing order, so "the first 50" is really "the 50
oldest-listed survivors" — a survivorship- and age-biased universe. The default
rule here instead ranks by market capitalisation observed *at the rebalance
date*. :class:`FirstPermnoUniverse` reproduces the old behaviour so the paper can
report the bias as a measured ablation rather than an assertion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from fpso.config.schema import DataConfig
from fpso.data.panel import ReturnPanel


class UniverseProvider(ABC):
    """Chooses the investable asset set for a given rebalance date."""

    def __init__(self, panel: ReturnPanel, size: int):
        self.panel = panel
        self.size = size

    @abstractmethod
    def at(self, as_of: pd.Timestamp) -> pd.Index:
        """Assets investable at `as_of`, using only information available then."""

    def _tradable(self, as_of: pd.Timestamp, lookback_days: int = 252) -> pd.Index:
        """Names with a complete-enough return history entering `as_of`.

        Filtering here rather than inside the optimizer keeps the "what could I
        have held" question separate from the "what should I have held" question.
        """
        window = self.panel.returns.loc[:as_of].tail(lookback_days)
        if window.empty:
            return pd.Index([])
        coverage = window.notna().mean()
        return coverage[coverage >= 0.90].index


class TopMarketCapUniverse(UniverseProvider):
    """Largest `size` names by market cap as of the rebalance date (default)."""

    def at(self, as_of: pd.Timestamp) -> pd.Index:
        tradable = self._tradable(as_of)
        caps = self.panel.market_cap_at(as_of).reindex(tradable).dropna()
        return pd.Index(caps.nlargest(self.size).index).sort_values()


class FirstPermnoUniverse(UniverseProvider):
    """Lowest `size` PERMNOs — the legacy rule, retained as a bias ablation."""

    def at(self, as_of: pd.Timestamp) -> pd.Index:
        return pd.Index(sorted(self._tradable(as_of))[: self.size])


def build_universe_provider(config: DataConfig, panel: ReturnPanel) -> UniverseProvider:
    """Instantiate the universe rule named by `config.universe_rule`."""
    providers = {
        "top_market_cap": TopMarketCapUniverse,
        "first_permno": FirstPermnoUniverse,
    }
    if config.universe_rule not in providers:
        raise ValueError(
            f"Unknown data.universe_rule '{config.universe_rule}'; "
            f"expected one of {sorted(providers)}."
        )
    return providers[config.universe_rule](panel, config.universe_size)
