"""Causal market-state features.

Four features, each computable from market returns alone at any date t:

======== ===================================================================
rv_21    21-day realized volatility, annualized — the level of turbulence.
rv_ratio log(rv_21 / rv_252) — turbulence *relative to its own trailing norm*,
         which keeps the feature comparable across decades.
ret_63   63-day cumulative market return — direction, which separates a calm
         grind up from a fast, high-volatility rally.
dd_252   drawdown from the trailing 252-day peak — the crisis marker.
======== ===================================================================

The standardization is the subtle part. Z-scoring against full-sample moments is
a lookahead leak: the mean and standard deviation used to normalize a 2011
observation would embed 2020. :class:`RegimeFeatureBuilder` therefore
standardizes with **expanding** moments, so the value at t uses only data through
t. The lookahead test checks this end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

FEATURE_LABELS = {
    "rv_21": "21d realized vol (ann.)",
    "rv_ratio": "log(rv_21 / rv_252)",
    "ret_63": "63d cumulative return",
    "dd_252": "Drawdown from 252d peak",
}


class RegimeFeatureBuilder:
    """Builds the standardized feature panel the detectors consume."""

    def __init__(
        self,
        features: tuple[str, ...] = ("rv_21", "rv_ratio", "ret_63", "dd_252"),
        min_standardization_obs: int = 252,
    ):
        unknown = set(features) - set(FEATURE_LABELS)
        if unknown:
            raise ValueError(
                f"Unknown regime feature(s) {sorted(unknown)}; "
                f"available: {sorted(FEATURE_LABELS)}"
            )
        self.features = features
        self.min_standardization_obs = min_standardization_obs

    def build(self, market_returns: pd.Series) -> pd.DataFrame:
        """Return the standardized feature frame over the full market history.

        Building the whole frame up front is safe *because every column is
        causal by construction*: each row depends only on rows at or before its
        own date. Detectors then slice it with `as_of` and never need to rebuild.
        """
        raw = self.build_raw(market_returns)
        return self._standardize_expanding(raw)

    def build_raw(self, market_returns: pd.Series) -> pd.DataFrame:
        """Un-standardized features, kept for interpretable figure axes."""
        returns = market_returns.astype(float).sort_index()
        equity = (1.0 + returns).cumprod()

        rv_21 = self._realized_vol(returns, 21)
        rv_252 = self._realized_vol(returns, TRADING_DAYS_PER_YEAR)

        columns = {
            "rv_21": rv_21,
            "rv_ratio": np.log(rv_21 / rv_252.replace(0.0, np.nan)),
            "ret_63": equity.pct_change(63),
            "dd_252": equity / equity.rolling(TRADING_DAYS_PER_YEAR, min_periods=63).max() - 1.0,
        }
        return pd.DataFrame({name: columns[name] for name in self.features})

    @staticmethod
    def _realized_vol(returns: pd.Series, window: int) -> pd.Series:
        """Annualized rolling standard deviation of daily returns."""
        return returns.rolling(window, min_periods=window // 2).std() * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )

    def _standardize_expanding(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Z-score each column against its own expanding mean and std.

        `min_periods` here is what stops the first few hundred observations from
        being standardized against a two-point sample; rows before that threshold
        come out NaN and are dropped by ``RegimeDetector._causal_slice``.
        """
        expanding = raw.expanding(min_periods=self.min_standardization_obs)
        mean = expanding.mean()
        std = expanding.std()
        standardized = (raw - mean) / std.replace(0.0, np.nan)
        return standardized.replace([np.inf, -np.inf], np.nan)
