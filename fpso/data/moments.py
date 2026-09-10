"""Estimation of the optimizer's inputs: expected returns mu and covariance Sigma.

Two estimators live here, and the difference between them is the paper's second
mechanism.

:class:`SampleMoments` is the flat trailing-window estimator. It is regime-blind
by construction: in March 2020 the covariance it hands the optimizer is dominated
by the preceding eleven calm months.

:class:`RegimeWeightedMoments` weights each historical day by how much its latent
regime resembles today's, using the detector's posterior. In a crisis the
covariance is then built mostly from past crisis days, which carry both higher
volatility and higher cross-sectional correlation — so the optimizer *sees* the
risk in its inputs instead of being told about it through a penalty weight.

That distinction is why the first mechanism failed. Conditioning the objective's
risk aversion on regime modulates the optimizer's *preferences* while leaving its
*beliefs* stale; the oracle arm showed that even hindsight-perfect *contemporaneous*
labels bought nothing that way. This moves the signal upstream, into the beliefs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Moments:
    """Annualised first and second moments over a common asset ordering."""

    assets: pd.Index
    mu: np.ndarray
    sigma: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.assets)
        if self.mu.shape != (n,) or self.sigma.shape != (n, n):
            raise ValueError(
                f"Moment shapes {self.mu.shape}/{self.sigma.shape} do not match "
                f"{n} assets."
            )


@dataclass(frozen=True)
class EstimationContext:
    """Regime information an estimator may condition on.

    Passed by the backtest engine, which is the only component allowed to touch
    dates — so `posteriors` is already causally sliced, and an estimator cannot
    accidentally reach past the decision date.
    """

    as_of: pd.Timestamp
    posterior: np.ndarray
    """P(s_t | x_{1:t}) at the decision date, over canonical regime labels."""
    posteriors: pd.DataFrame
    """P(s_d | x_{1:t}) for each historical day d, same column order."""


class MomentEstimator(ABC):
    """Maps a window of daily returns onto annualised (mu, Sigma)."""

    needs_regime_context: bool = False
    """Whether the engine must compute a posterior history for this estimator.
    Building one costs an extra HMM forward-backward pass per rebalance, so it is
    skipped for estimators that ignore it."""

    @abstractmethod
    def estimate(
        self, window: pd.DataFrame, context: EstimationContext | None = None
    ) -> Moments:
        """Estimate moments from `window`, which must already be causally sliced."""


class SampleMoments(MomentEstimator):
    """Annualised sample mean and covariance — the paper's stated estimator.

    Assets whose observation count in the window falls below `min_observations`
    are dropped rather than zero-filled: a name that stopped trading has no
    return, and imputing 0.0 would understate its risk contribution.
    """

    def __init__(self, min_observations: int = 120, ridge: float = 1e-8):
        self.min_observations = min_observations
        self.ridge = ridge

    def estimate(
        self, window: pd.DataFrame, context: EstimationContext | None = None
    ) -> Moments:
        usable = window.loc[:, window.notna().sum() >= self.min_observations]
        usable = usable.dropna(axis=0, how="all")
        if usable.shape[1] < 2:
            raise ValueError(
                f"Only {usable.shape[1]} asset(s) have >= {self.min_observations} "
                "observations in the estimation window."
            )

        mu = usable.mean() * TRADING_DAYS_PER_YEAR
        sigma = usable.cov() * TRADING_DAYS_PER_YEAR
        # A small ridge keeps Sigma positive definite when the window is short
        # relative to the number of assets; it does not change the ranking of
        # candidate portfolios in any material way.
        sigma_values = sigma.to_numpy(dtype=float)
        sigma_values[np.diag_indices_from(sigma_values)] += self.ridge

        return Moments(
            assets=usable.columns,
            mu=mu.to_numpy(dtype=float),
            sigma=sigma_values,
        )


class RegimeWeightedMoments(MomentEstimator):
    """Covariance estimated from the days that most resemble today's regime.

    Each historical day *d* in the lookback gets a weight

        w_d = sum_k  P(s_t = k) * P(s_d = k)

    — the probability that *d* and today were drawn from the same latent state,
    under the detector's own posterior. Weights are normalised to sum to one, and
    the covariance is the weighted second moment about the weighted mean.

    Three deliberate choices a reviewer will ask about:

    **The mean is left unweighted by default.** Regime-conditional expected
    returns are far noisier than regime-conditional risk, and estimating them
    credibly is a forecasting paper of its own. The mechanism's claim is about
    *risk*, so only Sigma is conditioned unless `weight_mean` is set — which
    exists so the alternative can be reported as an ablation rather than debated.

    **A longer lookback than the flat estimator.** Weighting effectively discards
    the days that do not match, so the same 252-day window would leave far too
    few observations behind. The default reaches back five years, which is why
    the panel cache starts well before the evaluation window.

    **Shrinkage toward the flat estimate, scaled by effective sample size.** When
    the posterior is concentrated the weighted covariance rests on few effective
    observations and is badly conditioned for 50 assets, so it is shrunk toward
    the ordinary sample covariance over the same window with intensity
    ``n_assets / (n_assets + n_eff)``. When the posterior is diffuse, n_eff is
    large, the shrinkage vanishes, and the estimator degrades gracefully to the
    flat one — which is also what makes the null hypothesis reachable.
    """

    needs_regime_context = True

    def __init__(
        self,
        min_observations: int = 120,
        ridge: float = 1e-8,
        weight_mean: bool = False,
        min_effective_observations: float = 60.0,
    ):
        self.min_observations = min_observations
        self.ridge = ridge
        self.weight_mean = weight_mean
        self.min_effective_observations = min_effective_observations
        self._fallback = SampleMoments(min_observations, ridge)

    def estimate(
        self, window: pd.DataFrame, context: EstimationContext | None = None
    ) -> Moments:
        if context is None or context.posteriors.empty:
            # No regime information available (burn-in, or a detector that does
            # not produce posteriors). Falling back keeps the arm well-defined
            # rather than silently producing a degenerate covariance.
            return self._fallback.estimate(window)

        usable = window.loc[:, window.notna().sum() >= self.min_observations]
        usable = usable.dropna(axis=0, how="any")
        if usable.shape[1] < 2 or usable.shape[0] < self.min_observations:
            return self._fallback.estimate(window)

        weights = self._similarity_weights(usable.index, context)
        if weights is None:
            return self._fallback.estimate(window)

        observations = usable.to_numpy(dtype=float)
        n_effective = float(1.0 / np.sum(weights**2))
        if n_effective < self.min_effective_observations:
            return self._fallback.estimate(window)

        mean = (
            weights @ observations
            if self.weight_mean
            else observations.mean(axis=0)
        )
        deviations = observations - mean
        weighted = (deviations * weights[:, None]).T @ deviations
        # Reliability-weight correction: with weights summing to one, the
        # unbiased scaling is n_eff / (n_eff - 1) rather than n / (n - 1).
        weighted *= n_effective / (n_effective - 1.0)

        flat = np.cov(observations, rowvar=False)
        intensity = usable.shape[1] / (usable.shape[1] + n_effective)
        sigma = (1.0 - intensity) * weighted + intensity * flat

        sigma *= TRADING_DAYS_PER_YEAR
        sigma[np.diag_indices_from(sigma)] += self.ridge
        return Moments(
            assets=usable.columns,
            mu=mean * TRADING_DAYS_PER_YEAR,
            sigma=sigma,
        )

    def _similarity_weights(
        self, dates: pd.Index, context: EstimationContext
    ) -> np.ndarray | None:
        """Normalised P(same regime as today) for each day in `dates`."""
        aligned = context.posteriors.reindex(dates).dropna()
        if len(aligned) < self.min_observations:
            return None

        raw = aligned.to_numpy(dtype=float) @ np.asarray(context.posterior, dtype=float)
        raw = np.clip(raw, 0.0, None)
        total = raw.sum()
        if total <= 0:
            return None

        # Reindexing may have dropped days; renormalise over what survives and
        # pad the dropped days with zero so the vector aligns with `dates`.
        weights = pd.Series(raw / total, index=aligned.index).reindex(dates).fillna(0.0)
        return weights.to_numpy(dtype=float)


def build_moment_estimator(
    name: str,
    min_observations: int,
    tilt_strength: dict[str, float] | None = None,
    scorer: str = "defensive",
    market_cap: pd.DataFrame | None = None,
) -> MomentEstimator:
    """Instantiate the estimator named by ``ExperimentConfig.moment_estimator``."""
    if name == "regime_tilt":
        # Imported here: tilt.py imports from this module, so a top-level import
        # would be circular.
        from fpso.data.tilt import RegimeTiltedMoments, build_scorer

        return RegimeTiltedMoments(
            base=SampleMoments(min_observations=min_observations),
            scorer=build_scorer(scorer, market_cap),
            strength_by_state=dict(tilt_strength or {}),
        )
    if name == "sample":
        return SampleMoments(min_observations=min_observations)
    if name == "regime_weighted":
        return RegimeWeightedMoments(min_observations=min_observations)
    if name == "regime_weighted_mean":
        return RegimeWeightedMoments(min_observations=min_observations, weight_mean=True)
    raise ValueError(
        f"Unknown moment_estimator '{name}'; expected sample, regime_weighted, "
        "regime_weighted_mean or regime_tilt."
    )
