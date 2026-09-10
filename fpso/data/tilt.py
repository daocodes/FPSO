"""Regime-conditioned tilts on the expected-return vector.

The third injection point, and the only one that moves the *optimum* rather than
the route taken to it.

Where the earlier two mechanisms failed
---------------------------------------
Conditioning FPSO's hyperparameters (`alpha`, `gamma`, `lambda_v`, ...) changes
how the swarm searches, and the search already converges: measured on the real
panel, iterations-to-converge correlates with starting quality at rho = -0.15,
so the landscape decides the answer and the search parameters decide only the
route. Conditioning Sigma changes the risk estimate, but the entire measured
effect was the longer estimation window and the turnover it suppressed.

Tilting mu is different in kind. It moves the location of the optimum, so
brightness rises around the favoured names and the swarm's own attraction
dynamics carry it there — no change to the optimizer, the constraints, or the
universe. This is the mechanism for which "the fireflies cluster there" is
literally true.

What is actually being claimed
------------------------------
A tilt is a **cross-sectional return forecast**: asserting that defensive names
outperform in a crisis is a prediction, and the mechanism inherits the standard
burden of one. Both other mechanisms were about search and risk, which is what
made their failure informative about regimes specifically. This one will be right
or wrong about returns, so the controls below are what separate "the regime
predicted the cross-section" from "a static defensive tilt would have done the
same".

The inertia bias this addresses
-------------------------------
The objective charges `lambda_t * ||w - w_prev||_1` for trading, so holding
collects the expected return without paying the fee and the incumbent portfolio
starts ahead of every challenger. Any mechanism whose effect is "trade less"
therefore looks good for reasons that have nothing to do with market states —
which is what the study's turnover-matched control demonstrates. A mu tilt is the
one intervention that can overcome that bias on purpose, because it changes what
the optimizer is trying to hold rather than how reluctant it is to move.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from fpso.data.moments import EstimationContext, MomentEstimator, Moments


class AssetScorer(ABC):
    """Scores each asset for how much the current state favours it.

    Scores are standardized before use, so only the cross-sectional *ordering*
    and relative spread matter; the absolute scale is set by `tilt_strength`.
    """

    @abstractmethod
    def score(self, window: pd.DataFrame, assets: pd.Index) -> np.ndarray:
        """One score per asset in `assets`, from causally-sliced `window`."""


class DefensiveScorer(AssetScorer):
    """Favours low-volatility, low-drawdown names — the classic defensive tilt.

    Deliberately transparent rather than fitted. The study's own evidence is that
    a learned detector never beat a rule: `regime_volq` (volatility terciles,
    nothing estimated) matched the fitted HMM at every cost level. Starting from
    a rule means any benefit is attributable to the *timing* of the tilt rather
    than to a second layer of estimation.
    """

    def __init__(self, lookback: int = 63):
        self.lookback = lookback

    def score(self, window: pd.DataFrame, assets: pd.Index) -> np.ndarray:
        recent = window.tail(self.lookback).reindex(columns=assets)
        volatility = recent.std().to_numpy(dtype=float)
        volatility = np.nan_to_num(volatility, nan=np.nanmean(volatility))
        # Negated: low volatility scores high.
        return -volatility


class RegimeTiltedMoments(MomentEstimator):
    """Wraps any estimator and tilts its `mu` according to the detected state.

    A wrapper rather than a new estimator so the tilt composes with whatever is
    already producing moments, and so the *only* difference between a tilted arm
    and its baseline is this one transformation.

        mu_tilted = mu + strength(state) * sd(mu) * z(score)

    Scaling by `sd(mu)` keeps the tilt commensurate with the dispersion of the
    estimated returns themselves, so one `tilt_strength` behaves consistently
    across universes and periods instead of needing to be re-tuned whenever the
    return scale changes.

    Args:
        base: The estimator producing untilted moments.
        scorer: Supplies the cross-sectional ordering to tilt toward.
        strength_by_state: Regime label name -> tilt strength. Absent states are
            untilted, so a partially specified map degrades to the baseline
            rather than silently applying someone else's tilt.
    """

    needs_regime_context = True

    def __init__(
        self,
        base: MomentEstimator,
        scorer: AssetScorer,
        strength_by_state: dict[str, float],
    ):
        self.base = base
        self.scorer = scorer
        self.strength_by_state = dict(strength_by_state)
        # A tilt keyed on the posterior needs the same context the base estimator
        # may need, so the engine must build it either way.
        self.needs_regime_context = True

    def estimate(
        self, window: pd.DataFrame, context: EstimationContext | None = None
    ) -> Moments:
        moments = self.base.estimate(window, context)
        strength = self._strength_for(context)
        if strength == 0.0:
            return moments

        scores = self.scorer.score(window, moments.assets)
        z = _standardize(scores)
        scale = float(np.std(moments.mu))
        if scale <= 0.0 or not np.isfinite(scale):
            return moments

        return Moments(
            assets=moments.assets,
            mu=moments.mu + strength * scale * z,
            sigma=moments.sigma,
        )

    def _strength_for(self, context: EstimationContext | None) -> float:
        """Posterior-weighted tilt strength.

        Weighting by the posterior rather than switching on the argmax label
        means a 55/45 call applies roughly half the tilt instead of the whole of
        it. Hard switching on a marginal posterior is a large, discrete portfolio
        change justified by a coin flip, and it is precisely the behaviour that
        drives the turnover the rest of this study is about.
        """
        if context is None or not self.strength_by_state:
            return 0.0
        from fpso.config.schema import RegimeLabel

        total = 0.0
        for state, probability in enumerate(context.posterior):
            name = RegimeLabel(state).name
            total += float(probability) * self.strength_by_state.get(name, 0.0)
        return total


def _standardize(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-sd scores; all-zero when the cross-section is degenerate.

    Zero mean matters: it keeps the tilt a pure *relative* statement about which
    names are favoured, so it cannot inflate or deflate the level of expected
    return and quietly change how aggressive the portfolio is overall.
    """
    values = np.asarray(values, dtype=float)
    spread = float(np.std(values))
    if spread <= 0.0 or not np.isfinite(spread):
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / spread


class ExogenousScorer(AssetScorer):
    """Base for scorers reading data the optimizer never receives.

    Every mechanism tested before this one was built from the return panel — and
    so were mu and Sigma. A regime detector fitted on realized volatility is a
    function of returns; the covariance the optimizer already holds is another
    function of the same returns. Re-deriving one from the other cannot add
    information, only rearrange it, which is why four structurally different
    injection points all collapsed onto the same turnover explanation and why
    perfect knowledge of the regime was worth ~0.01 Sharpe.

    Scorers below break that circle: they read quantities that are genuinely
    absent from `returns`, so a tilt built on them carries information the
    optimizer could not have recovered on its own. Whether that information
    *predicts* anything is the empirical question — but unlike the earlier
    mechanisms, it is at least possible in principle.
    """


class NetIssuanceScorer(ExogenousScorer):
    """Favours firms shrinking their share count (buybacks) over those diluting.

    Share count is invisible to a return series: two firms with identical return
    paths can be issuing and repurchasing respectively, and mu and Sigma cannot
    tell them apart. Recovered from cached data without a new WRDS pull, since
    the part of a market-cap change that returns do *not* explain is exactly the
    change in shares outstanding:

        shrout_growth_t = (mcap_t / mcap_{t-1}) / (1 + ret_t)

    Measured on this panel the signal has a real cross-sectional spread (8.1%
    between the 10th and 90th percentiles) and correlates with trailing returns
    at only +0.11 — near-orthogonal, which is the property that makes it worth
    testing at all.

    Negated so that *shrinking* share counts score high.
    """

    def __init__(self, market_cap: pd.DataFrame, lookback: int = 252):
        self.market_cap = market_cap
        self.lookback = lookback

    def score(self, window: pd.DataFrame, assets: pd.Index) -> np.ndarray:
        recent = window.tail(self.lookback)
        caps = self.market_cap.reindex(index=recent.index, columns=assets)
        rets = recent.reindex(columns=assets)

        unexplained = (caps / caps.shift(1)) / (1.0 + rets)
        unexplained = unexplained.replace([np.inf, -np.inf], np.nan)
        growth = unexplained.apply(lambda c: np.nanprod(c.to_numpy()) - 1.0)

        scores = -growth.to_numpy(dtype=float)
        return np.nan_to_num(scores, nan=float(np.nanmedian(scores)) if np.isfinite(
            np.nanmedian(scores)) else 0.0)


class SizeScorer(ExogenousScorer):
    """Favours smaller names within the universe.

    Market capitalisation is a level, and returns are changes: no window of
    returns reveals whether a firm is worth $50bn or $500bn. Included as a second
    exogenous input mainly as a contrast to `NetIssuanceScorer` — if a tilt on
    *any* outside variable helps, the mechanism is about breaking the circle
    rather than about issuance specifically, and that distinction matters for
    what the paper can claim.
    """

    def __init__(self, market_cap: pd.DataFrame):
        self.market_cap = market_cap

    def score(self, window: pd.DataFrame, assets: pd.Index) -> np.ndarray:
        as_of = window.index[-1]
        caps = self.market_cap.reindex(columns=assets)
        caps = caps.loc[caps.index <= as_of]
        latest = caps.iloc[-1].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_cap = np.log(np.where(latest > 0, latest, np.nan))
        return -np.nan_to_num(log_cap, nan=float(np.nanmedian(log_cap)))


def build_scorer(name: str, market_cap: pd.DataFrame | None = None) -> AssetScorer:
    """Instantiate the scorer named by ``ExperimentConfig.tilt_scorer``.

    The exogenous scorers need market caps, which the engine takes from the
    panel; asking for one without them is a configuration error rather than a
    silent fallback, since falling back would quietly turn an exogenous arm into
    a return-derived one and invalidate the comparison it exists to make.
    """
    if name == "defensive":
        return DefensiveScorer()
    if name in ("net_issuance", "size"):
        if market_cap is None:
            raise ValueError(f"Scorer '{name}' needs market caps; none supplied.")
        return NetIssuanceScorer(market_cap) if name == "net_issuance" else SizeScorer(market_cap)
    raise ValueError(
        f"Unknown tilt_scorer '{name}'; expected defensive, net_issuance or size."
    )
