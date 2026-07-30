"""Regime-weighted moment estimation — the second mechanism.

The estimator has to satisfy two things that are easy to get wrong. It must
*degrade to the flat estimator* whenever the regime signal is uninformative, so
that the null hypothesis stays reachable and a failure of the mechanism cannot
masquerade as a failure of the plumbing. And it must remain **causal** — it now
reaches back five years instead of one and consumes a posterior computed by the
detector, both of which are new opportunities to leak the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpso.config.schema import RegimeConfig, RegimeLabel
from fpso.data.moments import (
    EstimationContext,
    RegimeWeightedMoments,
    SampleMoments,
    build_moment_estimator,
)
from fpso.regime import build_detector
from fpso.regime.features import RegimeFeatureBuilder

N_LABELS = len(RegimeLabel)


@pytest.fixture
def window(panel) -> pd.DataFrame:
    """Five years of returns — the lookback the regime-weighted estimator uses."""
    return panel.returns.loc["2010-01-01":"2014-12-31"]


def context_from(window: pd.DataFrame, posterior, day_posteriors) -> EstimationContext:
    """Build an estimation context with an explicit posterior history."""
    return EstimationContext(
        as_of=window.index[-1],
        posterior=np.asarray(posterior, dtype=float),
        posteriors=pd.DataFrame(
            day_posteriors, index=window.index, columns=[state.name for state in RegimeLabel]
        ),
    )


# ------------------------------------------------------------- degeneracy ----

def test_uniform_posterior_reproduces_the_flat_estimator(window):
    """With no regime information the two estimators must agree.

    This is what keeps the experiment honest: if the mechanism does nothing, the
    regime arm collapses onto the baseline rather than onto some third thing.
    """
    flat = np.full((len(window), N_LABELS), 1.0 / N_LABELS)
    context = context_from(window, np.full(N_LABELS, 1.0 / N_LABELS), flat)

    weighted = RegimeWeightedMoments(min_observations=100).estimate(window, context)
    sample = SampleMoments(min_observations=100).estimate(window)

    np.testing.assert_allclose(weighted.mu, sample.mu, rtol=1e-6)
    np.testing.assert_allclose(weighted.sigma, sample.sigma, rtol=0.05)


def test_missing_context_falls_back_to_the_flat_estimator(window):
    """Burn-in and non-probabilistic detectors must not produce a broken Sigma."""
    weighted = RegimeWeightedMoments(min_observations=100).estimate(window, None)
    sample = SampleMoments(min_observations=100).estimate(window)
    np.testing.assert_allclose(weighted.sigma, sample.sigma, rtol=1e-9)


def test_degenerate_posterior_falls_back_rather_than_producing_garbage(window):
    """A posterior concentrated on too few days must not yield a singular Sigma."""
    day_posteriors = np.zeros((len(window), N_LABELS))
    day_posteriors[:, int(RegimeLabel.CALM)] = 1.0
    # Only five days are flagged as crisis; today is crisis.
    day_posteriors[:5] = 0.0
    day_posteriors[:5, int(RegimeLabel.CRISIS)] = 1.0

    today = np.zeros(N_LABELS)
    today[int(RegimeLabel.CRISIS)] = 1.0

    moments = RegimeWeightedMoments(min_observations=100).estimate(
        window, context_from(window, today, day_posteriors)
    )
    assert np.all(np.isfinite(moments.sigma))
    assert np.all(np.linalg.eigvalsh(moments.sigma) > 0)


# ---------------------------------------------------------------- mechanism --

def test_crisis_weighting_raises_estimated_risk(window):
    """The point of the mechanism: crisis days should produce a riskier Sigma.

    Days are labelled crisis exactly when the cross-sectional return is most
    volatile, so conditioning on crisis must recover a higher-variance covariance
    than the flat estimate. If it does not, the weighting is not reaching the data.
    """
    daily_dispersion = window.std(axis=1)
    is_crisis = daily_dispersion > daily_dispersion.quantile(0.75)

    day_posteriors = np.zeros((len(window), N_LABELS))
    day_posteriors[is_crisis.to_numpy(), int(RegimeLabel.CRISIS)] = 1.0
    day_posteriors[~is_crisis.to_numpy(), int(RegimeLabel.CALM)] = 1.0

    today = np.zeros(N_LABELS)
    today[int(RegimeLabel.CRISIS)] = 1.0

    weighted = RegimeWeightedMoments(min_observations=100).estimate(
        window, context_from(window, today, day_posteriors)
    )
    flat = SampleMoments(min_observations=100).estimate(window)

    assert np.trace(weighted.sigma) > np.trace(flat.sigma), (
        "Conditioning on the high-volatility state did not raise estimated risk."
    )


def test_mean_is_unweighted_by_default(window):
    """Only risk is regime-conditioned unless the mean variant is requested."""
    day_posteriors = np.zeros((len(window), N_LABELS))
    day_posteriors[: len(window) // 2, int(RegimeLabel.CRISIS)] = 1.0
    day_posteriors[len(window) // 2 :, int(RegimeLabel.CALM)] = 1.0
    today = np.zeros(N_LABELS)
    today[int(RegimeLabel.CRISIS)] = 1.0
    context = context_from(window, today, day_posteriors)

    default = RegimeWeightedMoments(min_observations=100).estimate(window, context)
    variant = RegimeWeightedMoments(min_observations=100, weight_mean=True).estimate(
        window, context
    )
    flat = SampleMoments(min_observations=100).estimate(window)

    np.testing.assert_allclose(default.mu, flat.mu, rtol=1e-6)
    assert not np.allclose(variant.mu, flat.mu)


# ------------------------------------------------------------------ causality --

def test_posterior_series_uses_no_future_data(panel):
    """The posterior history at `as_of` must not move when the future changes."""
    features = RegimeFeatureBuilder().build(panel.market)
    as_of = pd.Timestamp("2012-06-29")

    detector = build_detector(RegimeConfig(detector="hmm"), np.random.default_rng(0))
    detector.fit(features, as_of)
    clean = detector.posterior_series(features, as_of)

    poisoned = features.copy()
    poisoned.loc[poisoned.index > as_of] = 99.0
    dirty_detector = build_detector(RegimeConfig(detector="hmm"), np.random.default_rng(0))
    dirty_detector.fit(poisoned, as_of)
    dirty = dirty_detector.posterior_series(poisoned, as_of)

    assert clean is not None and dirty is not None
    pd.testing.assert_frame_equal(clean, dirty)


def test_posterior_series_is_a_probability_distribution(panel):
    features = RegimeFeatureBuilder().build(panel.market)
    as_of = pd.Timestamp("2012-06-29")
    detector = build_detector(RegimeConfig(detector="hmm"), np.random.default_rng(0))
    detector.fit(features, as_of)

    posteriors = detector.posterior_series(features, as_of)
    assert posteriors.index.max() <= as_of, "Posterior history reached past as_of."
    np.testing.assert_allclose(posteriors.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
    assert (posteriors.to_numpy() >= -1e-12).all()


def test_unfitted_detector_returns_no_posteriors(panel):
    features = RegimeFeatureBuilder().build(panel.market)
    detector = build_detector(RegimeConfig(detector="hmm"), np.random.default_rng(0))
    assert detector.posterior_series(features, pd.Timestamp("2006-01-31")) is None


def test_detectors_without_a_state_model_return_none(panel):
    features = RegimeFeatureBuilder().build(panel.market)
    for name in ("none", "constant", "volq"):
        detector = build_detector(RegimeConfig(detector=name), np.random.default_rng(0))
        assert detector.posterior_series(features, pd.Timestamp("2012-06-29")) is None


# ------------------------------------------------------------------ registry --

def test_registry_builds_each_estimator():
    assert isinstance(build_moment_estimator("sample", 120), SampleMoments)
    assert isinstance(build_moment_estimator("regime_weighted", 120), RegimeWeightedMoments)
    assert build_moment_estimator("regime_weighted_mean", 120).weight_mean is True


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="Unknown moment_estimator"):
        build_moment_estimator("magic", 120)


def test_only_the_regime_estimator_requests_context():
    """The flat estimator must not pay for a posterior it will ignore."""
    assert build_moment_estimator("sample", 120).needs_regime_context is False
    assert build_moment_estimator("regime_weighted", 120).needs_regime_context is True
