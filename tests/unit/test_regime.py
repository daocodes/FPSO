"""Regime subsystem: canonical labelling, smoothing, burn-in and the policy.

The most dangerous failure mode in this subsystem is silent: HMM state indices
permute between refits, so without canonicalization the regime-to-parameter map
would mean a different thing each year and the experiment would quietly become
noise. :func:`test_state_labels_are_ordered_by_volatility` is the guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpso.config.schema import FPSOParams, RegimeConfig, RegimeLabel
from fpso.regime import build_detector
from fpso.regime.features import RegimeFeatureBuilder
from fpso.regime.policy import RegimeParameterPolicy
from fpso.regime.smoothing import MinimumDwellSmoother


@pytest.fixture
def features(panel) -> pd.DataFrame:
    return RegimeFeatureBuilder().build(panel.market)


# ------------------------------------------------------------------ smoother --

def test_smoother_ignores_a_single_flip():
    """A one-off label flip must not move the acting regime at min_dwell = 2."""
    smoother = MinimumDwellSmoother(min_dwell=2)
    sequence = [RegimeLabel.CALM] * 3 + [RegimeLabel.CRISIS] + [RegimeLabel.CALM] * 3
    acting = [smoother.update(label) for label in sequence]
    assert all(label == RegimeLabel.CALM for label in acting)


def test_smoother_switches_after_the_dwell_requirement():
    """A regime that persists for min_dwell observations does become acting."""
    smoother = MinimumDwellSmoother(min_dwell=3)
    sequence = [RegimeLabel.CALM] * 2 + [RegimeLabel.CRISIS] * 4
    acting = [smoother.update(label) for label in sequence]
    assert acting[:4] == [RegimeLabel.CALM] * 4
    assert acting[4:] == [RegimeLabel.CRISIS] * 2


def test_smoother_is_causal():
    """The acting label at step t depends only on labels up to t."""
    prefix = [RegimeLabel.CALM, RegimeLabel.TURBULENT, RegimeLabel.CALM]
    short = MinimumDwellSmoother(2)
    long = MinimumDwellSmoother(2)

    short_out = [short.update(label) for label in prefix]
    long_out = [long.update(label) for label in prefix + [RegimeLabel.CRISIS] * 5]
    assert short_out == long_out[: len(prefix)]


def test_smoother_reset_clears_state():
    smoother = MinimumDwellSmoother(2)
    smoother.update(RegimeLabel.CRISIS)
    smoother.reset()
    assert smoother.acting_label is None


# ------------------------------------------------------------------ detectors --

@pytest.mark.parametrize("name", ["hmm", "gmm", "volq", "none"])
def test_detectors_never_read_past_as_of(features, name):
    """Every detector's inference is unchanged when future features are corrupted."""
    config = RegimeConfig(detector=name, burn_in_months=0)
    as_of = pd.Timestamp("2010-06-30")

    clean = build_detector(config, np.random.default_rng(0))
    clean.fit(features, as_of)
    clean_label = clean.infer(features, as_of).label

    poisoned = features.copy()
    poisoned.loc[poisoned.index > as_of] = 99.0
    dirty = build_detector(config, np.random.default_rng(0))
    dirty.fit(poisoned, as_of)

    assert dirty.infer(poisoned, as_of).label == clean_label


def test_state_labels_are_ordered_by_volatility(features):
    """CALM must be the least volatile fitted state and CRISIS the most.

    Fitted state indices are arbitrary and permute between refits. If this
    ordering ever breaks, the parameter policy silently applies crisis settings
    in calm markets — a failure that produces plausible-looking numbers and no
    error, which is exactly why it is tested.
    """
    detector = build_detector(RegimeConfig(detector="hmm"), np.random.default_rng(0))
    as_of = features.dropna().index[-1]
    detector.fit(features, as_of)

    labels = detector.label_series(features, as_of)
    volatility = features.loc[labels.index, "rv_21"]
    means = volatility.groupby(labels).mean()

    present = sorted(means.index)
    assert list(means.loc[present].sort_values().index) == present, (
        f"Mean volatility is not monotone in the canonical label: {means.to_dict()}"
    )


def test_posterior_is_a_probability_vector(features):
    detector = build_detector(RegimeConfig(detector="hmm"), np.random.default_rng(0))
    as_of = pd.Timestamp("2011-06-30")
    detector.fit(features, as_of)
    assignment = detector.infer(features, as_of)

    assert assignment.posterior.shape == (len(RegimeLabel),)
    assert np.isclose(assignment.posterior.sum(), 1.0)
    assert (assignment.posterior >= 0).all()


def test_shuffled_control_matches_marginals_but_not_the_signal(features):
    """The falsification arm must switch regimes without tracking the market."""
    config = RegimeConfig(detector="shuffled", burn_in_months=0)
    real = build_detector(RegimeConfig(detector="hmm", burn_in_months=0), np.random.default_rng(0))
    control = build_detector(config, np.random.default_rng(0))

    dates = pd.DatetimeIndex(features.dropna().index[::21][-60:])
    real_labels, control_labels = [], []
    for as_of in dates:
        real.fit(features, as_of)
        control.fit(features, as_of)
        real_labels.append(real.infer(features, as_of).label)
        control_labels.append(control.infer(features, as_of).label)

    assert len(set(control_labels)) > 1, "Control never switches; it provides no variation."
    agreement = np.mean([a == b for a, b in zip(real_labels, control_labels, strict=True)])
    assert agreement < 0.9, "Control tracks the real detector too closely to falsify anything."


# --------------------------------------------------------------------- policy --

def test_burn_in_leaves_base_parameters_untouched(features):
    """During burn-in the regime mechanism must be provably inactive."""
    base = FPSOParams()
    config = RegimeConfig(
        detector="hmm", burn_in_months=12, overrides={"CRISIS": {"alpha": 9.0}}
    )
    policy = RegimeParameterPolicy(
        build_detector(config, np.random.default_rng(0)), config, base
    )

    for as_of in features.dropna().index[::21][:12]:
        decision = policy.decide(features, as_of)
        assert not decision.applied
        assert decision.params == base


def test_policy_applies_overrides_after_burn_in(features):
    """Once past burn-in, the acting regime's overrides reach the parameters."""
    base = FPSOParams(alpha=0.2, lambda_v=1.0)
    config = RegimeConfig(
        detector="hmm",
        burn_in_months=1,
        min_dwell=1,
        overrides={
            "CALM": {"alpha": 0.1},
            "TURBULENT": {"alpha": 0.3},
            "CRISIS": {"alpha": 0.35, "lambda_v": 3.0},
        },
    )
    policy = RegimeParameterPolicy(
        build_detector(config, np.random.default_rng(0)), config, base
    )

    applied = [
        policy.decide(features, as_of)
        for as_of in features.dropna().index[::21][:60]
    ]
    active = [d for d in applied if d.applied]
    assert active, "No rebalance ever applied a regime override."
    assert {d.params.alpha for d in active} <= {0.1, 0.3, 0.35}


def test_empty_override_map_reproduces_the_static_arm(features):
    """The static arm is this class with no overrides — same code path, same result."""
    base = FPSOParams()
    config = RegimeConfig(detector="hmm", burn_in_months=0, overrides={})
    policy = RegimeParameterPolicy(
        build_detector(config, np.random.default_rng(0)), config, base
    )
    for as_of in features.dropna().index[::21][:24]:
        decision = policy.decide(features, as_of)
        assert decision.params == base
        assert not decision.applied


def test_unknown_override_field_is_rejected():
    """A typo in a YAML regime block must fail loudly, not silently do nothing."""
    with pytest.raises(ValueError, match="Unknown FPSOParams override"):
        FPSOParams().with_overrides({"alhpa": 0.3})
