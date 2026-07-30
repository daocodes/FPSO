"""The falsification control and the pinned-label diagnostics.

The tests here are the acceptance criteria for the control's rebuild. The first
version of `ShuffledLabelDetector` sampled a Markov chain from the HMM's fitted
transition matrix; because the stressed states are not persistent month-to-month,
the minimum-dwell smoother filtered nearly all of its variation away and the
control degenerated into a static arm. It still *ran* — nothing errored, no test
failed, and the resulting numbers looked plausible. Only a direct comparison of
label statistics against the headline arm exposed it.

So that comparison is now a test: a control that does not deliver the same amount
of parameter variation as the treatment cannot falsify anything, and this suite
fails if it stops doing so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpso.config.schema import FPSOParams, RegimeConfig, RegimeLabel
from fpso.regime import build_detector
from fpso.regime.controls import _split_into_episodes
from fpso.regime.features import RegimeFeatureBuilder
from fpso.regime.policy import RegimeParameterPolicy

OVERRIDES = {
    "CALM": {"alpha": 0.10, "lambda_v": 0.60},
    "TURBULENT": {"alpha": 0.30, "lambda_v": 1.60},
    "CRISIS": {"alpha": 0.35, "lambda_v": 3.00},
}


@pytest.fixture
def features(panel) -> pd.DataFrame:
    return RegimeFeatureBuilder().build(panel.market)


def acting_labels(detector_name: str, features: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Run one detector through the policy and collect the acting labels.

    Uses the policy rather than the detector directly so the smoothing and
    burn-in behaviour under test is the same code the backtest executes.
    """
    config = RegimeConfig(
        detector=detector_name, burn_in_months=6, min_dwell=2, overrides=OVERRIDES
    )
    policy = RegimeParameterPolicy(
        build_detector(config, np.random.default_rng(seed)), config, FPSOParams()
    )
    # Month-end grid: the same cadence MonthlySchedule hands the engine, so the
    # control's internally-reconstructed schedule and this comparison line up.
    usable = features.dropna().index
    dates = pd.DatetimeIndex(
        pd.Series(usable, index=usable).resample("ME").last().dropna().values
    )

    labels = {}
    for as_of in dates:
        decision = policy.decide(features, as_of)
        if not decision.assignment.is_burn_in:
            labels[as_of] = decision.label.name
    return pd.Series(labels, name="regime")


def switch_count(labels: pd.Series) -> int:
    """Number of times the acting label changes."""
    return int((labels != labels.shift()).sum() - 1)


# ------------------------------------------------------- the shuffled control --

def test_control_delivers_comparable_parameter_variation(features):
    """The control must switch about as often as the headline arm.

    This is the property whose absence made the first control useless: it emitted
    1.1 switches per run against the treatment's 10, so it supplied almost no
    parameter variation and could not isolate the contribution of the signal.
    """
    treatment = acting_labels("hmm", features)
    control = acting_labels("shuffled", features)

    treatment_switches = switch_count(treatment)
    control_switches = switch_count(control)

    assert treatment_switches >= 2, "Fixture is too short to exercise this property."
    assert control_switches >= 0.5 * treatment_switches, (
        f"Control switches {control_switches} times vs. the treatment's "
        f"{treatment_switches}; it supplies too little variation to falsify anything."
    )


def test_control_matches_the_treatment_marginals(features):
    """Label frequencies must match, or the arms differ in more than timing."""
    treatment = acting_labels("hmm", features).value_counts(normalize=True)
    control = acting_labels("shuffled", features).value_counts(normalize=True)

    # Permuting whole episodes moves labels around in time without creating or
    # destroying any, so the match here is exact rather than approximate.
    for label in treatment.index:
        assert control.get(label, 0.0) == pytest.approx(treatment[label], abs=1e-9), (
            f"Control spends {control.get(label, 0.0):.1%} of rebalances in {label} "
            f"vs. the treatment's {treatment[label]:.1%}."
        )


def test_control_is_not_aligned_with_the_treatment(features):
    """Matched marginals are worthless if the control tracks the market anyway."""
    treatment = acting_labels("hmm", features)
    control = acting_labels("shuffled", features)

    common = treatment.index.intersection(control.index)
    agreement = float((treatment[common] == control[common]).mean())
    assert agreement < 0.95, (
        f"Control agrees with the treatment on {agreement:.1%} of rebalances; "
        "the timing was not destroyed."
    )


def test_control_varies_with_its_seed(features):
    """Different seeds must draw different permutations, or 30 runs collapse to 1."""
    first = acting_labels("shuffled", features, seed=0)
    second = acting_labels("shuffled", features, seed=7)
    assert not first.equals(second)


def test_episode_decomposition_round_trips():
    """The permutation preserves marginals only if the split is lossless."""
    labels = pd.Series([0, 0, 1, 1, 1, 2, 0, 0])
    episodes = _split_into_episodes(labels)

    assert episodes == [(0, 2), (1, 3), (2, 1), (0, 2)]
    rebuilt = [label for label, length in episodes for _ in range(length)]
    assert rebuilt == labels.tolist()


def test_control_covers_the_same_rebalances(features):
    """The control must decide on exactly the dates the treatment decides on."""
    treatment = acting_labels("hmm", features)
    control = acting_labels("shuffled", features)

    assert control.index.equals(treatment.index)
    assert len(control) == len(treatment)


def test_control_episode_merging_stays_bounded(features):
    """Permutation merges same-label neighbours; the loss must stay modest.

    Reordering episodes occasionally places two CALM blocks side by side, and they
    then read as one longer episode. With three labels roughly a third of the
    joins merge, so the control ends up with fewer, longer episodes than the
    treatment even though every individual label-month is preserved. That is
    inherent to block permutation rather than a defect — but if it ever ran away,
    the control would stop supplying comparable variation, so it is bounded here.
    """
    treatment_episodes = _split_into_episodes(acting_labels("hmm", features))
    control_episodes = _split_into_episodes(acting_labels("shuffled", features))

    assert len(control_episodes) >= 0.5 * len(treatment_episodes), (
        f"Control collapsed to {len(control_episodes)} episodes from the "
        f"treatment's {len(treatment_episodes)}."
    )
    assert len(control_episodes) <= len(treatment_episodes), (
        "Permutation cannot create episodes that were not in the original path."
    )


# ---------------------------------------------------- the pinned diagnostics --

@pytest.mark.parametrize("label", ["CALM", "TURBULENT", "CRISIS"])
def test_constant_detector_pins_its_label(features, label):
    config = RegimeConfig(detector="constant", constant_label=label, burn_in_months=0)
    detector = build_detector(config, np.random.default_rng(0))

    for as_of in features.dropna().index[::63][:10]:
        assignment = detector.infer(features, as_of)
        assert assignment.label == RegimeLabel[label]
        assert assignment.posterior[int(RegimeLabel[label])] == 1.0


def test_pinned_arm_applies_one_parameter_set_throughout(features):
    """A pinned arm must be the headline pipeline with the label held fixed."""
    config = RegimeConfig(
        detector="constant", constant_label="CRISIS", burn_in_months=3, overrides=OVERRIDES
    )
    policy = RegimeParameterPolicy(
        build_detector(config, np.random.default_rng(0)), config, FPSOParams()
    )

    decisions = [policy.decide(features, d) for d in features.dropna().index[::21][:30]]
    active = [d for d in decisions if d.applied]

    assert active, "Pinned arm never applied its parameter block."
    assert {d.params.alpha for d in active} == {0.35}
    assert {d.params.lambda_v for d in active} == {3.0}
    # Burn-in must still be honoured, so the arm stays comparable to regime_hmm.
    assert any(d.assignment.is_burn_in for d in decisions)
