"""Falsification and upper-bound controls, plus the pinned-label diagnostics.

These arms are what make the study an experiment rather than a demonstration.

:class:`ShuffledLabelDetector` emits a label stream matched to the headline arm's
realized marginals *and* its realized dwell-time distribution, but with the
timing destroyed. It therefore delivers the same amount of parameter variation
with none of the signal. If the headline arm does not beat it, the mechanism is
parameter jitter and the paper must say so.

:class:`ConstantLabelDetector` pins the label to one value for the whole run. Two
arms built on it — always-CALM and always-CRISIS — separate the two things the
regime mechanism could be doing: applying a *better parameter level*, or applying
it *at the right time*. If a pinned arm matches the headline arm, the answer is
levels, not timing.

:class:`OracleOOSDetector` fits on the full sample, so its labels are perfect but
unattainable. It bounds how much regime information could be worth at all, and is
reported as a lookahead upper bound, never as a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpso.config.schema import RegimeConfig, RegimeLabel
from fpso.regime.base import RegimeAssignment, RegimeDetector
from fpso.regime.hmm import GaussianHMMDetector
from fpso.regime.smoothing import MinimumDwellSmoother

REBALANCE_FREQUENCY = "ME"
"""Month-end grid used to reconstruct the headline arm's decision sequence."""


class ShuffledLabelDetector(RegimeDetector):
    """Signal-free labels matched to the headline arm's marginals and dwell times.

    **Why not sample a Markov chain?** The first version of this control drew each
    label from the HMM's fitted transition matrix, lifted to monthly by raising it
    to the 21st power. That produced a degenerate control: the fitted chain is
    persistent for CALM (monthly self-transition around 0.88) but *not* for the
    stressed states (around 0.28), so random draws almost never produced two
    consecutive turbulent or crisis observations — and the minimum-dwell smoother,
    doing exactly its job of rejecting non-persistent labels, filtered them out.
    The control collapsed to 97% CALM with 1.1 switches per run against the
    headline arm's 83% and 10, i.e. to a static arm wearing a control's name.

    The fix is to match the *realized* label path rather than the generating
    process. This detector reconstructs the acting-label sequence the HMM arm
    actually produces, splits it into contiguous episodes, permutes the order of
    those episodes, and replays them. Marginal frequencies and the dwell-time
    distribution are preserved by construction; alignment with the market is
    destroyed.

    Like the oracle, this is a *control*, not a deployable strategy: choosing the
    permutation requires the whole label path, so it is not causal. That is
    legitimate and standard — the control's job is to carry no information about
    the current market state, which it does — but it means this detector is
    excluded from the lookahead test and must be described as a control wherever
    it appears.
    """

    name = "shuffled"
    requires_smoothing = False
    """Labels are replayed real episodes, already smoothed. Re-smoothing them would
    delay every switch by one rebalance and re-shrink the variation this control
    exists to supply."""

    def __init__(self, config: RegimeConfig, rng: np.random.Generator):
        self.config = config
        self.n_states = config.n_states
        self.rng = rng
        self._inner = GaussianHMMDetector(config)
        self._schedule: pd.Series | None = None
        self._calls_seen = 0

    def fit(self, features: pd.DataFrame, as_of: pd.Timestamp) -> None:
        """Keep the inner HMM current; the schedule is built on first use."""
        self._inner.fit(features, as_of)

    def infer(self, features: pd.DataFrame, as_of: pd.Timestamp) -> RegimeAssignment:
        """Replay the permuted schedule; burn-in still follows the real detector."""
        real = self._inner.infer(features, as_of)
        self._calls_seen += 1
        if real.is_burn_in or self._calls_seen <= self.config.burn_in_months:
            return real

        if self._schedule is None:
            # Built on the first rebalance the *policy* will actually act on, and
            # spanning from there to the end of the features. Both halves matter.
            # Deferring keeps the control's marginals matched to the dates the
            # treatment arm acts on; mirroring the policy's own burn-in counter —
            # which starts at the first decision, not at the end of detector
            # burn-in — is what makes the two windows line up exactly.
            self._schedule = self._build_permuted_schedule(features, start=as_of)

        available = self._schedule.loc[:as_of]
        if available.empty:
            return real

        label = RegimeLabel(int(available.iloc[-1]))
        posterior = np.zeros(len(RegimeLabel))
        posterior[int(label)] = 1.0
        return RegimeAssignment(as_of, label, posterior, label, is_burn_in=False)

    def posterior_series(self, features, as_of):
        """Delegate to the inner HMM: the control shuffles labels, not beliefs."""
        return self._inner.posterior_series(features, as_of)

    def _build_permuted_schedule(
        self, features: pd.DataFrame, start: pd.Timestamp
    ) -> pd.Series:
        """Reconstruct the headline arm's acting labels, then permute the episodes."""
        acting = self._headline_acting_labels(features, start)
        if acting.empty:
            return acting

        episodes = _split_into_episodes(acting)
        order = self.rng.permutation(len(episodes))

        labels: list[int] = []
        for index in order:
            label, length = episodes[index]
            labels.extend([int(label)] * length)
        return pd.Series(labels, index=acting.index, name="regime")

    def _headline_acting_labels(
        self, features: pd.DataFrame, start: pd.Timestamp
    ) -> pd.Series:
        """The acting-label sequence the HMM arm produces from `start` onward.

        Replays the treatment arm's pipeline — periodic refit, filtered inference,
        minimum-dwell smoothing — so the episode lengths being permuted are the
        ones the headline arm actually experiences, not an idealisation of them.
        The smoother is warmed on the pre-`start` history so the first episode
        begins in the same state the treatment arm would be in. `start` is the
        first date the policy acts on, so the permutation spans exactly the dates
        the run will read — if it spanned a wider range, the slice actually read
        would carry a different label mix than the treatment arm ever sees, and
        the two arms would differ in their marginals rather than only in timing.
        """
        usable = features.dropna()
        if usable.empty:
            return pd.Series(dtype=int, name="regime")

        grid = pd.Series(usable.index, index=usable.index).resample(
            REBALANCE_FREQUENCY
        ).last().dropna()
        detector = GaussianHMMDetector(self.config)
        smoother = MinimumDwellSmoother(self.config.min_dwell)

        acting: dict[pd.Timestamp, int] = {}
        for as_of in pd.DatetimeIndex(grid.values):
            detector.fit(features, as_of)
            assignment = detector.infer(features, as_of)
            if assignment.is_burn_in:
                continue
            label = int(smoother.update(assignment.raw_label))
            if as_of >= start:
                acting[as_of] = label
        if not acting:
            return pd.Series(dtype=int, name="regime")
        return pd.Series(acting, name="regime").sort_index()


class ConstantLabelDetector(RegimeDetector):
    """Always reports the same label. Separates parameter *levels* from *timing*.

    Built for two diagnostic arms. Always-CALM and always-CRISIS run the headline
    arm's pipeline — same burn-in, same parameter map, same seeds — with the
    regime label pinned. Any difference between a pinned arm and the headline arm
    is attributable to *when* the parameters were applied; any difference between
    a pinned arm and `static` is attributable to *which* parameters they were.
    """

    name = "constant"
    requires_smoothing = False
    """A constant label has nothing to smooth."""

    def __init__(self, config: RegimeConfig):
        self.config = config
        self.n_states = config.n_states
        self.label = RegimeLabel[config.constant_label]

    def fit(self, features: pd.DataFrame, as_of: pd.Timestamp) -> None:
        return None

    def infer(self, features: pd.DataFrame, as_of: pd.Timestamp) -> RegimeAssignment:
        posterior = np.zeros(len(RegimeLabel))
        posterior[int(self.label)] = 1.0
        return RegimeAssignment(
            as_of, self.label, posterior, self.label, is_burn_in=False
        )


class OracleOOSDetector(RegimeDetector):
    """HMM fitted once on the entire sample. Lookahead by construction.

    Reported in the paper as an upper bound on what perfect regime knowledge would
    be worth, and labelled as such everywhere it appears. It is the only component
    permitted to see the future, and it is excluded from the lookahead test for
    exactly that reason.
    """

    name = "oracle"

    def __init__(self, config: RegimeConfig):
        self.config = config
        self.n_states = config.n_states
        self._inner = GaussianHMMDetector(config)
        self._labels: pd.Series | None = None

    def fit(self, features: pd.DataFrame, as_of: pd.Timestamp) -> None:
        """Fit once on the full feature history, ignoring `as_of`."""
        if self._labels is not None:
            return
        full_history = features.dropna()
        if full_history.empty:
            return
        last_date = full_history.index[-1]
        self._inner.fit(full_history, last_date)
        self._labels = self._inner.label_series(full_history, last_date)

    def posterior_series(self, features, as_of):
        """Full-sample posteriors. Lookahead, like everything else in this arm."""
        if self._labels is None:
            return None
        full_history = features.dropna()
        if full_history.empty:
            return None
        return self._inner.posterior_series(full_history, full_history.index[-1]).loc[:as_of]

    def infer(self, features: pd.DataFrame, as_of: pd.Timestamp) -> RegimeAssignment:
        if self._labels is None or self._labels.empty:
            posterior = np.full(len(RegimeLabel), 1.0 / len(RegimeLabel))
            return RegimeAssignment(
                as_of, RegimeLabel.CALM, posterior, RegimeLabel.CALM, is_burn_in=True
            )

        available = self._labels.loc[:as_of]
        label = RegimeLabel(int(available.iloc[-1]) if len(available) else RegimeLabel.CALM)
        posterior = np.zeros(len(RegimeLabel))
        posterior[int(label)] = 1.0
        return RegimeAssignment(as_of, label, posterior, label, is_burn_in=False)


def _split_into_episodes(labels: pd.Series) -> list[tuple]:
    """Compress a label sequence into (label, run length) episodes.

    Label-type agnostic so it can compress either the integer paths used
    internally or the human-readable name paths the tests and result tables use.
    """
    episodes: list[tuple] = []
    current, length = labels.iloc[0], 1
    for value in labels.iloc[1:]:
        if value == current:
            length += 1
        else:
            episodes.append((current, length))
            current, length = value, 1
    episodes.append((current, length))
    return episodes
