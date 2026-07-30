"""Minimum-dwell smoothing of the regime label sequence.

A detector run monthly will occasionally flip state for a single observation.
Acting on those flips means paying turnover for noise. The smoother below is a
causal finite-state machine: a candidate regime must be proposed for `min_dwell`
consecutive rebalances before it becomes the acting regime.

This deliberately introduces a lag of up to `min_dwell - 1` rebalances. That is a
cost, not a leak — the alternative (centred smoothing) would use future labels
and would invalidate the whole experiment.
"""

from __future__ import annotations

from fpso.config.schema import RegimeLabel


class MinimumDwellSmoother:
    """Stateful, forward-only smoother over a stream of raw regime labels."""

    def __init__(self, min_dwell: int = 2):
        if min_dwell < 1:
            raise ValueError("min_dwell must be at least 1.")
        self.min_dwell = min_dwell
        self.reset()

    def reset(self) -> None:
        """Clear all state. Called between backtest runs so arms cannot leak into one another."""
        self._acting: RegimeLabel | None = None
        self._candidate: RegimeLabel | None = None
        self._candidate_streak = 0

    @property
    def acting_label(self) -> RegimeLabel | None:
        """The currently held label, or None before the first observation."""
        return self._acting

    def update(self, raw_label: RegimeLabel) -> RegimeLabel:
        """Feed one raw label and return the acting label after smoothing."""
        if self._acting is None:
            self._acting = raw_label
            self._candidate = raw_label
            self._candidate_streak = 1
            return self._acting

        if raw_label == self._acting:
            # Staying put resets any partial switch that was building.
            self._candidate = raw_label
            self._candidate_streak = self.min_dwell
            return self._acting

        if raw_label == self._candidate:
            self._candidate_streak += 1
        else:
            self._candidate = raw_label
            self._candidate_streak = 1

        if self._candidate_streak >= self.min_dwell:
            self._acting = self._candidate
        return self._acting
