"""The regime -> parameter mechanism: the paper's actual contribution.

:class:`RegimeParameterPolicy` owns the full decision at a rebalance: fit the
detector if due, infer the acting regime, smooth it, honour burn-in, and return
the :class:`~fpso.config.schema.FPSOParams` the optimizer should use.

Two design consequences worth stating in the paper:

* The static arm runs this same class with an empty override map. The static and
  regime arms therefore share every line of code and differ only in
  configuration, so the ablation isolates the mechanism and nothing else.
* During burn-in the policy returns the *base* parameters unchanged, so the
  regime mechanism is provably inactive rather than merely untrained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpso.config.schema import FPSOParams, RegimeConfig, RegimeLabel
from fpso.regime.base import RegimeAssignment, RegimeDetector
from fpso.regime.smoothing import MinimumDwellSmoother


@dataclass(frozen=True)
class RegimeDecision:
    """What the policy concluded at one rebalance, recorded for the results table."""

    assignment: RegimeAssignment
    params: FPSOParams
    applied: bool
    """False when burn-in or an empty override map left the base parameters in force."""

    @property
    def label(self) -> RegimeLabel:
        return self.assignment.label


class RegimeParameterPolicy:
    """Turns a feature panel into per-rebalance optimizer parameters."""

    def __init__(
        self,
        detector: RegimeDetector,
        config: RegimeConfig,
        base_params: FPSOParams,
    ):
        self.detector = detector
        self.config = config
        self.base_params = base_params
        self.smoother = MinimumDwellSmoother(config.min_dwell)
        self._observations_seen = 0

    def reset(self) -> None:
        """Clear per-run state so a fresh backtest cannot inherit another's labels."""
        self.smoother.reset()
        self._observations_seen = 0

    def decide(self, features: pd.DataFrame, as_of: pd.Timestamp) -> RegimeDecision:
        """Fit-if-due, infer, smooth, and map the acting regime to parameters."""
        self.detector.fit(features, as_of)
        raw_assignment = self.detector.infer(features, as_of)
        self._observations_seen += 1

        if self._in_burn_in(raw_assignment):
            return RegimeDecision(
                assignment=self._as_burn_in(raw_assignment),
                params=self.base_params,
                applied=False,
            )

        acting_label = (
            self.smoother.update(raw_assignment.raw_label)
            if self.detector.requires_smoothing
            else raw_assignment.raw_label
        )
        assignment = RegimeAssignment(
            as_of=raw_assignment.as_of,
            label=acting_label,
            posterior=raw_assignment.posterior,
            raw_label=raw_assignment.raw_label,
            is_burn_in=False,
        )

        overrides = self.config.overrides_for(acting_label)
        if not overrides:
            return RegimeDecision(assignment, self.base_params, applied=False)
        return RegimeDecision(
            assignment=assignment,
            params=self.base_params.with_overrides(overrides),
            applied=True,
        )

    def _in_burn_in(self, assignment: RegimeAssignment) -> bool:
        """True while the detector is unfitted or history is shorter than burn-in."""
        return assignment.is_burn_in or self._observations_seen <= self.config.burn_in_months

    @staticmethod
    def _as_burn_in(assignment: RegimeAssignment) -> RegimeAssignment:
        """Re-flag an assignment as burn-in without discarding its posterior."""
        return RegimeAssignment(
            as_of=assignment.as_of,
            label=RegimeLabel.CALM,
            posterior=np.asarray(assignment.posterior, dtype=float),
            raw_label=assignment.raw_label,
            is_burn_in=True,
        )
