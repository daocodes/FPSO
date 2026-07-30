"""The regime-detector contract.

Every detector, including the falsification controls, implements
:class:`RegimeDetector`. The two-method split (`fit` then `infer`) is not
cosmetic: it forces the causality boundary to be an explicit argument (`as_of`)
rather than an implicit property of how the caller happened to slice its data.
``tests/unit/test_lookahead.py`` exploits this by corrupting all data after a
date and asserting every decision at or before that date is unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpso.config.schema import RegimeLabel


@dataclass(frozen=True)
class RegimeAssignment:
    """The detector's answer at one rebalance date."""

    as_of: pd.Timestamp
    label: RegimeLabel
    """The *acting* label: post-smoothing, and what the parameter policy consumes."""
    posterior: np.ndarray
    """P(s_t | x_{1:t}) over canonical labels; uniform for non-probabilistic detectors."""
    raw_label: RegimeLabel
    """The pre-smoothing label, retained for the dwell-time diagnostics (Figure F4)."""
    is_burn_in: bool = False
    """True while history is too short to condition on; the caller must fall back
    to the static parameterization."""

    @property
    def confidence(self) -> float:
        """Posterior mass on the acting label."""
        return float(self.posterior[int(self.label)])


class RegimeDetector(ABC):
    """Estimates the latent market state from causal features."""

    name: str = "detector"
    n_states: int = 3
    requires_smoothing: bool = True
    """Whether the policy should run this detector's labels through the
    minimum-dwell smoother. False for detectors whose output is already a smoothed
    label path (the shuffled control replays real episodes; a constant label has
    nothing to smooth) — re-smoothing those would delay every switch by a
    rebalance and distort the dwell distribution the control exists to match."""

    @abstractmethod
    def fit(self, features: pd.DataFrame, as_of: pd.Timestamp) -> None:
        """Fit using only rows indexed on or before `as_of`."""

    @abstractmethod
    def infer(self, features: pd.DataFrame, as_of: pd.Timestamp) -> RegimeAssignment:
        """Return the filtered state estimate at `as_of`.

        Implementations must not read rows indexed after `as_of`.
        """

    def posterior_series(
        self, features: pd.DataFrame, as_of: pd.Timestamp
    ) -> pd.DataFrame | None:
        """P(s_d | x_{1:as_of}) for every day d <= as_of, in canonical label order.

        Returns None for detectors with no probabilistic state model. Consumed by
        :class:`~fpso.data.moments.RegimeWeightedMoments`, which weights each
        historical day by how much its regime resembles the current one.
        """
        return None

    @staticmethod
    def _causal_slice(features: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        """History available at `as_of`, with incomplete leading rows dropped.

        Every detector routes its data access through this one helper, so the
        no-lookahead property is enforced in a single place instead of being
        re-argued per detector.
        """
        return features.loc[:as_of].dropna()


class NullDetector(RegimeDetector):
    """Always reports CALM. Backs the `static` arm without special-casing the engine."""

    name = "none"

    def fit(self, features: pd.DataFrame, as_of: pd.Timestamp) -> None:
        return None

    def infer(self, features: pd.DataFrame, as_of: pd.Timestamp) -> RegimeAssignment:
        posterior = np.zeros(len(RegimeLabel))
        posterior[int(RegimeLabel.CALM)] = 1.0
        return RegimeAssignment(
            as_of=as_of,
            label=RegimeLabel.CALM,
            posterior=posterior,
            raw_label=RegimeLabel.CALM,
            is_burn_in=False,
        )
