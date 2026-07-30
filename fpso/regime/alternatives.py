"""Ablation detectors that answer "why an HMM?" before a reviewer asks it.

:class:`GaussianMixtureDetector` clusters the same features with no temporal
structure, isolating the contribution of the Markov transition model.
:class:`VolatilityQuantileDetector` is a transparent, learning-free rule; if it
matches the HMM, the "learned" part of the contribution is not doing work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpso.config.schema import RegimeConfig, RegimeLabel
from fpso.regime.base import RegimeAssignment, RegimeDetector
from fpso.regime.hmm import VOLATILITY_FEATURE


class GaussianMixtureDetector(RegimeDetector):
    """Gaussian mixture over the same features — identical inputs, no Markov chain."""

    name = "gmm"

    def __init__(self, config: RegimeConfig):
        self.config = config
        self.n_states = config.n_states
        self._model = None
        self._state_order: np.ndarray | None = None
        self._last_fit: pd.Timestamp | None = None

    def fit(self, features: pd.DataFrame, as_of: pd.Timestamp) -> None:
        if not self._refit_due(as_of):
            return
        history = self._causal_slice(features, as_of)
        if len(history) < max(252, 60 * self.n_states):
            return

        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(
            n_components=self.n_states,
            covariance_type=self.config.covariance_type,
            random_state=self.config.detector_seed,
            n_init=3,
        )
        model.fit(history.to_numpy(dtype=float))

        self._model = model
        self._state_order = self._canonical_state_order(history)
        self._last_fit = as_of

    def _refit_due(self, as_of: pd.Timestamp) -> bool:
        if self._model is None or self._last_fit is None:
            return True
        months = (as_of.year - self._last_fit.year) * 12 + (as_of.month - self._last_fit.month)
        return months >= self.config.refit_months

    def _canonical_state_order(self, history: pd.DataFrame) -> np.ndarray:
        """Same volatility-ordering rule as the HMM, so labels mean the same thing."""
        columns = list(history.columns)
        axis = columns.index(VOLATILITY_FEATURE) if VOLATILITY_FEATURE in columns else None
        key = self._model.means_[:, axis] if axis is not None else self._model.means_.mean(axis=1)
        ranked = np.argsort(key)
        order = np.empty(self.n_states, dtype=int)
        order[ranked] = np.arange(self.n_states)
        return order

    def infer(self, features: pd.DataFrame, as_of: pd.Timestamp) -> RegimeAssignment:
        history = self._causal_slice(features, as_of)
        if self._model is None or history.empty:
            return _burn_in_assignment(as_of)

        latest = history.to_numpy(dtype=float)[-1].reshape(1, -1)
        posterior_raw = self._model.predict_proba(latest)[0]
        posterior = np.zeros(len(RegimeLabel))
        for fitted, canonical in enumerate(self._state_order):
            posterior[canonical] = posterior_raw[fitted]
        label = RegimeLabel(int(np.argmax(posterior)))
        return RegimeAssignment(as_of, label, posterior, label, is_burn_in=False)


class VolatilityQuantileDetector(RegimeDetector):
    """Terciles of trailing realized volatility — the transparent, unlearned rule.

    The cut points are expanding quantiles of the feature's own history, so the
    rule is still causal, but nothing is estimated beyond two order statistics.
    """

    name = "volq"

    def __init__(self, config: RegimeConfig, quantiles: tuple[float, float] = (0.6, 0.9)):
        self.config = config
        self.n_states = 3
        self.quantiles = quantiles

    def fit(self, features: pd.DataFrame, as_of: pd.Timestamp) -> None:
        return None  # Nothing is learned; cut points are recomputed at inference.

    def infer(self, features: pd.DataFrame, as_of: pd.Timestamp) -> RegimeAssignment:
        history = self._causal_slice(features, as_of)
        if history.empty or VOLATILITY_FEATURE not in history.columns:
            return _burn_in_assignment(as_of)

        volatility = history[VOLATILITY_FEATURE]
        if len(volatility) < 252:
            return _burn_in_assignment(as_of)

        turbulent_cut, crisis_cut = volatility.quantile(list(self.quantiles))
        latest = float(volatility.iloc[-1])
        if latest >= crisis_cut:
            label = RegimeLabel.CRISIS
        elif latest >= turbulent_cut:
            label = RegimeLabel.TURBULENT
        else:
            label = RegimeLabel.CALM

        posterior = np.zeros(len(RegimeLabel))
        posterior[int(label)] = 1.0
        return RegimeAssignment(as_of, label, posterior, label, is_burn_in=False)


def _burn_in_assignment(as_of: pd.Timestamp) -> RegimeAssignment:
    """Flat posterior flagged as burn-in; the engine falls back to static params."""
    posterior = np.full(len(RegimeLabel), 1.0 / len(RegimeLabel))
    return RegimeAssignment(
        as_of=as_of,
        label=RegimeLabel.CALM,
        posterior=posterior,
        raw_label=RegimeLabel.CALM,
        is_burn_in=True,
    )
