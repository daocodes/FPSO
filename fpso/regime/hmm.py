"""Gaussian hidden Markov model detector — the headline regime mechanism.

Three implementation details carry the paper's causality claim:

**Canonical state labels.** HMM state indices are arbitrary and permute between
refits. Left alone, "state 2" would mean something different in 2013 than in
2019 and the regime-to-parameter map would silently become noise. After every
fit the states are sorted by their mean volatility feature and relabelled
CALM < TURBULENT < CRISIS. The mapping is deterministic and interpretable.

**Filtered, not smoothed, inference.** At the final observation of a sequence the
smoothed posterior equals the filtered one (gamma_T = alpha_T, since there is no
future evidence to condition on). So ``predict_proba(X[:t+1])[-1]`` is exactly
P(s_t | x_{1:t}). Historical label *series* used for plotting are smoothed and
labelled as such in the figure captions; labels used for *decisions* are always
this last-row estimate.

**Periodic refit with warm start.** Refitting monthly churns parameters for no
benefit. The model is refit every `refit_months` and warm-started from the
previous solution; inference still runs at every rebalance.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from fpso.config.schema import RegimeConfig, RegimeLabel
from fpso.regime.base import RegimeAssignment, RegimeDetector

VOLATILITY_FEATURE = "rv_21"
"""Feature used to order latent states. Must be one of the configured features."""


class GaussianHMMDetector(RegimeDetector):
    """Three-state Gaussian HMM over standardized market features."""

    name = "hmm"

    def __init__(self, config: RegimeConfig):
        self.config = config
        self.n_states = config.n_states
        self._model = None
        self._state_order: np.ndarray | None = None
        self._last_fit: pd.Timestamp | None = None
        self._feature_names: list[str] | None = None

    # ------------------------------------------------------------------ fit --

    def fit(self, features: pd.DataFrame, as_of: pd.Timestamp) -> None:
        """(Re)fit on all history through `as_of`, if the refit cadence is due."""
        if not self._refit_due(as_of):
            return
        history = self._causal_slice(features, as_of)
        if len(history) < self._minimum_fit_observations():
            return

        self._feature_names = list(history.columns)
        self._model = self._fit_model(history.to_numpy(dtype=float))
        self._state_order = self._canonical_state_order(history)
        self._last_fit = as_of

    def _refit_due(self, as_of: pd.Timestamp) -> bool:
        """True on the first call and every `refit_months` months thereafter."""
        if self._model is None or self._last_fit is None:
            return True
        months_elapsed = (as_of.year - self._last_fit.year) * 12 + (
            as_of.month - self._last_fit.month
        )
        return months_elapsed >= self.config.refit_months

    def _minimum_fit_observations(self) -> int:
        """Enough data for a diagonal-covariance Gaussian HMM to be identified."""
        return max(252, 60 * self.n_states)

    def _fit_model(self, observations: np.ndarray):
        """Fit hmmlearn's GaussianHMM, warm-starting from the previous solution."""
        from hmmlearn.hmm import GaussianHMM

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.config.covariance_type,
            n_iter=200,
            tol=1e-4,
            random_state=self.config.detector_seed,
            init_params="" if self._model is not None else "stmc",
            params="stmc",
        )
        if self._model is not None:
            # Warm start keeps consecutive refits close to one another, which in
            # turn keeps the canonical relabelling stable across refit boundaries.
            model.startprob_ = self._model.startprob_
            model.transmat_ = self._model.transmat_
            model.means_ = self._model.means_
            model.covars_ = self._model._covars_

        with warnings.catch_warnings():
            # hmmlearn warns on non-convergence within n_iter; the warm start plus
            # a 200-iteration budget makes that benign and it would otherwise flood
            # the log with one warning per refit.
            warnings.simplefilter("ignore")
            model.fit(observations)
        return model

    def _canonical_state_order(self, history: pd.DataFrame) -> np.ndarray:
        """Permutation mapping fitted state index -> canonical RegimeLabel.

        States are ordered by their fitted mean on the volatility feature, so
        index 0 is always the least volatile state. Returns an array `order` with
        ``order[fitted_index] = canonical_index``.
        """
        if VOLATILITY_FEATURE in history.columns:
            axis = list(history.columns).index(VOLATILITY_FEATURE)
            key = self._model.means_[:, axis]
        else:  # pragma: no cover - only if the feature set is reconfigured
            key = self._model.means_.mean(axis=1)

        ranked = np.argsort(key)
        order = np.empty(self.n_states, dtype=int)
        order[ranked] = np.arange(self.n_states)
        return order

    # ---------------------------------------------------------------- infer --

    def infer(self, features: pd.DataFrame, as_of: pd.Timestamp) -> RegimeAssignment:
        """Filtered posterior over canonical labels at `as_of`."""
        history = self._causal_slice(features, as_of)
        if self._model is None or history.empty:
            return self._uninformative(as_of)

        posterior_raw = self._model.predict_proba(history.to_numpy(dtype=float))[-1]
        posterior = self._to_canonical(posterior_raw)
        label = RegimeLabel(int(np.argmax(posterior)))
        return RegimeAssignment(
            as_of=as_of,
            label=label,
            posterior=posterior,
            raw_label=label,
            is_burn_in=False,
        )

    def _to_canonical(self, posterior_raw: np.ndarray) -> np.ndarray:
        """Permute a posterior over fitted states into canonical label order."""
        posterior = np.zeros(len(RegimeLabel), dtype=float)
        for fitted_index, canonical_index in enumerate(self._state_order):
            posterior[canonical_index] = posterior_raw[fitted_index]
        return posterior

    def _uninformative(self, as_of: pd.Timestamp) -> RegimeAssignment:
        """CALM with a flat posterior, flagged as burn-in so the caller falls back."""
        posterior = np.full(len(RegimeLabel), 1.0 / len(RegimeLabel))
        return RegimeAssignment(
            as_of=as_of,
            label=RegimeLabel.CALM,
            posterior=posterior,
            raw_label=RegimeLabel.CALM,
            is_burn_in=True,
        )

    # ---------------------------------------------------------- diagnostics --

    @property
    def transition_matrix(self) -> np.ndarray | None:
        """Fitted transition matrix in canonical label order (Figure F4)."""
        if self._model is None or self._state_order is None:
            return None
        order = np.argsort(self._state_order)
        return self._model.transmat_[np.ix_(order, order)]

    def posterior_series(
        self, features: pd.DataFrame, as_of: pd.Timestamp
    ) -> pd.DataFrame | None:
        """Smoothed posteriors for every day through `as_of`, canonically ordered.

        `predict_proba` conditions on the whole sequence it is given, and it is
        given only rows at or before `as_of` — so every value here is
        P(s_d | x_{1:as_of}), which uses no future data. That is exactly the
        quantity needed to ask "how much did day d look like today's regime?"
        """
        history = self._causal_slice(features, as_of)
        if self._model is None or history.empty:
            return None

        raw = self._model.predict_proba(history.to_numpy(dtype=float))
        canonical = np.zeros((len(history), len(RegimeLabel)), dtype=float)
        for fitted_index, canonical_index in enumerate(self._state_order):
            canonical[:, canonical_index] = raw[:, fitted_index]
        return pd.DataFrame(
            canonical, index=history.index, columns=[label.name for label in RegimeLabel]
        )

    def label_series(self, features: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Smoothed (Viterbi) label path through `as_of`, for plotting only.

        This uses the whole sliced history to label each date, so it must never feed
        a decision on any causally valid arm. Two callers are permitted: figures,
        and `OracleOOSDetector`, whose entire purpose is to be acausal and which is
        reported only as an upper bound.
        """
        history = self._causal_slice(features, as_of)
        if self._model is None or history.empty:
            return pd.Series(dtype=int)
        states = self._model.predict(history.to_numpy(dtype=float))
        return pd.Series(self._state_order[states], index=history.index, name="regime")
