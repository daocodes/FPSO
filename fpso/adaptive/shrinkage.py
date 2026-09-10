"""Learning how much the optimizer should trust its own return forecasts.

Solving the stated objective *better* makes the portfolio *worse* gross of costs
(-0.068 Sharpe when exact weighting replaces the swarm's). `mu` is a trailing
252-day sample mean, and optimising harder against a poor forecast amplifies its
estimation error. FPSO's failure to converge was accidentally regularising it.

Deliberate regularisation dominates the accident: shrinking `mu` toward its
cross-sectional mean before the exact solve recovers +0.108 Sharpe gross on ~6%
less turnover, so it is not a turnover effect. But the best intensity is not a
constant — it depends on how unreliable `mu` is at that moment, which is exactly
what :mod:`fpso.adaptive.features` measures.

This module learns the map from uncertainty to shrinkage intensity.

**Causality.** Every policy here is fit strictly on rebalances that precede the
one it acts on. `walk_forward` never passes a row to `fit` that is not already in
the past at prediction time, and `tests/unit/test_adaptive.py` asserts that
perturbing future outcomes leaves earlier decisions bit-identical. That is the
same property the harness's lookahead proof establishes for the backtest, and it
must hold here or the result is worthless.

**Design of the target.** For each past rebalance the grid records what each
shrinkage intensity would have realised, so the ex-post best intensity is known
*for the past*. The policy imitates that. The target is discrete and noisy, so
the model is a heavily regularised linear one — with ~100 usable observations
anything richer fits the noise, and the point of the exercise is a signal that
survives its own controls, not a fitted curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ShrinkageGrid",
    "RidgeShrinkagePolicy",
    "walk_forward_learned",
    "walk_forward_best_constant",
    "oracle_shrinkage",
    "shuffled_shrinkage",
]

DEFAULT_RIDGE = 10.0
DEFAULT_MIN_TRAIN = 36


@dataclass(frozen=True)
class ShrinkageGrid:
    """What each shrinkage intensity would have realised at each rebalance.

    Attributes:
        deltas: `(n_deltas,)` shrinkage intensities, ascending.
        features: `(n_dates, n_features)` ex-ante uncertainty features.
        outcomes: `(n_dates, n_deltas)` realised outcome of choosing that
            intensity at that rebalance — higher is better. Only ever read for
            rows strictly before the decision date.
    """

    deltas: np.ndarray
    features: np.ndarray
    outcomes: np.ndarray

    def __post_init__(self) -> None:
        if self.features.shape[0] != self.outcomes.shape[0]:
            raise ValueError(
                f"{self.features.shape[0]} feature rows against "
                f"{self.outcomes.shape[0]} outcome rows."
            )
        if self.outcomes.shape[1] != len(self.deltas):
            raise ValueError(
                f"{self.outcomes.shape[1]} outcome columns against "
                f"{len(self.deltas)} deltas."
            )

    @property
    def n_dates(self) -> int:
        return self.features.shape[0]

    def best_delta_per_date(self) -> np.ndarray:
        """Ex-post optimal intensity at each rebalance. Uses the realised
        outcome, so it is causally invalid except as a training target for
        *past* rows or as an explicit oracle upper bound."""
        return self.deltas[np.argmax(self.outcomes, axis=1)]


class RidgeShrinkagePolicy:
    """Ridge regression from uncertainty features onto shrinkage intensity.

    Features are standardised on the training rows only — using the full-sample
    mean and scale would leak the future through the normalisation, which is a
    subtle enough channel to be worth stating explicitly.
    """

    def __init__(self, ridge: float = DEFAULT_RIDGE):
        self.ridge = ridge
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._coefficients: np.ndarray | None = None
        self._intercept: float = 0.0

    def fit(self, features: np.ndarray, targets: np.ndarray) -> RidgeShrinkagePolicy:
        X = np.atleast_2d(np.asarray(features, dtype=float))
        y = np.asarray(targets, dtype=float).ravel()
        self._mean = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0)
        # A constant feature carries no information; a zero scale would make the
        # standardisation blow up, so it is neutralised rather than dropped, which
        # keeps the coefficient vector aligned with FEATURE_NAMES.
        self._scale = np.where(scale > 1e-12, scale, 1.0)
        Z = (X - self._mean) / self._scale
        self._intercept = float(y.mean())
        centred = y - self._intercept
        gram = Z.T @ Z + self.ridge * np.eye(Z.shape[1])
        self._coefficients = np.linalg.solve(gram, Z.T @ centred)
        return self

    def predict(self, features: np.ndarray, *, lo: float = 0.0, hi: float = 1.0) -> float:
        """Predicted intensity for one rebalance, clipped to the grid's range."""
        if self._coefficients is None:
            raise RuntimeError("Policy used before fit.")
        z = (np.asarray(features, dtype=float).ravel() - self._mean) / self._scale
        return float(np.clip(self._intercept + z @ self._coefficients, lo, hi))

    @property
    def coefficients(self) -> np.ndarray:
        """Standardised coefficients, for reporting which features drive the map."""
        if self._coefficients is None:
            raise RuntimeError("Policy used before fit.")
        return self._coefficients.copy()


def walk_forward_learned(
    grid: ShrinkageGrid,
    *,
    ridge: float = DEFAULT_RIDGE,
    min_train: int = DEFAULT_MIN_TRAIN,
    fallback: float | None = None,
) -> np.ndarray:
    """Learned intensity at each rebalance, fit only on strictly earlier ones.

    Args:
        grid: Features and per-intensity outcomes.
        ridge: Regularisation strength.
        min_train: Rebalances required before the model is trusted. Until then
            the fallback is used, which is what an operator without history would
            have to do.
        fallback: Intensity used during the burn-in. Defaults to the grid median,
            i.e. no view.

    Returns:
        `(n_dates,)` chosen intensity, snapped to the nearest grid point so the
        realised outcome is one the grid actually priced.
    """
    lo, hi = float(grid.deltas.min()), float(grid.deltas.max())
    if fallback is None:
        fallback = float(np.median(grid.deltas))

    targets = grid.best_delta_per_date()
    chosen = np.empty(grid.n_dates, dtype=float)
    for i in range(grid.n_dates):
        if i < min_train:
            chosen[i] = fallback
            continue
        policy = RidgeShrinkagePolicy(ridge=ridge).fit(grid.features[:i], targets[:i])
        chosen[i] = policy.predict(grid.features[i], lo=lo, hi=hi)
    return _snap(chosen, grid.deltas)


def walk_forward_best_constant(
    grid: ShrinkageGrid, *, min_train: int = DEFAULT_MIN_TRAIN, fallback: float | None = None
) -> np.ndarray:
    """The honest baseline: the intensity that has worked best *so far*.

    This is the comparison that matters. Beating a fixed intensity chosen with
    hindsight proves nothing, and beating no shrinkage at all proves only that
    shrinkage helps. The question is whether *conditioning* the intensity on the
    optimizer's uncertainty beats re-using whatever has worked to date.
    """
    if fallback is None:
        fallback = float(np.median(grid.deltas))
    chosen = np.empty(grid.n_dates, dtype=float)
    for i in range(grid.n_dates):
        if i < min_train:
            chosen[i] = fallback
            continue
        chosen[i] = grid.deltas[int(np.argmax(grid.outcomes[:i].sum(axis=0)))]
    return _snap(chosen, grid.deltas)


def oracle_shrinkage(grid: ShrinkageGrid) -> np.ndarray:
    """Upper bound: the ex-post best intensity at every rebalance.

    Causally invalid by construction and reported as a ceiling, in the same role
    as the study's existing lookahead arms. If the learned policy approaches it,
    the features carry the available signal; if the oracle itself is barely above
    the best constant, there was little to learn and the paper should say so.
    """
    return grid.best_delta_per_date()


def shuffled_shrinkage(deltas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Signal-free control: the same intensities, applied on the wrong dates.

    Preserves the marginal distribution exactly and destroys only the timing, so
    any gap between this and the treatment is attributable to *when* the policy
    shrinks rather than to how much it shrinks on average. Mirrors the permutation
    control in :mod:`fpso.regime.controls`.
    """
    return rng.permutation(np.asarray(deltas, dtype=float))


def _snap(values: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Round each choice to the nearest priced grid point."""
    idx = np.abs(np.asarray(values, dtype=float)[:, None] - deltas[None, :]).argmin(axis=1)
    return deltas[idx]
