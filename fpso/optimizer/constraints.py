"""Feasibility repair for the simplex-with-box-and-cardinality constraint set.

Repair, not penalty, is what actually enforces feasibility in this codebase: the
optimizer projects every candidate before scoring it, so the penalty terms in the
objective act only as a safety net for the (unreachable) infeasible case. The
paper's methods section should describe it that way — see
:class:`fpso.optimizer.objective.PenalizedObjective` for the counterpart.

The projection is the classic "waterfilling" argument: normalise onto the
simplex, clamp anything above its box bound, redistribute the freed mass among
the still-unclamped assets, repeat. It terminates in at most `n` passes because
each pass clamps at least one asset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConstraintSet:
    """Long-only, fully-invested, at most `max_assets` names, each at most `max_weight`."""

    n_assets: int
    max_assets: int
    max_weight: float

    def __post_init__(self) -> None:
        if self.n_assets < 1:
            raise ValueError("n_assets must be positive.")
        if not 1 <= self.max_assets:
            raise ValueError("max_assets must be at least 1.")
        if not 0.0 < self.max_weight <= 1.0:
            raise ValueError("max_weight must lie in (0, 1].")

    @property
    def effective_cardinality(self) -> int:
        """Cardinality bound, clipped to the number of assets actually available."""
        return int(min(self.max_assets, self.n_assets))

    @property
    def is_feasible(self) -> bool:
        """Whether a fully-invested portfolio satisfying both bounds exists."""
        return self.effective_cardinality * self.max_weight >= 1.0 - 1e-12


class SimplexBoxCardinalityRepair:
    """Projects an arbitrary real vector onto the feasible portfolio set.

    Guarantees on the returned vector `w` (asserted in ``tests/unit/test_constraints.py``):
    ``sum(w) == 1``, ``0 <= w <= max_weight``, ``count(w > 0) <= max_assets``, and
    idempotence ``repair(repair(w)) == repair(w)``.
    """

    def __init__(self, constraints: ConstraintSet):
        if not constraints.is_feasible:
            raise ValueError(
                f"Infeasible constraint set: max_assets ({constraints.max_assets}) * "
                f"max_weight ({constraints.max_weight}) < 1, so no fully-invested "
                "portfolio exists."
            )
        self.constraints = constraints

    def repair(self, raw_weights: np.ndarray) -> np.ndarray:
        """Return the feasible portfolio closest in spirit to `raw_weights`."""
        w = np.asarray(raw_weights, dtype=float).reshape(-1)
        if w.shape[0] != self.constraints.n_assets:
            raise ValueError(
                f"Expected {self.constraints.n_assets} weights, got {w.shape[0]}."
            )

        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.maximum(w, 0.0)
        w = self._select_top_k(w)
        return self._project_onto_capped_simplex(w)

    def _select_top_k(self, w: np.ndarray) -> np.ndarray:
        """Zero out all but the `max_assets` largest weights.

        Ties are broken by index so the projection is deterministic; without that
        the same input could yield different portfolios across NumPy versions,
        which would break the reproducibility guarantee.
        """
        k = self.constraints.effective_cardinality
        if k >= w.shape[0]:
            return w
        # Sorting on (-weight, index) makes the tie-break explicit and stable.
        order = np.lexsort((np.arange(w.shape[0]), -w))
        kept = np.zeros_like(w)
        kept[order[:k]] = w[order[:k]]
        return kept

    def _project_onto_capped_simplex(self, w: np.ndarray) -> np.ndarray:
        """Scale the surviving weights to sum to 1 without exceeding `max_weight`."""
        n = self.constraints.n_assets
        k = self.constraints.effective_cardinality
        cap = self.constraints.max_weight

        support = w > 0
        if not support.any():
            # Degenerate input (all-zero, all-negative, all-NaN): fall back to an
            # equal-weight portfolio over the first k assets, which is feasible by
            # construction because k * cap >= 1 is checked in __init__.
            w = np.zeros(n)
            w[:k] = 1.0
            support = w > 0

        weights = np.zeros(n, dtype=float)
        active = support.copy()
        remaining_mass = 1.0

        while active.any() and remaining_mass > 1e-12:
            share = w[active] / w[active].sum()
            proposed = remaining_mass * share
            over_cap = proposed > cap

            if not over_cap.any():
                weights[active] = proposed
                remaining_mass = 0.0
                break

            clamped = np.where(active)[0][over_cap]
            weights[clamped] = cap
            active[clamped] = False
            remaining_mass = 1.0 - weights.sum()

        if remaining_mass > 1e-12:
            # Every supported asset is at its cap and the portfolio is still short
            # of fully invested. Widen the support to the next-largest names, which
            # k * cap >= 1 guarantees exists.
            weights = self._fill_remaining_at_cap(weights, cap)

        total = weights.sum()
        return weights / total if total > 0 else np.full(n, 1.0 / n)

    def _fill_remaining_at_cap(self, weights: np.ndarray, cap: float) -> np.ndarray:
        """Top up an under-invested portfolio using currently-unheld assets."""
        for index in np.argsort(weights):
            shortfall = 1.0 - weights.sum()
            if shortfall <= 1e-12:
                break
            weights[index] = min(cap, weights[index] + shortfall)
        return weights
