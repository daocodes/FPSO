"""Deterministic external baselines: 1/N and long-only minimum variance.

Both satisfy the same cardinality and box constraints as FPSO, so the comparison
is between allocation rules rather than between constraint sets. Both ignore the
`rng` argument, which is exactly why the experiment runner evaluates them once
instead of over 30 seeds.
"""

from __future__ import annotations

import time

import numpy as np

from fpso.config.schema import FPSOParams
from fpso.data.moments import Moments
from fpso.optimizer.base import OptimizationResult, PortfolioOptimizer
from fpso.optimizer.constraints import ConstraintSet, SimplexBoxCardinalityRepair
from fpso.optimizer.objective import MeanVarianceTurnover, PenalizedObjective


class _ConstrainedAllocator(PortfolioOptimizer):
    """Shared plumbing: build the repair operator and score the final portfolio."""

    def __init__(self, params: FPSOParams):
        self.params = params

    def _repair_for(self, n_assets: int) -> SimplexBoxCardinalityRepair:
        return SimplexBoxCardinalityRepair(
            ConstraintSet(
                n_assets=n_assets,
                max_assets=self.params.max_assets,
                max_weight=self.params.max_weight,
            )
        )

    def _score(
        self, weights: np.ndarray, moments: Moments, weights_prev: np.ndarray
    ) -> float:
        """Evaluate the same objective FPSO maximizes, for a like-for-like number."""
        constraints = ConstraintSet(
            n_assets=len(moments.assets),
            max_assets=self.params.max_assets,
            max_weight=self.params.max_weight,
        )
        objective = PenalizedObjective(
            base=MeanVarianceTurnover(
                mu=moments.mu,
                sigma=moments.sigma,
                weights_prev=np.asarray(weights_prev, dtype=float),
                lambda_v=self.params.lambda_v,
                lambda_t=self.params.lambda_t,
            ),
            constraints=constraints,
            params=self.params,
        )
        return objective.score(weights)


class EqualWeightAllocator(_ConstrainedAllocator):
    """1/N over the `max_assets` largest-expected-return names in the universe.

    Ranking by mu keeps the cardinality constraint binding in a defensible way;
    with `max_assets >= n_assets` this is plain 1/N over the whole universe.
    """

    name = "equal_weight"

    def solve(self, moments, weights_prev, rng) -> OptimizationResult:
        started = time.perf_counter()
        n_assets = len(moments.assets)
        k = min(self.params.max_assets, n_assets)

        selected = np.argsort(-moments.mu)[:k]
        raw = np.zeros(n_assets)
        raw[selected] = 1.0

        weights = self._repair_for(n_assets).repair(raw)
        return OptimizationResult(
            weights=weights,
            objective_value=self._score(weights, moments, weights_prev),
            solve_seconds=time.perf_counter() - started,
        )


class MinimumVarianceAllocator(_ConstrainedAllocator):
    """Long-only minimum variance, solved by projected gradient descent.

    Projected gradient (rather than a QP solver) keeps the dependency footprint
    small and lets the *same* repair operator enforce the cardinality and box
    constraints that make the problem non-convex — so this baseline is subject to
    identical constraints as FPSO rather than to a relaxed version of them.
    """

    name = "min_variance"

    def __init__(self, params: FPSOParams, max_iter: int = 500, step_size: float = 0.05):
        super().__init__(params)
        self.max_iter = max_iter
        self.step_size = step_size

    def solve(self, moments, weights_prev, rng) -> OptimizationResult:
        started = time.perf_counter()
        n_assets = len(moments.assets)
        repair = self._repair_for(n_assets)

        weights = repair.repair(np.full(n_assets, 1.0 / n_assets))
        # Normalising the step by the largest eigenvalue proxy (the trace) keeps
        # the descent stable across universes with very different volatility.
        scale = self.step_size / max(np.trace(moments.sigma) / n_assets, 1e-12)

        for _ in range(self.max_iter):
            gradient = 2.0 * moments.sigma @ weights
            candidate = repair.repair(weights - scale * gradient)
            if np.max(np.abs(candidate - weights)) < 1e-10:
                weights = candidate
                break
            weights = candidate

        return OptimizationResult(
            weights=weights,
            objective_value=self._score(weights, moments, weights_prev),
            solve_seconds=time.perf_counter() - started,
        )
