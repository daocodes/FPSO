"""FPSO-CW: the swarm selects the assets, an exact solver sets the weights.

FPSO searches selection and weighting jointly and is measurably bad at the second
half. Re-solving weights exactly on the swarm's own 20 names raises the objective
by ~130% at 100% of rebalances and beats the best portfolio FPSO reaches with a
10x search budget — so the weighting, not the selection, is what more search
cannot fix (:mod:`fpso.optimizer.exact`).

Splitting the problem also makes the search better posed. Under the joint scheme a
good selection with unlucky weights scores below a mediocre selection with lucky
weights, so the fitness signal the swarm follows is contaminated by weighting
noise. Once weights are exact, fitness depends on the *support alone* — which is
both a cleaner search signal and memoisable, since two candidates holding the same
names have the same score.

The `shrinkage` parameter pulls `mu` toward its cross-sectional mean before the
exact solve. This is not a detail: solving the raw objective better makes the
portfolio worse gross of costs, because `mu` is a trailing sample mean and
optimising harder against it amplifies estimation error. See
:mod:`fpso.adaptive.shrinkage` for the learned setting of this parameter and for
the controls a claimed improvement has to survive.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from fpso.config.schema import FPSOParams
from fpso.data.moments import Moments
from fpso.optimizer.base import OptimizationResult
from fpso.optimizer.exact import solve_weights_on_support
from fpso.optimizer.fpso import FPSOOptimizer

__all__ = ["FPSOCWOptimizer", "shrink_mu"]

_ACTIVE = 1e-8


def shrink_mu(mu: np.ndarray, intensity: float) -> np.ndarray:
    """Pull expected returns toward their cross-sectional mean.

    `intensity = 0` leaves the forecast untouched; `intensity = 1` discards it
    entirely, leaving a minimum-variance problem on the support. The target is the
    cross-sectional mean rather than zero so that the overall return level — which
    is estimated far more reliably than the cross-section — is preserved.
    """
    if intensity <= 0.0:
        return mu
    return (1.0 - intensity) * mu + intensity * float(np.mean(mu))


class FPSOCWOptimizer(FPSOOptimizer):
    """FPSO for selection, exact convex programming for weights."""

    name = "fpso_cw"

    def __init__(self, params: FPSOParams, shrinkage: float = 0.0):
        super().__init__(params)
        self.shrinkage = float(shrinkage)

    def solve(
        self,
        moments: Moments,
        weights_prev: np.ndarray,
        rng: np.random.Generator,
        seeds: Sequence[np.ndarray] | None = None,
    ) -> OptimizationResult:
        started = time.perf_counter()
        search = super().solve(moments, weights_prev, rng, seeds)
        support = np.flatnonzero(search.weights > _ACTIVE)
        weights, objective = self.solve_weights(
            support, moments, np.asarray(weights_prev, dtype=float), self.shrinkage
        )
        return OptimizationResult(
            weights=weights,
            objective_value=objective,
            convergence=search.convergence,
            solve_seconds=time.perf_counter() - started,
            population=search.population,
            population_fitness=search.population_fitness,
        )

    def solve_over_shrinkage(
        self,
        moments: Moments,
        weights_prev: np.ndarray,
        rng: np.random.Generator,
        intensities: Sequence[float],
        seeds: Sequence[np.ndarray] | None = None,
    ) -> tuple[OptimizationResult, dict[float, np.ndarray]]:
        """One search, then one exact solve per shrinkage intensity.

        The swarm selects the support using the unshrunk objective, so the support
        does not depend on the intensity and a whole grid can be priced from a
        single search. That is what makes the walk-forward policy in
        :mod:`fpso.adaptive.shrinkage` affordable: the grid is computed once and
        the policy only chooses among rows of it.

        Returns the underlying search result and a mapping from intensity to the
        weights it would have produced.
        """
        search = super().solve(moments, weights_prev, rng, seeds)
        support = np.flatnonzero(search.weights > _ACTIVE)
        previous = np.asarray(weights_prev, dtype=float)
        by_intensity = {
            float(d): self.solve_weights(support, moments, previous, float(d))[0]
            for d in intensities
        }
        return search, by_intensity

    def solve_weights(
        self,
        support: np.ndarray,
        moments: Moments,
        weights_prev: np.ndarray,
        intensity: float,
    ) -> tuple[np.ndarray, float]:
        """Exact weights on `support`, with `mu` shrunk by `intensity`.

        Public because the adaptive experiment prices a grid of intensities
        against one search and needs to call it directly.
        """
        return solve_weights_on_support(
            support,
            shrink_mu(moments.mu, intensity),
            moments.sigma,
            weights_prev,
            lambda_v=self.params.lambda_v,
            lambda_t=self.params.lambda_t,
            max_weight=self.params.max_weight,
        )
