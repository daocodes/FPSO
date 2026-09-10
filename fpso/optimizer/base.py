"""The interface every allocator in the study implements.

FPSO and the two external baselines all go through :class:`PortfolioOptimizer`,
so the backtest engine is agnostic to which one it is driving and an arm change
is a config change.

The `rng` argument is not optional and is not stored on the optimizer. Every
stochastic component draws from the generator handed to it by the engine, whose
streams are derived from one master ``SeedSequence`` per (arm, seed). That is
what makes "backtested thirty times and averaged" a reproducible claim.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from fpso.data.moments import Moments


@dataclass(frozen=True)
class OptimizationResult:
    """The solved portfolio plus the diagnostics the paper reports."""

    weights: np.ndarray
    objective_value: float
    convergence: np.ndarray = field(default_factory=lambda: np.array([]))
    """Best-so-far objective at each iteration; empty for closed-form baselines."""
    solve_seconds: float = 0.0
    """Wall-clock solve time — RQ1 asks about tractability, so it is measured."""
    population: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    """Terminal swarm, `(n_particles, n_assets)`; empty for closed-form allocators.

    Population methods discard this and return one candidate. It is retained here
    because the spread across the surviving candidates is the only direct evidence
    the optimizer produces about whether its own answer was well determined, and
    that spread turns out to predict realised outcome dispersion
    (see :mod:`fpso.adaptive.features`)."""
    population_fitness: np.ndarray = field(default_factory=lambda: np.array([]))
    """Objective value of each candidate in `population`, same ordering."""

    @property
    def n_active(self) -> int:
        """Number of non-zero positions in the solution."""
        return int(np.sum(self.weights > 1e-8))


class PortfolioOptimizer(ABC):
    """Maps estimated moments and a starting portfolio onto target weights."""

    name: str = "optimizer"

    @abstractmethod
    def solve(
        self,
        moments: Moments,
        weights_prev: np.ndarray,
        rng: np.random.Generator,
        seeds: Sequence[np.ndarray] | None = None,
    ) -> OptimizationResult:
        """Solve for target weights.

        Args:
            moments: Annualised mu and Sigma over the current asset set.
            weights_prev: Current holdings on the same asset ordering; used by the
                turnover term and as the starting point for warm-started search.
            rng: The only source of randomness the optimizer may use.
            seeds: Optional extra starting points. Population methods may search
                from them; closed-form allocators ignore them. The optimizer
                attaches no meaning to a seed beyond "a candidate worth trying",
                which is what lets the regime archive stay outside the search.
        """


def iterations_to_converge(convergence: np.ndarray, fraction: float = 0.99) -> int:
    """First iteration reaching `fraction` of the run's final best objective.

    The metric behind the speed claim. Comparing wall-clock at a fixed iteration
    budget would say nothing — every arm runs the same 60 iterations — so the
    question is instead how many iterations a given solution quality costs. A
    starting point that is already good should reach the same plateau sooner.

    Objectives here can be negative (the fitness nets risk and turnover off
    return), so "99% of the final value" is taken on the run's own improvement
    range rather than as a ratio, which would invert whenever the sign does.

    Returns -1 for an empty trace, i.e. for the closed-form allocators.
    """
    if convergence.size == 0:
        return -1
    start, final = float(convergence[0]), float(convergence[-1])
    if not np.isfinite(start) or not np.isfinite(final) or final <= start:
        return 1  # No improvement to make: the seed was already at the plateau.
    threshold = start + fraction * (final - start)
    reached = np.flatnonzero(convergence >= threshold)
    return int(reached[0]) + 1 if reached.size else int(convergence.size)
