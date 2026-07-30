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
    ) -> OptimizationResult:
        """Solve for target weights.

        Args:
            moments: Annualised mu and Sigma over the current asset set.
            weights_prev: Current holdings on the same asset ordering; used by the
                turnover term and as the starting point for warm-started search.
            rng: The only source of randomness the optimizer may use.
        """
