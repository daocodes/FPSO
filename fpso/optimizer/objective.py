"""The fitness function the swarm maximizes.

Two objects, matching the two equations in the paper:

* :class:`MeanVarianceTurnover` is R(w), the economic objective.
* :class:`PenalizedObjective` is Phi(w) = R(w) - rho' * violation(w).

Because every candidate is repaired before evaluation, the violation vector is
identically zero along the search path and Phi(w) == R(w) in practice. Rather
than hide that, :meth:`PenalizedObjective.violation` is exposed so the paper can
report a measured penalty-activation rate, and so the "penalty-only" ablation
(repair disabled) can be run through exactly the same objective code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from fpso.config.schema import FPSOParams
from fpso.optimizer.constraints import ConstraintSet


class Objective(ABC):
    """A scalar score for a candidate portfolio, higher is better."""

    @abstractmethod
    def score(self, weights: np.ndarray) -> float:
        """Evaluate the objective at `weights`."""


@dataclass(frozen=True)
class MeanVarianceTurnover(Objective):
    """R(w) = mu'w - lambda_v * sqrt(w'Sigma w) - lambda_t * ||w - w_prev||_1.

    Note the risk term is portfolio *standard deviation*, not variance, so
    `lambda_v` is unit-consistent with the return term and comparable across
    universes of different size.
    """

    mu: np.ndarray
    sigma: np.ndarray
    weights_prev: np.ndarray
    lambda_v: float
    lambda_t: float

    def score(self, weights: np.ndarray) -> float:
        w = np.asarray(weights, dtype=float).reshape(-1)
        expected_return = float(self.mu @ w)
        variance = float(w @ self.sigma @ w)
        volatility = float(np.sqrt(max(variance, 0.0)))
        turnover = float(np.abs(w - self.weights_prev).sum())
        return expected_return - self.lambda_v * volatility - self.lambda_t * turnover


@dataclass(frozen=True)
class ConstraintViolation:
    """How far a candidate is from feasible, decomposed by constraint."""

    budget: float
    box: float
    cardinality: float

    @property
    def is_feasible(self) -> bool:
        return max(self.budget, self.box, self.cardinality) <= 1e-9


class PenalizedObjective(Objective):
    """Phi(w) = R(w) minus weighted constraint violations."""

    def __init__(
        self,
        base: MeanVarianceTurnover,
        constraints: ConstraintSet,
        params: FPSOParams,
    ):
        self.base = base
        self.constraints = constraints
        self.rho_budget = params.rho_budget
        self.rho_box = params.rho_box
        self.rho_cardinality = params.rho_cardinality

    def violation(self, weights: np.ndarray) -> ConstraintViolation:
        """Per-constraint violation magnitudes at `weights`."""
        w = np.asarray(weights, dtype=float).reshape(-1)
        budget = abs(float(w.sum()) - 1.0)
        below = np.maximum(0.0, -w)
        above = np.maximum(0.0, w - self.constraints.max_weight)
        box = float(np.sum(below**2 + above**2))
        active = int(np.sum(np.abs(w) > 1e-8))
        excess = max(0, active - self.constraints.effective_cardinality)
        return ConstraintViolation(budget=budget, box=box, cardinality=float(excess**2))

    def score(self, weights: np.ndarray) -> float:
        violation = self.violation(weights)
        return (
            self.base.score(weights)
            - self.rho_budget * violation.budget
            - self.rho_box * violation.box
            - self.rho_cardinality * violation.cardinality
        )
