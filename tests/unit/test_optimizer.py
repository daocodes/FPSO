"""FPSO search behaviour: feasibility, convergence, and the PSO term.

The last test is the one that matters for the paper's honesty: before this
refactor `Particle.velocity` and `Particle.best_weights` were written but never
read, so the "PSO" in FPSO was not present in the code. `pso_blend` now has to
change the search, and this asserts that it does.
"""

from __future__ import annotations

import numpy as np
import pytest

from fpso.baselines.allocators import EqualWeightAllocator, MinimumVarianceAllocator
from fpso.config.schema import FPSOParams
from fpso.data.moments import SampleMoments
from fpso.optimizer.constraints import ConstraintSet
from fpso.optimizer.fpso import FPSOOptimizer
from tests.unit.test_constraints import assert_feasible


@pytest.fixture
def moments(panel):
    window = panel.returns.loc["2010-01-01":"2010-12-31"]
    return SampleMoments(min_observations=100).estimate(window)


@pytest.fixture
def previous(moments):
    return np.full(len(moments.assets), 1.0 / len(moments.assets))


def test_solution_is_feasible(moments, previous, fast_params):
    result = FPSOOptimizer(fast_params).solve(moments, previous, np.random.default_rng(0))
    assert_feasible(
        result.weights,
        ConstraintSet(len(moments.assets), fast_params.max_assets, fast_params.max_weight),
    )


def test_objective_improves_monotonically(moments, previous, fast_params):
    """Best-so-far fitness can never decrease — it is a running maximum."""
    params = FPSOParams(**{**fast_params.__dict__, "max_iter": 30})
    result = FPSOOptimizer(params).solve(moments, previous, np.random.default_rng(0))

    assert len(result.convergence) == params.max_iter
    assert np.all(np.diff(result.convergence) >= -1e-12)
    assert result.objective_value == pytest.approx(result.convergence[-1])


def test_search_beats_its_own_starting_point(moments, previous, fast_params):
    """The optimizer must improve on the warm start, or it is not optimizing."""
    from fpso.optimizer.objective import MeanVarianceTurnover, PenalizedObjective

    params = FPSOParams(**{**fast_params.__dict__, "max_iter": 40, "num_particles": 20})
    constraints = ConstraintSet(len(moments.assets), params.max_assets, params.max_weight)
    objective = PenalizedObjective(
        MeanVarianceTurnover(moments.mu, moments.sigma, previous, params.lambda_v, params.lambda_t),
        constraints,
        params,
    )

    result = FPSOOptimizer(params).solve(moments, previous, np.random.default_rng(0))
    assert result.objective_value > objective.score(previous)


def test_pso_blend_changes_the_search(moments, previous, fast_params):
    """pso_blend must actually alter the trajectory, not sit unused.

    Before the refactor the PSO velocity terms were dead code and the paper's own
    description of the algorithm was not true of the implementation.
    """
    pure_firefly = FPSOParams(**{**fast_params.__dict__, "pso_blend": 0.0})
    hybrid = FPSOParams(**{**fast_params.__dict__, "pso_blend": 0.8})

    firefly_result = FPSOOptimizer(pure_firefly).solve(moments, previous, np.random.default_rng(5))
    hybrid_result = FPSOOptimizer(hybrid).solve(moments, previous, np.random.default_rng(5))

    assert not np.allclose(firefly_result.weights, hybrid_result.weights)


def test_solve_time_is_measured(moments, previous, fast_params):
    """RQ1 asks about tractability, so wall-clock time is recorded, not guessed."""
    result = FPSOOptimizer(fast_params).solve(moments, previous, np.random.default_rng(0))
    assert result.solve_seconds > 0.0


@pytest.mark.parametrize("allocator_type", [EqualWeightAllocator, MinimumVarianceAllocator])
def test_baselines_respect_the_same_constraints(moments, previous, fast_params, allocator_type):
    """Baselines must be constrained identically to FPSO, or the comparison is unfair."""
    result = allocator_type(fast_params).solve(moments, previous, np.random.default_rng(0))
    assert_feasible(
        result.weights,
        ConstraintSet(len(moments.assets), fast_params.max_assets, fast_params.max_weight),
    )


def test_min_variance_has_lower_variance_than_equal_weight(moments, previous, fast_params):
    """A basic validity check on the minimum-variance baseline's optimization."""
    rng = np.random.default_rng(0)
    min_var = MinimumVarianceAllocator(fast_params).solve(moments, previous, rng)
    equal = EqualWeightAllocator(fast_params).solve(moments, previous, rng)

    def variance(weights):
        return float(weights @ moments.sigma @ weights)

    assert variance(min_var.weights) < variance(equal.weights)


def test_baselines_are_deterministic(moments, previous, fast_params):
    """Deterministic baselines justify running them at a single seed."""
    allocator = MinimumVarianceAllocator(fast_params)
    first = allocator.solve(moments, previous, np.random.default_rng(0)).weights
    second = allocator.solve(moments, previous, np.random.default_rng(99)).weights
    np.testing.assert_allclose(first, second)
