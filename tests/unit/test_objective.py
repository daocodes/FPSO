"""Objective correctness against closed forms, and the repair/penalty interaction."""

from __future__ import annotations

import numpy as np
import pytest

from fpso.config.schema import FPSOParams
from fpso.optimizer.constraints import ConstraintSet, SimplexBoxCardinalityRepair
from fpso.optimizer.objective import MeanVarianceTurnover, PenalizedObjective


@pytest.fixture
def objective_inputs():
    generator = np.random.default_rng(0)
    factor = generator.normal(size=(200, 5))
    sigma = np.cov(factor.T) + np.eye(5) * 0.01
    return np.array([0.10, 0.08, 0.12, 0.05, 0.09]), sigma


def test_objective_matches_the_closed_form(objective_inputs):
    """R(w) must equal mu'w - lambda_v*sqrt(w'Sigma w) - lambda_t*||w - w_prev||_1."""
    mu, sigma = objective_inputs
    w = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
    w_prev = np.full(5, 0.2)
    lambda_v, lambda_t = 1.5, 0.02

    expected = (
        mu @ w
        - lambda_v * np.sqrt(w @ sigma @ w)
        - lambda_t * np.abs(w - w_prev).sum()
    )
    actual = MeanVarianceTurnover(mu, sigma, w_prev, lambda_v, lambda_t).score(w)
    assert actual == pytest.approx(expected)


def test_zero_turnover_when_holding_the_same_portfolio(objective_inputs):
    mu, sigma = objective_inputs
    w = np.full(5, 0.2)
    with_penalty = MeanVarianceTurnover(mu, sigma, w, 1.0, 100.0).score(w)
    without_penalty = MeanVarianceTurnover(mu, sigma, w, 1.0, 0.0).score(w)
    assert with_penalty == pytest.approx(without_penalty)


def test_penalties_never_bind_on_repaired_portfolios(objective_inputs):
    """B8 in the plan: repair dominates, so Phi(w) == R(w) along the search path.

    This is a factual claim the paper makes about its own objective, so it is
    asserted rather than assumed. If a future change lets infeasible candidates
    reach the objective, this test is what will catch it.
    """
    mu, sigma = objective_inputs
    params = FPSOParams(max_assets=3, max_weight=0.5)
    constraints = ConstraintSet(n_assets=5, max_assets=3, max_weight=0.5)
    repair = SimplexBoxCardinalityRepair(constraints)

    base = MeanVarianceTurnover(mu, sigma, np.full(5, 0.2), 1.0, 0.01)
    penalized = PenalizedObjective(base, constraints, params)

    generator = np.random.default_rng(1)
    for _ in range(100):
        w = repair.repair(generator.normal(size=5))
        assert penalized.violation(w).is_feasible
        assert penalized.score(w) == pytest.approx(base.score(w))


def test_penalties_do_bind_on_infeasible_portfolios(objective_inputs):
    """The safety net still works when repair is bypassed (penalty-only ablation)."""
    mu, sigma = objective_inputs
    params = FPSOParams(max_assets=2, max_weight=0.3)
    constraints = ConstraintSet(n_assets=5, max_assets=2, max_weight=0.3)
    base = MeanVarianceTurnover(mu, sigma, np.full(5, 0.2), 1.0, 0.01)
    penalized = PenalizedObjective(base, constraints, params)

    infeasible = np.full(5, 0.4)  # sums to 2.0, all above the cap, 5 active names
    violation = penalized.violation(infeasible)

    assert not violation.is_feasible
    assert violation.budget == pytest.approx(1.0)
    assert violation.box > 0
    assert violation.cardinality > 0
    assert penalized.score(infeasible) < base.score(infeasible)


def test_higher_risk_aversion_prefers_the_lower_variance_portfolio(objective_inputs):
    """A sanity check that lambda_v points the way the paper says it does."""
    mu, sigma = objective_inputs
    concentrated = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    diversified = np.full(5, 0.2)
    w_prev = np.full(5, 0.2)

    assert np.sqrt(diversified @ sigma @ diversified) < np.sqrt(
        concentrated @ sigma @ concentrated
    )
    high_aversion = MeanVarianceTurnover(mu, sigma, w_prev, 10.0, 0.0)
    assert high_aversion.score(diversified) > high_aversion.score(concentrated)
