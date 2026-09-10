"""Tests for exact weighting and learned shrinkage.

The load-bearing test here is `test_learned_policy_ignores_the_future`. A learned
component inside a backtest is exactly where lookahead creeps in, and the whole
result is worthless if a decision at date t depends on anything after it. That
test poisons future outcomes and asserts earlier decisions are bit-identical,
mirroring the harness's lookahead proof.
"""

from __future__ import annotations

import numpy as np
import pytest

from fpso.adaptive.features import FEATURE_NAMES, market_features, swarm_features
from fpso.adaptive.shrinkage import (
    RidgeShrinkagePolicy,
    ShrinkageGrid,
    oracle_shrinkage,
    shuffled_shrinkage,
    walk_forward_best_constant,
    walk_forward_learned,
)
from fpso.data.moments import Moments
from fpso.optimizer.exact import solve_weights_on_support
from fpso.optimizer.fpso_cw import shrink_mu


@pytest.fixture
def moments() -> Moments:
    rng = np.random.default_rng(0)
    n = 12
    factor = rng.normal(size=(n, 3))
    sigma = factor @ factor.T + np.eye(n) * 0.5
    return Moments(assets=np.arange(n), mu=rng.normal(0.08, 0.05, n), sigma=sigma)


def _grid(n_dates: int = 80, n_deltas: int = 4, seed: int = 0) -> ShrinkageGrid:
    rng = np.random.default_rng(seed)
    deltas = np.linspace(0.0, 1.0, n_deltas)
    features = rng.normal(size=(n_dates, len(FEATURE_NAMES)))
    # Outcome depends on the first feature, so there is a real map to learn.
    ideal = np.clip(0.5 + 0.4 * features[:, 0], 0, 1)
    outcomes = -((deltas[None, :] - ideal[:, None]) ** 2) + rng.normal(0, 0.01, (n_dates, n_deltas))
    return ShrinkageGrid(deltas=deltas, features=features, outcomes=outcomes)


# --------------------------------------------------------------------------
# Exact weighting


def test_exact_solution_is_feasible(moments: Moments) -> None:
    support = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    w, _ = solve_weights_on_support(
        support, moments.mu, moments.sigma, np.zeros(12),
        lambda_v=1.0, lambda_t=0.01, max_weight=0.15,
    )
    assert w.shape == (12,)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-12).all() and (w <= 0.15 + 1e-9).all()
    off_support = np.setdiff1d(np.arange(12), support)
    assert np.allclose(w[off_support], 0.0)


def test_exact_solution_beats_equal_weight_on_the_same_support(moments: Moments) -> None:
    """The premise of FPSO-CW: on a fixed support the weighting is solvable, so
    an exact solve cannot be beaten by an arbitrary feasible allocation."""
    support = np.arange(8)
    prev = np.zeros(12)
    w, value = solve_weights_on_support(
        support, moments.mu, moments.sigma, prev,
        lambda_v=1.0, lambda_t=0.01, max_weight=0.15,
    )

    def objective(x: np.ndarray) -> float:
        return float(
            moments.mu @ x
            - np.sqrt(x @ moments.sigma @ x)
            - 0.01 * 0.5 * np.abs(x - prev).sum()
        )

    equal = np.zeros(12)
    equal[support] = 1.0 / len(support)
    assert value >= objective(equal) - 1e-9
    assert np.isclose(value, objective(w), atol=1e-6)


def test_infeasible_support_is_rejected(moments: Moments) -> None:
    """5 names capped at 15% cannot reach a unit budget. Silently returning an
    infeasible portfolio would corrupt every downstream number."""
    with pytest.raises(ValueError, match="unit budget"):
        solve_weights_on_support(
            np.arange(5), moments.mu, moments.sigma, np.zeros(12),
            lambda_v=1.0, lambda_t=0.01, max_weight=0.15,
        )


def test_shrinkage_interpolates_to_a_flat_forecast() -> None:
    mu = np.array([0.10, 0.02, 0.06])
    assert np.allclose(shrink_mu(mu, 0.0), mu)
    assert np.allclose(shrink_mu(mu, 1.0), mu.mean())
    assert np.allclose(shrink_mu(mu, 0.5), 0.5 * mu + 0.5 * mu.mean())


# --------------------------------------------------------------------------
# Features


def test_swarm_features_detect_unanimity_and_disagreement() -> None:
    identical = np.tile(np.array([0.5, 0.5, 0.0, 0.0]), (6, 1))
    agreed = swarm_features(identical, np.ones(6))
    assert agreed["support_disagreement"] == pytest.approx(0.0)
    assert agreed["weight_dispersion"] == pytest.approx(0.0)
    assert agreed["distinct_supports"] == pytest.approx(1 / 6)

    disjoint = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
    split = swarm_features(disjoint, np.array([1.0, 0.5]))
    assert split["support_disagreement"] == pytest.approx(1.0)
    assert split["weight_dispersion"] == pytest.approx(1.0)
    assert split["distinct_supports"] == pytest.approx(1.0)


def test_market_features_are_finite_and_named(moments: Moments) -> None:
    feats = market_features(moments)
    assert set(feats) == {
        "mu_dispersion", "mean_volatility", "avg_correlation", "condition_number"
    }
    assert all(np.isfinite(v) for v in feats.values())


def test_single_particle_swarm_reports_no_disagreement() -> None:
    lone = swarm_features(np.array([[1.0, 0.0]]), np.array([0.3]))
    assert all(v == 0.0 for v in lone.values())


# --------------------------------------------------------------------------
# The learned policy


def test_learned_policy_ignores_the_future() -> None:
    """Poison every outcome from date k onward; decisions before k must not move.

    This is the property that makes the component admissible inside a causal
    backtest. If it ever fails, any performance the policy shows is lookahead.
    """
    grid = _grid()
    k = 55
    chosen = walk_forward_learned(grid, min_train=20)

    poisoned_outcomes = grid.outcomes.copy()
    poisoned_outcomes[k:] = np.random.default_rng(99).normal(size=poisoned_outcomes[k:].shape)
    poisoned = ShrinkageGrid(
        deltas=grid.deltas, features=grid.features, outcomes=poisoned_outcomes
    )
    after = walk_forward_learned(poisoned, min_train=20)

    np.testing.assert_array_equal(chosen[:k], after[:k])
    assert not np.array_equal(chosen[k:], after[k:]), (
        "Poisoning the future changed nothing at all — the policy is not reading "
        "outcomes, so this test would pass vacuously."
    )


def test_learned_policy_recovers_a_planted_signal() -> None:
    grid = _grid(n_dates=200, seed=3)
    learned = walk_forward_learned(grid, min_train=30)
    ideal = np.clip(0.5 + 0.4 * grid.features[:, 0], 0, 1)
    tail = slice(30, None)
    assert np.corrcoef(learned[tail], ideal[tail])[0, 1] > 0.5


def test_policies_stay_inside_the_priced_grid() -> None:
    grid = _grid()
    for chosen in (
        walk_forward_learned(grid, min_train=20),
        walk_forward_best_constant(grid, min_train=20),
        oracle_shrinkage(grid),
    ):
        assert np.isin(chosen, grid.deltas).all()


def test_burn_in_uses_the_fallback() -> None:
    grid = _grid()
    chosen = walk_forward_learned(grid, min_train=25, fallback=0.0)
    assert np.all(chosen[:25] == 0.0)


def test_oracle_is_an_upper_bound_on_the_learned_policy() -> None:
    grid = _grid(n_dates=150, seed=7)
    index = {d: i for i, d in enumerate(grid.deltas)}

    def realised(chosen: np.ndarray) -> float:
        return float(sum(grid.outcomes[t, index[d]] for t, d in enumerate(chosen)))

    assert realised(oracle_shrinkage(grid)) >= realised(walk_forward_learned(grid, min_train=30))


def test_shuffled_control_preserves_the_marginal_distribution() -> None:
    deltas = np.array([0.0, 0.5, 0.5, 1.0, 0.0, 0.5])
    shuffled = shuffled_shrinkage(deltas, np.random.default_rng(1))
    np.testing.assert_array_equal(np.sort(shuffled), np.sort(deltas))


def test_ridge_standardises_on_training_rows_only() -> None:
    """Standardising on the full sample would leak the future through the scale."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, len(FEATURE_NAMES)))
    y = rng.uniform(0, 1, 40)
    policy = RidgeShrinkagePolicy().fit(X[:20], y[:20])
    baseline = policy.predict(X[25])

    shifted = X.copy()
    shifted[20:] += 100.0
    again = RidgeShrinkagePolicy().fit(shifted[:20], y[:20]).predict(X[25])
    assert baseline == pytest.approx(again)


def test_constant_feature_does_not_break_the_fit() -> None:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(30, len(FEATURE_NAMES)))
    X[:, 2] = 4.0
    policy = RidgeShrinkagePolicy().fit(X, rng.uniform(0, 1, 30))
    assert np.isfinite(policy.predict(X[0]))
    assert len(policy.coefficients) == len(FEATURE_NAMES)


def test_grid_shape_mismatch_is_caught() -> None:
    with pytest.raises(ValueError, match="outcome rows"):
        ShrinkageGrid(
            deltas=np.array([0.0, 1.0]),
            features=np.zeros((5, len(FEATURE_NAMES))),
            outcomes=np.zeros((4, 2)),
        )
    with pytest.raises(ValueError, match="outcome columns"):
        ShrinkageGrid(
            deltas=np.array([0.0, 0.5, 1.0]),
            features=np.zeros((5, len(FEATURE_NAMES))),
            outcomes=np.zeros((5, 2)),
        )
