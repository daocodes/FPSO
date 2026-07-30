"""T3 — constraint invariants.

Repair is what actually enforces feasibility in this codebase (the penalty terms
never bind along the search path), so its guarantees are load-bearing for every
number in the paper. These tests hit it with adversarial inputs: all-zeros,
negatives, NaN, infinities, and single-asset universes.
"""

from __future__ import annotations

import numpy as np
import pytest

from fpso.optimizer.constraints import ConstraintSet, SimplexBoxCardinalityRepair

TOLERANCE = 1e-12


def assert_feasible(weights: np.ndarray, constraints: ConstraintSet) -> None:
    """The three hard constraints, asserted together."""
    assert np.isfinite(weights).all(), "Repair produced non-finite weights."
    assert weights.min() >= -TOLERANCE, "Long-only lower bound violated."
    assert weights.max() <= constraints.max_weight + 1e-9, "Box upper bound violated."
    assert abs(weights.sum() - 1.0) < 1e-9, "Budget constraint violated."
    active = int((weights > 1e-8).sum())
    assert active <= constraints.effective_cardinality, (
        f"Cardinality violated: {active} active > K={constraints.effective_cardinality}."
    )


@pytest.fixture
def constraints() -> ConstraintSet:
    return ConstraintSet(n_assets=40, max_assets=10, max_weight=0.2)


@pytest.fixture
def repair(constraints) -> SimplexBoxCardinalityRepair:
    return SimplexBoxCardinalityRepair(constraints)


@pytest.mark.parametrize("trial", range(50))
def test_random_inputs_are_repaired_to_feasible(repair, constraints, trial):
    """Property test: any real vector maps into the feasible set."""
    generator = np.random.default_rng(trial)
    raw = generator.normal(0.0, 3.0, constraints.n_assets)
    assert_feasible(repair.repair(raw), constraints)


@pytest.mark.parametrize(
    "name,raw",
    [
        ("all zeros", np.zeros(40)),
        ("all negative", -np.ones(40)),
        ("all nan", np.full(40, np.nan)),
        ("infinities", np.full(40, np.inf)),
        ("mixed non-finite", np.array([np.nan, np.inf, -np.inf] + [0.1] * 37)),
        ("single spike", np.eye(40)[0]),
        ("tiny values", np.full(40, 1e-300)),
    ],
)
def test_adversarial_inputs(repair, constraints, name, raw):
    """Degenerate inputs must still yield a feasible portfolio, not a crash."""
    assert_feasible(repair.repair(raw), constraints)


def test_repair_is_idempotent(repair, constraints):
    """repair(repair(w)) == repair(w) — required for a stable search trajectory."""
    generator = np.random.default_rng(7)
    for _ in range(20):
        once = repair.repair(generator.normal(size=constraints.n_assets))
        np.testing.assert_allclose(repair.repair(once), once, atol=1e-12)


def test_cardinality_keeps_the_largest_weights(repair):
    """The surviving names are the top-K by raw weight, so repair preserves intent."""
    raw = np.arange(40, dtype=float)
    weights = repair.repair(raw)
    assert set(np.nonzero(weights)[0]) == set(range(30, 40))


def test_ties_are_broken_deterministically(repair, constraints):
    """Identical inputs must select identical names across repeated calls.

    Without an explicit tie-break, `argpartition` could return different index
    sets for the same input on different NumPy builds, which would silently break
    the reproducibility guarantee the paper makes.
    """
    raw = np.ones(constraints.n_assets)
    first = repair.repair(raw)
    for _ in range(5):
        np.testing.assert_array_equal(repair.repair(raw), first)


def test_cardinality_larger_than_universe_is_clipped():
    """K > n must degrade to 'hold everything', not raise."""
    constraints = ConstraintSet(n_assets=5, max_assets=50, max_weight=0.5)
    weights = SimplexBoxCardinalityRepair(constraints).repair(np.ones(5))
    assert_feasible(weights, constraints)
    assert (weights > 0).sum() == 5


def test_infeasible_constraint_set_is_rejected_at_construction():
    """K * u < 1 means no fully-invested portfolio exists; fail loudly and early."""
    constraints = ConstraintSet(n_assets=100, max_assets=5, max_weight=0.1)
    assert not constraints.is_feasible
    with pytest.raises(ValueError, match="Infeasible constraint set"):
        SimplexBoxCardinalityRepair(constraints)


def test_wrong_length_input_is_rejected(repair):
    with pytest.raises(ValueError, match="Expected 40 weights"):
        repair.repair(np.ones(7))


def test_binding_box_constraint_caps_every_name():
    """When K * u == 1 exactly, the only feasible portfolio is K names at the cap."""
    constraints = ConstraintSet(n_assets=20, max_assets=5, max_weight=0.2)
    repair = SimplexBoxCardinalityRepair(constraints)
    weights = repair.repair(np.random.default_rng(0).random(20))
    assert_feasible(weights, constraints)
    np.testing.assert_allclose(np.sort(weights)[-5:], np.full(5, 0.2), atol=1e-9)
