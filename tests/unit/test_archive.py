"""The regime-keyed solution archive and the convergence metric behind the speed claim.

The archive's whole purpose is to change *where the search starts* without
changing what it is searching for. These tests pin that separation down: the
optimizer must accept seeds without knowing where they came from, the archive
must survive universe drift, and an unseen state must degrade to the ordinary
random start rather than failing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpso.optimizer.base import iterations_to_converge
from fpso.regime.archive import RegimeSolutionArchive


def portfolio(names, values) -> pd.Series:
    return pd.Series(values, index=pd.Index(names), dtype=float)


# ------------------------------------------------------------------ archive --

def test_keeps_only_the_best_solutions_per_state():
    archive = RegimeSolutionArchive(capacity=2)
    for score in (0.1, 0.9, 0.5, 0.3):
        archive.record("CALM", portfolio(["a", "b"], [0.5, 0.5]), score)

    seeds = archive.seeds_for("CALM", pd.Index(["a", "b"]), limit=5)
    assert len(seeds) == 2, "Capacity must bound how many solutions are retained."
    assert archive.size("CALM") == 2


def test_states_are_kept_separate():
    archive = RegimeSolutionArchive()
    archive.record("CALM", portfolio(["a", "b"], [0.9, 0.1]), 1.0)
    archive.record("CRISIS", portfolio(["a", "b"], [0.1, 0.9]), 1.0)

    calm = archive.seeds_for("CALM", pd.Index(["a", "b"]), limit=1)[0]
    crisis = archive.seeds_for("CRISIS", pd.Index(["a", "b"]), limit=1)[0]
    assert calm[0] > calm[1] and crisis[1] > crisis[0], (
        "A state must return its own solutions, not another state's."
    )


def test_unseen_state_degrades_to_no_seeds():
    """The mechanism must fall back to a normal random start, not raise.

    Every run begins with an empty archive, and a state first seen in year nine
    has nothing stored. That has to be ordinary behaviour.
    """
    archive = RegimeSolutionArchive()
    assert archive.seeds_for("CRISIS", pd.Index(["a"]), limit=3) == []


def test_survives_universe_drift():
    """Stored by asset name, so reconstitution cannot scramble a seed.

    If entries were stored positionally, a portfolio archived when the universe
    held one set of names would be silently reinterpreted as weights on whatever
    occupied those positions later — an allocation nobody chose.
    """
    archive = RegimeSolutionArchive()
    archive.record("CALM", portfolio(["AAPL", "MSFT", "XOM"], [0.5, 0.3, 0.2]), 1.0)

    # XOM leaves the universe, NVDA joins, and the ordering changes.
    seeds = archive.seeds_for("CALM", pd.Index(["NVDA", "MSFT", "AAPL"]), limit=1)
    assert seeds[0] == pytest.approx([0.0, 0.3, 0.5]), (
        "Weights must follow their assets, not their former positions."
    )


def test_non_finite_objective_is_not_archived():
    archive = RegimeSolutionArchive()
    archive.record("CALM", portfolio(["a"], [1.0]), float("nan"))
    archive.record("CALM", portfolio(["a"], [1.0]), float("-inf"))
    assert archive.size("CALM") == 0


# -------------------------------------------------------- convergence metric --

def test_a_better_start_converges_in_fewer_iterations():
    early = np.array([0.0, 0.95, 0.99, 1.0, 1.0, 1.0])
    late = np.array([0.0, 0.1, 0.3, 0.6, 0.9, 1.0])
    assert iterations_to_converge(early) < iterations_to_converge(late)


def test_handles_negative_objectives():
    """Fitness nets risk and turnover off return and is routinely negative.

    A ratio-based threshold would invert whenever the sign does, so the metric
    works on the run's improvement range instead.
    """
    trace = np.array([-5.0, -3.0, -1.1, -1.0])
    assert 1 <= iterations_to_converge(trace) <= len(trace)


def test_already_converged_start_costs_one_iteration():
    """The mechanism's best case: seeded at the plateau, nothing left to find."""
    assert iterations_to_converge(np.array([1.0, 1.0, 1.0])) == 1


def test_closed_form_allocators_report_no_iterations():
    assert iterations_to_converge(np.array([])) == -1


# ------------------------------------------------- separation of the mechanism --

def test_optimizer_accepts_seeds_without_knowing_their_origin(panel, fast_params):
    """FPSO must take seeds as plain vectors — that is what keeps regime logic out.

    If the optimizer had to be told the regime, the mechanism would no longer be
    separable and would carry the same coupling that sank the preference-based
    injection point.
    """
    from dataclasses import replace

    from fpso.data.moments import SampleMoments
    from fpso.optimizer.fpso import FPSOOptimizer

    window = panel.returns.loc["2010-01-01":"2010-12-31"]
    moments = SampleMoments(min_observations=100).estimate(window)
    n = len(moments.assets)
    params = replace(fast_params, max_iter=10, num_particles=8)
    previous = np.full(n, 1.0 / n)

    seeded = FPSOOptimizer(params).solve(
        moments, previous, np.random.default_rng(0), seeds=[np.full(n, 1.0 / n)]
    )
    unseeded = FPSOOptimizer(params).solve(moments, previous, np.random.default_rng(0))

    assert np.isfinite(seeded.objective_value)
    assert len(seeded.weights) == len(unseeded.weights)


def test_more_seeds_never_exceed_the_swarm(panel, fast_params):
    """Passing more seeds than particles must not grow the population."""
    from dataclasses import replace

    from fpso.data.moments import SampleMoments
    from fpso.optimizer.constraints import ConstraintSet, SimplexBoxCardinalityRepair
    from fpso.optimizer.fpso import Swarm

    window = panel.returns.loc["2010-01-01":"2010-12-31"]
    moments = SampleMoments(min_observations=100).estimate(window)
    n = len(moments.assets)
    params = replace(fast_params, num_particles=4)
    repair = SimplexBoxCardinalityRepair(
        ConstraintSet(n, params.max_assets, params.max_weight)
    )

    swarm = Swarm.initialize(
        n_particles=4, n_assets=n, repair=repair,
        weights_prev=np.full(n, 1.0 / n), rng=np.random.default_rng(0),
        seeds=[np.full(n, 1.0 / n)] * 20,
    )
    assert len(swarm.particles) == 4
