"""The turnover-matching solver.

The search itself is a bisection, which is only correct if turnover really is
monotone decreasing in `lambda_t`. That assumption is checked against the
optimizer rather than assumed, on a small synthetic problem; the rest of the
suite exercises the solver's bracketing and the config it emits.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fpso.data.moments import SampleMoments
from fpso.experiments.match_turnover import (
    solve_lambda_t,
    target_turnover,
    write_matched_config,
)
from fpso.optimizer.constraints import ConstraintSet, SimplexBoxCardinalityRepair
from fpso.optimizer.fpso import FPSOOptimizer


def one_way_turnover(new: np.ndarray, previous: np.ndarray) -> float:
    return float(0.5 * np.abs(new - previous).sum())


def test_turnover_decreases_with_the_penalty(panel, fast_params):
    """The monotonicity the bisection depends on, checked end to end.

    If a heavier `lambda_t` did not buy less trading, the solver would converge to
    an arbitrary point and the matched control would be meaningless.

    The starting portfolio must be *feasible* for this to be a fair test. Warm
    starting from 1/N over the full universe when the cardinality bound allows
    only a third of it forces the optimizer to liquidate the rest whatever
    `lambda_t` says, so turnover sits on a constraint-imposed floor and the
    penalty has no purchase. The engine always warm starts from the previous
    rebalance's holdings, which are feasible by construction, so that is what is
    reproduced here.
    """
    window = panel.returns.loc["2010-01-01":"2010-12-31"]
    moments = SampleMoments(min_observations=100).estimate(window)

    repair = SimplexBoxCardinalityRepair(
        ConstraintSet(len(moments.assets), fast_params.max_assets, fast_params.max_weight)
    )
    previous = repair.repair(np.ones(len(moments.assets)))

    turnovers = []
    for lambda_t in (0.0, 0.05, 0.25, 1.0):
        params = replace(fast_params, lambda_t=lambda_t, max_iter=25, num_particles=16)
        solution = FPSOOptimizer(params).solve(moments, previous, np.random.default_rng(0))
        turnovers.append(one_way_turnover(solution.weights, previous))

    assert turnovers[0] > turnovers[-1], (
        f"Turnover did not fall as lambda_t rose: {turnovers}"
    )


def test_solver_hits_a_reachable_target(base_config, panel):
    """Bisection must land within tolerance of a turnover the arm can reach."""
    config = replace(base_config, name="match_test")
    reachable = solve_lambda_t(
        config, panel, target=0.30, n_seeds=1, tolerance=0.05, max_iterations=5,
        verbose=False,
    )
    lambda_t, achieved, trace = reachable

    assert 0.0 <= lambda_t <= 0.5
    assert trace, "Solver returned no search trace."
    assert abs(achieved - 0.30) <= 0.12, (
        f"Solver settled at turnover {achieved:.3f}, far from the 0.30 target."
    )


def test_unreachable_target_fails_loudly(base_config, panel):
    """A target below what the heaviest penalty can reach must raise, not silently
    return the bracket end and pretend it matched."""
    config = replace(base_config, name="match_impossible")
    with pytest.raises(ValueError, match="Widen DEFAULT_BRACKET"):
        solve_lambda_t(
            config, panel, target=1e-6, n_seeds=1, max_iterations=2, verbose=False
        )


def test_emitted_config_loads_and_carries_provenance(tmp_path):
    """The matched arm must be a valid config and must say where it came from."""
    from fpso.config.loader import load_experiment_config

    output = tmp_path / "static_matched_regime_hmm.yaml"
    # `extends` is resolved relative to the config file, so the emitted arm needs
    # base.yaml beside it the way configs/arms/ has ../base.yaml.
    (tmp_path.parent / "base.yaml").write_text(
        Path("configs/base.yaml").read_text()
    )
    write_matched_config(0.0234, 0.3149, 0.3145, "regime_hmm", output)

    text = output.read_text()
    assert "TURNOVER-MATCHED CONTROL" in text
    assert "regime_hmm" in text
    assert "0.3145" in text, "Target turnover missing from the provenance header."
    assert "CONTROL, not a strategy" in text

    config = load_experiment_config(output)
    assert config.base_params.lambda_t == pytest.approx(0.0234)
    assert config.regime.detector == "none"
    assert config.regime.overrides == {}


def test_target_turnover_requires_a_completed_arm(tmp_path):
    with pytest.raises(FileNotFoundError, match="No results for arm"):
        target_turnover("does_not_exist", tmp_path)
