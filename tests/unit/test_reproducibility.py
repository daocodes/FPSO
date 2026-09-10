"""T2 — determinism and reproducibility.

The paper reports results averaged over 30 seeds. That claim is only meaningful
if (a) a seed reproduces exactly, and (b) different seeds actually differ. The
second half matters more than it looks: a dropped or ignored RNG argument makes
every "seed" identical, which would silently turn 30 runs into 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fpso.backtest.engine import RollingBacktestEngine, _SeedStreams
from fpso.backtest.results import build_manifest, source_hash
from fpso.data.moments import SampleMoments
from fpso.optimizer.fpso import FPSOOptimizer


@pytest.fixture
def moments(panel):
    """Moments over one estimation window of the synthetic panel."""
    window = panel.returns.loc["2009-01-01":"2009-12-31"]
    return SampleMoments(min_observations=100).estimate(window)


def test_same_seed_reproduces_the_optimizer_exactly(moments, fast_params):
    """Identical inputs and identical streams must give bit-identical weights."""
    optimizer = FPSOOptimizer(fast_params)
    previous = np.full(len(moments.assets), 1.0 / len(moments.assets))

    first = optimizer.solve(moments, previous, np.random.default_rng(11))
    second = optimizer.solve(moments, previous, np.random.default_rng(11))

    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.objective_value == second.objective_value


def test_different_seeds_give_different_solutions(moments, fast_params):
    """A silently-ignored RNG would make this fail, and nothing else would."""
    optimizer = FPSOOptimizer(fast_params)
    previous = np.full(len(moments.assets), 1.0 / len(moments.assets))

    solutions = [
        optimizer.solve(moments, previous, np.random.default_rng(seed)).weights
        for seed in range(5)
    ]
    distinct = {tuple(np.round(w, 10)) for w in solutions}
    assert len(distinct) > 1, "All seeds produced the same portfolio; RNG is not wired in."


def test_full_backtest_reproduces(base_config, panel):
    """Two engines, same config and seed, same result."""
    first = RollingBacktestEngine(base_config, panel).run(seed=3)
    second = RollingBacktestEngine(base_config, panel).run(seed=3)

    pd.testing.assert_frame_equal(first.target_weights, second.target_weights)
    pd.testing.assert_series_equal(
        first.returns(0.0), second.returns(0.0), check_names=False
    )


def test_backtest_seeds_diverge(base_config, panel):
    """Different seeds must produce different portfolios."""
    first = RollingBacktestEngine(base_config, panel).run(seed=0)
    second = RollingBacktestEngine(base_config, panel).run(seed=1)
    assert not first.target_weights.equals(second.target_weights)


def test_rebalance_streams_are_independent_of_position_in_the_schedule():
    """A date's random stream must not depend on how many dates preceded it.

    This is what lets the lookahead test compare two runs decision by decision:
    if streams were drawn from a running counter, inserting or skipping a
    rebalance would shift every subsequent draw and make the comparison
    impossible.
    """
    streams = _SeedStreams(arm="a", seed=0)
    date = pd.Timestamp("2011-06-30")

    first = streams.rebalance_rng(date).random(5)
    _ = streams.rebalance_rng(pd.Timestamp("2011-01-31")).random(100)
    second = streams.rebalance_rng(date).random(5)

    np.testing.assert_array_equal(first, second)


def test_arms_receive_distinct_streams():
    """Two arms with the same seed must not share random draws.

    Seeds are matched across arms for pairing purposes, but the *streams* must
    differ; otherwise two arms would make correlated random moves and the paired
    comparison would understate its own uncertainty.
    """
    date = pd.Timestamp("2011-06-30")
    a = _SeedStreams(arm="static", seed=0).rebalance_rng(date).random(5)
    b = _SeedStreams(arm="regime_hmm", seed=0).rebalance_rng(date).random(5)
    assert not np.allclose(a, b)


# ------------------------------------------------------- provenance record --

def test_source_hash_is_stable_and_content_addressed(tmp_path):
    """Two calls agree; a changed byte of source would not.

    The git SHA cannot carry this claim on its own: `git status --porcelain`
    reports the whole tree, so regenerating a figure marks a run `-dirty` with no
    source change. This study's first result matrix was stamped that way, which
    is why the manifest records a content hash of the package as well.
    """
    first = source_hash()
    assert first == source_hash()
    assert len(first) == 64 and set(first) <= set("0123456789abcdef")


def test_source_hash_tracks_content_not_git_state(monkeypatch):
    """The hash must depend on the files themselves, not on the repository."""
    import fpso

    root = Path(fpso.__file__).resolve().parent
    scratch = root / "_provenance_probe.py"
    baseline = source_hash()
    scratch.write_text("# transient file used by the provenance test\n")
    try:
        assert source_hash() != baseline, "adding a module must move the hash"
    finally:
        scratch.unlink()
    assert source_hash() == baseline, "removing it must restore the hash"


def test_manifest_carries_both_provenance_fields(base_config):
    manifest = build_manifest(base_config, arm="static", seed=0)
    assert manifest["source_hash"] == source_hash()
    assert "git_sha" in manifest
    # The pair is the point: the SHA locates a commit, the hash identifies the
    # bytes that ran. A reproducibility claim needs the second.
    assert manifest["source_hash"] != manifest["git_sha"]
