"""Inference machinery: bootstrap coverage, the Sharpe test, and Holm.

The statistics decide what the paper is allowed to claim, so they are tested
against cases with known answers rather than only for "it runs".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpso.evaluation.metrics import summarize
from fpso.evaluation.statistics import (
    StationaryBootstrap,
    annualized_sharpe,
    holm_adjust,
    ledoit_wolf_sharpe_test,
    max_drawdown_statistic,
)


def make_path(mean: float, vol: float, n: int = 1500, seed: int = 0) -> pd.Series:
    """A daily return path with a known annualized drift and volatility."""
    generator = np.random.default_rng(seed)
    dates = pd.bdate_range("2011-01-03", periods=n)
    daily = generator.normal(mean / 252.0, vol / np.sqrt(252.0), n)
    return pd.Series(daily, index=dates)


# ------------------------------------------------------------------- metrics --

def test_metrics_recover_known_moments():
    path = make_path(mean=0.10, vol=0.16, n=5000)
    metrics = summarize(path)
    assert metrics.annual_volatility == pytest.approx(0.16, rel=0.05)
    assert metrics.sharpe == pytest.approx(0.10 / 0.16, rel=0.25)
    assert metrics.max_drawdown < 0


def test_cagr_matches_the_equity_curve():
    """Annual return must be consistent with terminal wealth, not the mean return."""
    path = make_path(mean=0.08, vol=0.12, n=2520)  # ten years
    metrics = summarize(path)
    terminal = float((1.0 + path).prod())
    assert (1.0 + metrics.annual_return) ** 10 == pytest.approx(terminal, rel=1e-6)


# ----------------------------------------------------------------- bootstrap --

def test_bootstrap_interval_contains_a_zero_difference():
    """Two paths from the same process must not produce a significant difference."""
    a, b = make_path(0.08, 0.15, seed=1), make_path(0.08, 0.15, seed=2)
    interval = StationaryBootstrap(n_resamples=500).paired_sharpe_difference(a, b)
    assert interval.lower < 0 < interval.upper
    assert interval.p_value > 0.05


def test_bootstrap_detects_a_large_difference():
    """A path with far higher Sharpe must be flagged."""
    a, b = make_path(0.30, 0.10, seed=1), make_path(0.00, 0.20, seed=2)
    interval = StationaryBootstrap(n_resamples=500).paired_sharpe_difference(a, b)
    assert interval.statistic > 0
    assert interval.lower > 0
    assert interval.p_value < 0.05


def test_bootstrap_is_deterministic_given_its_seed():
    a, b = make_path(0.10, 0.15, seed=1), make_path(0.05, 0.15, seed=2)
    first = StationaryBootstrap(n_resamples=200, seed=7).paired_sharpe_difference(a, b)
    second = StationaryBootstrap(n_resamples=200, seed=7).paired_sharpe_difference(a, b)
    assert first == second


def test_bootstrap_works_for_arbitrary_statistics():
    a, b = make_path(0.10, 0.12, seed=1), make_path(0.10, 0.30, seed=2)
    interval = StationaryBootstrap(n_resamples=300).paired_statistic(
        a, b, max_drawdown_statistic
    )
    # The lower-volatility path should have the shallower (less negative) drawdown.
    assert interval.statistic > 0


def test_block_resampling_preserves_sample_length():
    bootstrap = StationaryBootstrap(n_resamples=1, expected_block=21)
    index = bootstrap._resample_index(500)
    assert len(index) == 500
    assert index.min() >= 0 and index.max() < 500


# ----------------------------------------------------------- ledoit and wolf --

def test_ledoit_wolf_agrees_with_the_observed_sharpe_difference():
    """The test statistic must be the Sharpe difference the tables report.

    The tolerance is 1e-3 rather than exact because the Ledoit-Wolf estimator is
    method-of-moments (population volatility) while `annualized_sharpe` uses the
    ddof=1 sample standard deviation. The two differ by sqrt(n/(n-1)), which is
    about 0.03% at this sample size — see the note in `ledoit_wolf_sharpe_test`.
    """
    a, b = make_path(0.20, 0.15, seed=1), make_path(0.05, 0.15, seed=2)
    difference, p_value = ledoit_wolf_sharpe_test(a, b)
    observed = annualized_sharpe(a.to_numpy()) - annualized_sharpe(b.to_numpy())
    assert difference == pytest.approx(observed, rel=1e-3)
    assert 0.0 <= p_value <= 1.0


def test_ledoit_wolf_does_not_reject_identical_processes():
    a, b = make_path(0.08, 0.15, seed=11), make_path(0.08, 0.15, seed=12)
    _, p_value = ledoit_wolf_sharpe_test(a, b)
    assert p_value > 0.05


def test_ledoit_wolf_requires_enough_observations():
    short = make_path(0.1, 0.1, n=20)
    with pytest.raises(ValueError, match="at least 30"):
        ledoit_wolf_sharpe_test(short, short)


# ---------------------------------------------------------------------- holm --

def test_holm_is_monotone_and_conservative():
    raw = {"a": 0.001, "b": 0.01, "c": 0.04, "d": 0.30}
    adjusted = holm_adjust(raw)

    assert all(adjusted[k] >= raw[k] for k in raw)
    ordered = [adjusted[k] for k in sorted(raw, key=raw.get)]
    assert ordered == sorted(ordered), "Holm-adjusted values must be non-decreasing."
    assert adjusted["a"] == pytest.approx(0.004)


def test_holm_never_exceeds_one():
    adjusted = holm_adjust({"a": 0.6, "b": 0.7, "c": 0.9})
    assert all(value <= 1.0 for value in adjusted.values())


def test_holm_handles_an_empty_family():
    assert holm_adjust({}) == {}
