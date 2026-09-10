"""Inclusion frequency as an intrinsic explanation of optimizer conviction.

The measure's value depends entirely on it being computed from the decisions
themselves, so the tests pin the arithmetic rather than the interpretation: a
unanimous holding must read 1.0, a name half the runs chose must read 0.5, and
the two failure modes that would quietly invalidate it — a single run, and
misaligned rebalance dates — must raise instead of returning something plausible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpso.evaluation.conviction import (
    ConvictionProfile,
    conviction_profile,
    inclusion_frequency,
)

DATES = pd.date_range("2020-01-31", periods=3, freq="ME")


def _weights(rows: list[list[float]], columns: list[int]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=DATES, columns=columns)


@pytest.fixture
def runs() -> dict[int, pd.DataFrame]:
    """Three runs: asset 1 unanimous, asset 2 in two of three, asset 3 in one."""
    return {
        0: _weights([[0.5, 0.5, 0.0]] * 3, [1, 2, 3]),
        1: _weights([[0.5, 0.5, 0.0]] * 3, [1, 2, 3]),
        2: _weights([[0.5, 0.0, 0.5]] * 3, [1, 2, 3]),
    }


def test_frequency_counts_runs_not_weight(runs):
    """A large position in one run must not outvote small positions in many."""
    frequency = inclusion_frequency(runs)
    assert frequency.loc[DATES[0], 1] == pytest.approx(1.0)
    assert frequency.loc[DATES[0], 2] == pytest.approx(2 / 3)
    assert frequency.loc[DATES[0], 3] == pytest.approx(1 / 3)


def test_runs_need_not_share_columns(runs):
    """A name absent from a run is not held in it, which is a zero, not an error."""
    runs[2] = runs[2].drop(columns=[3])
    frequency = inclusion_frequency(runs)
    assert frequency.loc[DATES[0], 3] == pytest.approx(0.0)
    assert set(frequency.columns) == {1, 2, 3}


def test_a_single_run_is_rejected(runs):
    """One run reports every holding as unanimous — the most misleading output
    this function could produce, so it must refuse rather than return it."""
    with pytest.raises(ValueError, match="at least two runs"):
        inclusion_frequency({0: runs[0]})


def test_misaligned_dates_are_rejected(runs):
    runs[1] = runs[1].set_axis(DATES + pd.Timedelta(days=1), axis=0)
    with pytest.raises(ValueError, match="rebalance dates"):
        inclusion_frequency(runs)


def test_profile_counts_only_names_some_run_held():
    """The point-in-time universe changes, so never-held names must not inflate
    the denominator and make conviction look worse in later years by accident."""
    frequency = pd.DataFrame(
        [[1.0, 0.5, 0.0, 0.0]], index=DATES[:1], columns=[1, 2, 3, 4]
    )
    profile = conviction_profile(frequency)
    assert profile.table.loc[DATES[0], "candidates"] == 2
    assert profile.table.loc[DATES[0], "unanimous"] == 1
    assert profile.table.loc[DATES[0], "contested"] == 1


def test_profile_reports_defensible_and_contested_counts(runs):
    profile = conviction_profile(inclusion_frequency(runs))
    assert isinstance(profile, ConvictionProfile)
    # Asset 1 only: 2/3 and 1/3 both fall below the 0.9 threshold.
    assert profile.mean_strong == pytest.approx(1.0)
    assert profile.mean_contested == pytest.approx(2.0)
    assert profile.share_without_unanimity == pytest.approx(0.0)
    assert "defensible holdings" in profile.summary()


def test_unanimity_share_detects_total_disagreement():
    """The headline statistic: rebalances at which nothing was agreed on."""
    frequency = pd.DataFrame(
        [[0.5, 0.5], [1.0, 0.0]], index=DATES[:2], columns=[1, 2]
    )
    profile = conviction_profile(frequency)
    assert profile.share_without_unanimity == pytest.approx(0.5)


def test_frequency_is_bounded_and_finite(runs):
    frequency = inclusion_frequency(runs)
    values = frequency.to_numpy()
    assert np.all(np.isfinite(values))
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_skipped_rebalances_do_not_shift_later_dates():
    """A rebalance nobody traded is dropped with its date, not silently relabelled.

    Indexing the surviving rows by a positional slice of the original index would
    attach every later row to the wrong date — a corruption that shows up as a
    plausible time series rather than as an error.
    """
    frequency = pd.DataFrame(
        [[0.0, 0.0], [1.0, 0.5], [0.5, 0.5]], index=DATES, columns=[1, 2]
    )
    profile = conviction_profile(frequency)
    assert list(profile.table.index) == [DATES[1], DATES[2]]
    assert profile.table.loc[DATES[1], "unanimous"] == 1
    assert profile.table.loc[DATES[2], "unanimous"] == 0
