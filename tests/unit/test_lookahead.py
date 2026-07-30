"""T1 — the lookahead test. The flagship correctness property of this codebase.

The argument: run the full backtest on the real panel, then build a corrupted
panel identical up to date *T* and garbage after it, and run again. If any
decision at or before *T* differs between the two runs, some component read data
it was not entitled to see.

This is a machine-checked proof of causality for the *entire* pipeline —
features, expanding standardization, HMM fit and inference, smoothing, universe
selection, moment estimation and the optimizer — rather than an assertion about
any one of them. It is parameterized over several cut dates so that a leak
confined to one part of the sample cannot hide.

The oracle arm is deliberately excluded: it fits on the full sample by design and
is reported in the paper as a lookahead upper bound, never as a result.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fpso.backtest.engine import RollingBacktestEngine
from fpso.data.panel import ReturnPanel

CUT_DATES = ["2010-06-30", "2011-03-31", "2011-12-30"]


def corrupt_after(panel: ReturnPanel, cut: pd.Timestamp) -> ReturnPanel:
    """Replace every observation strictly after `cut` with implausible values.

    The corruption is large and of the opposite sign to normal returns, so any
    component that touches it will move visibly rather than by a rounding error.
    """
    def poison(frame: pd.DataFrame | pd.Series, value: float):
        corrupted = frame.copy()
        corrupted.loc[corrupted.index > cut] = value
        return corrupted

    return ReturnPanel(
        returns=poison(panel.returns, 1.0e3),
        market=poison(panel.market, -0.9),
        market_cap=poison(panel.market_cap, 1.0e12),
    )


def decisions_until(config, panel: ReturnPanel, cut: pd.Timestamp) -> tuple:
    """Generate the decision schedule and return the part dated on or before `cut`.

    `decide` rather than `run`: the poisoned panel produces non-finite prices, so
    the ledger cannot simulate it — and simulation is not what this test is
    about.
    """
    weights, records = RollingBacktestEngine(config, panel).decide(seed=0)
    diagnostics = pd.DataFrame([record.to_row() for record in records]).set_index("as_of")
    return weights.loc[:cut], diagnostics.loc[:cut]


@pytest.mark.parametrize("cut_date", CUT_DATES)
@pytest.mark.parametrize("arm", ["static", "hmm", "gmm", "volq"])
def test_decisions_are_unaffected_by_future_data(panel, regime_config, cut_date, arm):
    """No decision at or before the cut may change when the future is corrupted."""
    cut = pd.Timestamp(cut_date)
    config = replace(
        regime_config,
        name=f"lookahead_{arm}",
        regime=replace(
            regime_config.regime,
            detector="none" if arm == "static" else arm,
            overrides={} if arm == "static" else regime_config.regime.overrides,
        ),
    )
    _assert_decisions_match(config, panel, cut)


@pytest.mark.parametrize("cut_date", CUT_DATES)
def test_regime_weighted_moments_are_causal(panel, regime_config, cut_date):
    """The second mechanism must satisfy the same proof as the first.

    Regime-weighted estimation opens two new routes for the future to leak in: a
    five-year lookback instead of one, and a posterior history supplied by the
    detector. Both are covered here rather than argued about.
    """
    cut = pd.Timestamp(cut_date)
    config = replace(
        regime_config,
        name="lookahead_regime_moments",
        moment_estimator="regime_weighted",
        regime=replace(regime_config.regime, detector="hmm"),
    )
    _assert_decisions_match(config, panel, cut)


def _assert_decisions_match(config, panel: ReturnPanel, cut: pd.Timestamp) -> None:
    """Run clean and poisoned panels; require identical decisions through `cut`."""

    clean_weights, clean_diagnostics = decisions_until(config, panel, cut)
    dirty_weights, dirty_diagnostics = decisions_until(config, corrupt_after(panel, cut), cut)


    assert len(clean_weights) > 0, "Test is vacuous if no rebalance precedes the cut."

    # The two runs can hold different names *after* the cut, which widens the
    # column set of the full weight frame. Aligning on the union (missing => not
    # held => 0.0) compares like with like, and still catches a leak that put
    # weight into a name the clean run never touched before the cut.
    columns = clean_weights.columns.union(dirty_weights.columns)
    pd.testing.assert_frame_equal(
        clean_weights.reindex(columns=columns).fillna(0.0),
        dirty_weights.reindex(columns=columns).fillna(0.0),
    )
    pd.testing.assert_series_equal(
        clean_diagnostics["regime_label"], dirty_diagnostics["regime_label"]
    )
    for column in ("param_alpha", "param_lambda_v", "param_max_assets", "objective_value"):
        np.testing.assert_allclose(
            clean_diagnostics[column].to_numpy(dtype=float),
            dirty_diagnostics[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
            err_msg=f"'{column}' changed when future data was corrupted.",
        )


def test_corruption_actually_changes_later_decisions(panel, regime_config):
    """Guard against a vacuous test: the corruption must matter *after* the cut.

    If the poisoned panel produced identical decisions everywhere, the test above
    would pass for the wrong reason — for example because the corruption never
    reached the code path under test.
    """
    cut = pd.Timestamp("2010-06-30")
    config = replace(regime_config, name="lookahead_sanity")

    clean, _ = RollingBacktestEngine(config, panel).decide(seed=0)
    dirty, _ = RollingBacktestEngine(config, corrupt_after(panel, cut)).decide(seed=0)

    after_cut = clean.index > cut
    assert after_cut.any(), "Panel must extend past the cut for this check to mean anything."
    assert not np.allclose(
        clean[after_cut].to_numpy(),
        dirty.reindex(index=clean.index, columns=clean.columns)[after_cut].to_numpy(),
        equal_nan=True,
    ), "Corrupting the future changed nothing at all — the test is not exercising the pipeline."


def test_expanding_standardization_is_causal(panel):
    """Feature values at date t must not move when later market data changes."""
    from fpso.regime.features import RegimeFeatureBuilder

    cut = pd.Timestamp("2010-06-30")
    builder = RegimeFeatureBuilder()

    clean = builder.build(panel.market).loc[:cut]
    poisoned_market = panel.market.copy()
    poisoned_market.loc[poisoned_market.index > cut] = -0.9
    dirty = builder.build(poisoned_market).loc[:cut]

    pd.testing.assert_frame_equal(clean, dirty)
