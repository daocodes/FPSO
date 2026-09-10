"""Backtest engine, schedule, and the vectorbt ledger.

The ledger tests check the accounting the paper depends on: that a fully-invested
target reproduces the underlying asset return exactly at zero cost, that
transaction costs monotonically reduce performance, and that the portfolio drifts
between rebalances rather than being silently re-weighted daily.

`test_cost_drag_is_twice_rate_times_one_way_turnover` is the load-bearing one.
The study's central claim is that net performance is ordered by turnover, so the
relationship between the cost axis and the reported turnover is not an accounting
detail — it is the claim. That relationship carries a factor of two, it was once
mislabeled on three figure axes, and this test is what stops that recurring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpso.backtest.engine import RollingBacktestEngine
from fpso.backtest.ledger import VectorBTLedger, cost_per_unit_turnover
from fpso.backtest.schedule import AnnualSchedule, MonthlySchedule

# ------------------------------------------------------------------ schedule --

def test_monthly_schedule_lands_on_trading_days(panel):
    dates = MonthlySchedule().dates(panel.dates)
    assert set(dates).issubset(set(panel.dates))
    assert dates.is_monotonic_increasing
    assert len(dates.to_period("M").unique()) == len(dates)


def test_monthly_gives_far_more_decisions_than_annual(panel):
    """The cadence change is the reason regime conditioning can work at all."""
    monthly = MonthlySchedule().dates(panel.dates)
    annual = AnnualSchedule().dates(panel.dates)
    assert len(monthly) > 10 * len(annual)


# -------------------------------------------------------------------- ledger --

@pytest.fixture
def two_asset_returns():
    dates = pd.bdate_range("2020-01-01", periods=120)
    generator = np.random.default_rng(0)
    return pd.DataFrame(
        generator.normal(0.0005, 0.01, (120, 2)), index=dates, columns=[1, 2]
    )


def test_full_allocation_to_one_asset_reproduces_its_return(two_asset_returns):
    """At zero cost, a 100% position must earn exactly that asset's return."""
    ledger = VectorBTLedger(two_asset_returns)
    targets = pd.DataFrame(
        [[1.0, 0.0]], index=[two_asset_returns.index[0]], columns=two_asset_returns.columns
    )
    result = ledger.simulate(targets, transaction_cost_rate=0.0)

    # The order fills against the first bar's close, so the position earns from
    # the second bar onward; compare over that span.
    realised = (1.0 + result.returns.iloc[1:]).prod()
    expected = (1.0 + two_asset_returns[1].iloc[1:]).prod()
    assert realised == pytest.approx(expected, rel=1e-6)


def test_transaction_costs_reduce_performance_monotonically(two_asset_returns):
    ledger = VectorBTLedger(two_asset_returns)
    targets = pd.DataFrame(
        [[0.5, 0.5], [0.9, 0.1], [0.2, 0.8]],
        index=two_asset_returns.index[[0, 40, 80]],
        columns=two_asset_returns.columns,
    )
    finals = [
        ledger.simulate(targets, rate).equity.iloc[-1]
        for rate in (0.0, 0.005, 0.01, 0.02)
    ]
    assert finals == sorted(finals, reverse=True)


def test_portfolio_drifts_between_rebalances(two_asset_returns):
    """Weights are only correct on rebalance days; between them the market moves them.

    A ledger that re-weighted daily would silently add a rebalancing bonus and
    understate turnover, so this asserts the drift is real.
    """
    ledger = VectorBTLedger(two_asset_returns)
    targets = pd.DataFrame(
        [[0.5, 0.5]], index=[two_asset_returns.index[0]], columns=two_asset_returns.columns
    )
    result = ledger.simulate(targets, transaction_cost_rate=0.0)

    # Exactly one rebalance means exactly one set of orders.
    assert len(result.turnover) == 1
    # A daily-rebalanced portfolio would earn the average of the two assets;
    # a drifting one does not.
    drifting = (1.0 + result.returns.iloc[1:]).prod()
    rebalanced_daily = (1.0 + two_asset_returns.iloc[1:].mean(axis=1)).prod()
    assert drifting != pytest.approx(rebalanced_daily, rel=1e-9)


def test_turnover_is_reported_per_rebalance(two_asset_returns):
    ledger = VectorBTLedger(two_asset_returns)
    dates = two_asset_returns.index[[0, 40, 80]]
    targets = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], index=dates, columns=two_asset_returns.columns
    )
    result = ledger.simulate(targets, transaction_cost_rate=0.0)

    assert len(result.turnover) == 3
    # The full switch at date 2 trades essentially the whole portfolio one way.
    assert result.turnover.loc[dates[1]] == pytest.approx(1.0, abs=0.02)


def test_cost_per_unit_turnover_is_twice_the_per_trade_rate():
    """The conversion every table and axis label must go through."""
    assert cost_per_unit_turnover(0.0) == 0.0
    assert cost_per_unit_turnover(0.005) == pytest.approx(0.010)
    assert cost_per_unit_turnover(0.015) == pytest.approx(0.030)


@pytest.mark.parametrize("rate", [0.005, 0.01, 0.015])
def test_cost_drag_is_twice_rate_times_one_way_turnover(two_asset_returns, rate):
    """The invariant that makes the cost axis readable next to a turnover figure.

    The fee is charged per trade on traded notional; the turnover reported
    alongside it is one-way, i.e. already halved. So the drag a reader should
    expect from `rate` and `turnover` is `cost_per_unit_turnover(rate) * turnover`,
    not `rate * turnover`.

    This was mislabeled as "one-way transaction cost" on three figure axes, which
    invited exactly the factor-of-two error — in a study whose central claim is
    about the product of these two quantities. Asserting it here means the
    convention cannot drift back silently.

    The realised gap runs ~1-3% under the notional prediction because the fee is
    taken out of the portfolio before the day's return is computed, so the
    tolerance is one-sided in magnitude rather than exact.
    """
    ledger = VectorBTLedger(two_asset_returns)
    dates = two_asset_returns.index[[0, 40]]
    targets = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]], index=dates, columns=two_asset_returns.columns
    )

    free = ledger.simulate(targets, transaction_cost_rate=0.0)
    paid = ledger.simulate(targets, transaction_cost_rate=rate)

    for trade_date in dates:
        realised_drag = free.returns.loc[trade_date] - paid.returns.loc[trade_date]
        predicted = cost_per_unit_turnover(rate) * paid.turnover.loc[trade_date]
        assert realised_drag == pytest.approx(predicted, rel=0.035)
        # And the naive product must be visibly wrong, or the test proves nothing.
        assert realised_drag > 1.5 * rate * paid.turnover.loc[trade_date]

    assert paid.cost_per_unit_turnover == pytest.approx(2.0 * rate)


# -------------------------------------------------------------------- engine --

def test_engine_produces_one_weight_row_per_rebalance(base_config, panel):
    result = RollingBacktestEngine(base_config, panel).run(seed=0)
    schedule_dates = MonthlySchedule().dates(panel.dates)
    expected = schedule_dates[
        (schedule_dates >= pd.Timestamp(base_config.schedule.start))
        & (schedule_dates <= pd.Timestamp(base_config.schedule.end))
    ]
    assert len(result.target_weights) == len(expected)
    assert len(result.rebalances) == len(expected)


def test_every_recorded_portfolio_is_fully_invested(base_config, panel):
    result = RollingBacktestEngine(base_config, panel).run(seed=0)
    np.testing.assert_allclose(result.target_weights.sum(axis=1), 1.0, atol=1e-9)


def test_engine_respects_the_cardinality_constraint(base_config, panel):
    result = RollingBacktestEngine(base_config, panel).run(seed=0)
    active = (result.target_weights > 1e-8).sum(axis=1)
    assert active.max() <= base_config.base_params.max_assets


def test_regime_arm_varies_its_parameters(regime_config, panel):
    """The mechanism must actually fire; otherwise the arm is the static baseline."""
    result = RollingBacktestEngine(regime_config, panel).run(seed=0)
    diagnostics = result.diagnostics

    assert diagnostics["regime_applied"].any(), "Regime overrides never applied."
    assert diagnostics.loc[diagnostics["regime_applied"], "param_alpha"].nunique() > 1


def test_results_round_trip_through_disk(base_config, panel, tmp_path):
    result = RollingBacktestEngine(base_config, panel).run(seed=0)
    saved = result.save(tmp_path)

    from fpso.backtest.results import BacktestResult

    reloaded = BacktestResult.load(saved)
    pd.testing.assert_frame_equal(result.target_weights, reloaded.target_weights)
    # check_freq=False: Parquet does not round-trip a DatetimeIndex's inferred
    # `freq`, which the synthetic panel's business-day index carries. The dates
    # and values are identical, which is what the artifact needs to preserve.
    pd.testing.assert_series_equal(
        result.returns(0.0), reloaded.returns(0.0), check_names=False, check_freq=False
    )
    assert (saved / "manifest.json").exists()
