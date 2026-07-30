"""The rolling backtest engine.

This is the only component in the package that touches dates. Everything
downstream — universe, moments, detector, optimizer — receives data that has
already been sliced, so "no lookahead" is a property of one loop rather than an
invariant re-argued in six modules. That single choke point is what makes the
lookahead test meaningful.

Per rebalance date *t*:

1. Pick the universe investable at *t*.
2. Slice returns on ``[t - estimation_window, t]`` and estimate mu, Sigma.
3. Ask the regime policy for the parameters to use at *t* (fit-if-due, infer,
   smooth, honour burn-in).
4. Solve for target weights, warm-started from current holdings.
5. Record the decision. The target is executed against *t*'s close, so it earns
   returns from *t+1* onward.

Randomness is drawn from a per-(arm, seed, rebalance) stream derived from one
master ``SeedSequence``. Two consequences: a run is bit-reproducible from its
config plus its seed, and inserting or removing a rebalance date cannot silently
shift the random numbers every other rebalance receives.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from fpso.backtest.ledger import VectorBTLedger
from fpso.backtest.results import BacktestResult, RebalanceRecord
from fpso.backtest.schedule import build_schedule
from fpso.config.schema import ExperimentConfig, FPSOParams
from fpso.data.moments import EstimationContext, build_moment_estimator
from fpso.data.panel import ReturnPanel
from fpso.data.universe import build_universe_provider
from fpso.optimizer import build_optimizer
from fpso.optimizer.base import PortfolioOptimizer
from fpso.regime import RegimeFeatureBuilder, build_detector
from fpso.regime.policy import RegimeParameterPolicy

# Fields recorded per rebalance. The regime-conditioned subset is what the paper
# reports as "the mechanism"; recording all of them lets a reviewer verify that
# nothing else moved.
RECORDED_PARAMS = (
    "alpha", "gamma", "lambda_v", "lambda_t", "max_assets", "max_weight",
    "pso_blend", "num_particles", "max_iter",
)


class RollingBacktestEngine:
    """Runs one (arm, seed) backtest over the configured schedule."""

    def __init__(self, config: ExperimentConfig, panel: ReturnPanel):
        self.config = config
        self.panel = panel
        self.schedule = build_schedule(config.schedule.frequency)
        self.universe_provider = build_universe_provider(config.data, panel)
        self.moment_estimator = build_moment_estimator(
            config.moment_estimator, config.schedule.min_observations
        )
        self.feature_builder = RegimeFeatureBuilder(features=config.regime.features)
        self.features = self.feature_builder.build(panel.market)

    def run(self, seed: int) -> BacktestResult:
        """Decide, then simulate: the full backtest for one seed."""
        target_weights, records = self.decide(seed)
        return self._simulate(target_weights, records, seed)

    def decide(self, seed: int) -> tuple[pd.DataFrame, list[RebalanceRecord]]:
        """Produce the target-weight schedule without simulating an equity path.

        Split out from :meth:`run` because the causality property is a property of
        *decisions*, not of the accounting that follows. The lookahead test calls
        this directly, which lets it corrupt future prices arbitrarily without the
        ledger objecting to the resulting non-finite equity curve.
        """
        streams = _SeedStreams(arm=self.config.name, seed=seed)
        optimizer = build_optimizer(self.config.optimizer, self.config.base_params)
        policy = RegimeParameterPolicy(
            detector=build_detector(self.config.regime, streams.detector_rng()),
            config=self.config.regime,
            base_params=self.config.base_params,
        )
        policy.reset()

        rebalance_dates = self._rebalance_dates()
        weight_rows: dict[pd.Timestamp, pd.Series] = {}
        records: list[RebalanceRecord] = []
        holdings = pd.Series(dtype=float)

        for as_of in rebalance_dates:
            decision = policy.decide(self.features, as_of)
            outcome = self._solve_at(
                as_of=as_of,
                params=decision.params,
                optimizer=optimizer,
                holdings=holdings,
                rng=streams.rebalance_rng(as_of),
                context=self._estimation_context(policy, decision, as_of),
            )
            if outcome is None:
                continue

            weights, result = outcome
            weight_rows[as_of] = weights
            # `holdings` carries last month's *target*, not the drifted portfolio,
            # so the optimizer's turnover term is measured against the target it
            # last set. Measured on the static arm, one month of drift moves the
            # weights by an L1 of 0.044 against a mean target-to-target change of
            # 0.81 — 5.4% of the term, which at lambda_t <= 0.03 is roughly 1e-3
            # of the objective. The ledger simulates drift exactly, so *reported*
            # turnover and transaction costs are unaffected by this simplification.
            holdings = weights
            records.append(
                RebalanceRecord(
                    as_of=as_of,
                    regime_label=decision.label.name,
                    regime_posterior=tuple(float(p) for p in decision.assignment.posterior),
                    regime_applied=decision.applied,
                    is_burn_in=decision.assignment.is_burn_in,
                    n_assets=len(weights),
                    n_active=result.n_active,
                    objective_value=result.objective_value,
                    solve_seconds=result.solve_seconds,
                    params={k: getattr(decision.params, k) for k in RECORDED_PARAMS},
                )
            )

        if not weight_rows:
            raise RuntimeError(
                f"Arm '{self.config.name}' produced no rebalances between "
                f"{self.config.schedule.start} and {self.config.schedule.end}. "
                "Check that the panel covers the schedule window."
            )

        return pd.DataFrame(weight_rows).T.fillna(0.0).sort_index(), records

    # ------------------------------------------------------------- internals --

    def _rebalance_dates(self) -> pd.DatetimeIndex:
        """Schedule dates that fall inside the evaluation window."""
        all_dates = self.schedule.dates(self.panel.dates)
        start = pd.Timestamp(self.config.schedule.start)
        end = pd.Timestamp(self.config.schedule.end)
        return all_dates[(all_dates >= start) & (all_dates <= end)]

    def _solve_at(
        self,
        as_of: pd.Timestamp,
        params: FPSOParams,
        optimizer: PortfolioOptimizer,
        holdings: pd.Series,
        rng: np.random.Generator,
        context=None,
    ):
        """Optimize at one date, or return None when the window is unusable."""
        universe = self.universe_provider.at(as_of)
        if len(universe) < 2:
            return None

        lookback = (
            self.config.schedule.regime_window_days
            if self.moment_estimator.needs_regime_context
            else self.config.schedule.estimation_window_days
        )
        window = self.panel.returns_between(
            as_of - pd.Timedelta(days=lookback), as_of, assets=universe
        )
        try:
            moments = self.moment_estimator.estimate(window, context)
        except ValueError:
            # Too few usable assets in this window (early sample, or a data gap).
            # Skipping leaves the previous portfolio in place, which is what an
            # investor unable to re-estimate would actually do.
            return None

        # `params` may differ from `self.config.base_params` when the regime
        # policy is active, so the optimizer is rebuilt with the acting values.
        acting_optimizer = (
            build_optimizer(self.config.optimizer, params)
            if params is not self.config.base_params
            else optimizer
        )

        weights_prev = holdings.reindex(moments.assets).fillna(0.0).to_numpy(dtype=float)
        if weights_prev.sum() <= 1e-12:
            weights_prev = np.full(len(moments.assets), 1.0 / len(moments.assets))

        result = acting_optimizer.solve(moments, weights_prev, rng)
        return pd.Series(result.weights, index=moments.assets), result

    def _estimation_context(self, policy, decision, as_of: pd.Timestamp):
        """Regime posteriors for the moment estimator, or None if it wants none.

        Skipped entirely for the flat estimator, and during burn-in, so an arm
        that does not use regime beliefs pays nothing for the machinery — and so
        the regime-weighted estimator falls back to the flat one before the
        detector is trustworthy, exactly as the parameter policy does.
        """
        if not self.moment_estimator.needs_regime_context:
            return None
        if decision.assignment.is_burn_in:
            return None

        posteriors = policy.detector.posterior_series(self.features, as_of)
        if posteriors is None or posteriors.empty:
            return None
        return EstimationContext(
            as_of=as_of,
            posterior=decision.assignment.posterior,
            posteriors=posteriors,
        )

    def _simulate(
        self,
        target_weights: pd.DataFrame,
        records: list[RebalanceRecord],
        seed: int,
    ) -> BacktestResult:
        """Hand the weight schedule to vectorbt, once per transaction-cost rate."""
        first_rebalance = target_weights.index[0]
        end = pd.Timestamp(self.config.schedule.end)
        evaluation_returns = self.panel.returns.loc[first_rebalance:end, target_weights.columns]

        ledger = VectorBTLedger(evaluation_returns)
        returns_by_cost, equity_by_cost, turnover_by_cost = {}, {}, {}
        for rate in self.config.transaction_cost_rates:
            outcome = ledger.simulate(target_weights, rate)
            returns_by_cost[rate] = outcome.returns
            equity_by_cost[rate] = outcome.equity
            turnover_by_cost[rate] = outcome.turnover

        return BacktestResult(
            arm=self.config.name,
            seed=seed,
            target_weights=target_weights,
            rebalances=records,
            returns_by_cost=returns_by_cost,
            equity_by_cost=equity_by_cost,
            turnover_by_cost=turnover_by_cost,
            config=self.config,
        )


class _SeedStreams:
    """Independent random streams keyed by (arm, seed, rebalance date).

    Deriving each stream from a hash of its key, rather than from a counter,
    means the numbers a given rebalance sees do not depend on how many
    rebalances preceded it. Adding a date to the schedule therefore changes that
    date's draws and nothing else — which is what allows the lookahead test to
    compare two runs decision by decision.
    """

    def __init__(self, arm: str, seed: int):
        self.arm = arm
        self.seed = seed

    def detector_rng(self) -> np.random.Generator:
        """Stream for detectors with their own randomness (the shuffled control)."""
        return np.random.default_rng(self._entropy("detector"))

    def rebalance_rng(self, as_of: pd.Timestamp) -> np.random.Generator:
        """Stream for the optimizer at one rebalance date."""
        return np.random.default_rng(self._entropy(as_of.strftime("%Y-%m-%d")))

    def _entropy(self, key: str) -> np.random.SeedSequence:
        digest = hashlib.sha256(f"{self.arm}|{self.seed}|{key}".encode()).digest()
        return np.random.SeedSequence(int.from_bytes(digest[:8], "big"))
