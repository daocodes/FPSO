"""Immutable configuration objects for the regime-conditioned FPSO study.

Every knob the experiment can turn lives in one of the frozen dataclasses below.
Two properties matter for the paper:

1. *Frozen* means a running backtest cannot mutate its own configuration, so the
   manifest written next to a result set is guaranteed to describe the run that
   produced it.
2. The regime mechanism is expressed as **data** (a mapping from regime label to
   parameter overrides) rather than code. The static arm and the regime arm then
   execute the same code path and differ only in configuration, which is what
   makes the headline ablation in the paper a clean comparison.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any


class RegimeLabel(IntEnum):
    """Canonical market states, ordered by increasing realized volatility.

    Hidden-state indices from an HMM are arbitrary and permute between refits, so
    every detector must map its raw states onto this ordered enum before the
    parameter policy sees them (see :mod:`fpso.regime.hmm`).
    """

    CALM = 0
    TURBULENT = 1
    CRISIS = 2


@dataclass(frozen=True)
class FPSOParams:
    """Hyperparameters of the FPSO optimizer and of the objective it maximizes.

    Attributes are grouped as: swarm size/budget, firefly attraction terms, PSO
    velocity terms, objective trade-off weights, penalty weights, and the hard
    constraints. The regime policy overrides a subset of these per market state.
    """

    # --- Search budget -----------------------------------------------------
    num_particles: int = 30
    max_iter: int = 60

    # --- Firefly attraction (Yang 2008) ------------------------------------
    beta_0: float = 1.0
    """Attractiveness at zero distance."""
    gamma: float = 1.0
    """Light-absorption coefficient; larger => attraction decays faster with distance."""
    alpha: float = 0.2
    """Scale of the uniform random walk (exploration)."""
    delta: float = 1e-6
    """Numerical floor in the centroid denominator."""

    # --- PSO refinement ----------------------------------------------------
    # The paper describes FPSO as firefly centroid attraction *combined with*
    # PSO personal/global-best refinement. These three terms implement that
    # second half; setting pso_blend = 0 recovers the pure-firefly ablation.
    inertia: float = 0.5
    """Velocity carry-over coefficient w in the PSO update."""
    c_cognitive: float = 0.5
    """Pull toward the particle's own best position."""
    c_social: float = 0.5
    """Pull toward the swarm's best position."""
    pso_blend: float = 0.5
    """Convex weight on the PSO step vs. the firefly step (0 = pure firefly)."""

    # --- Objective R(w) = mu'w - lambda_v * sqrt(w'Sigma w) - lambda_t * TO --
    lambda_v: float = 1.0
    lambda_t: float = 0.01

    # --- Penalty weights on constraint violation (rho_1..rho_3 in the paper) -
    rho_budget: float = 100.0
    rho_box: float = 100.0
    rho_cardinality: float = 100.0

    # --- Hard constraints --------------------------------------------------
    max_assets: int = 20
    """Cardinality bound K: at most this many non-zero positions."""
    max_weight: float = 0.15
    """Box bound u: per-asset upper weight limit."""

    def with_overrides(self, overrides: Mapping[str, Any]) -> FPSOParams:
        """Return a copy with `overrides` applied, validating every field name.

        The regime policy calls this. Failing loudly on an unknown key matters:
        a typo in a YAML regime block would otherwise silently produce a run that
        is identical to the static baseline.
        """
        unknown = set(overrides) - {f for f in self.__dataclass_fields__}
        if unknown:
            raise ValueError(
                f"Unknown FPSOParams override(s): {sorted(unknown)}. "
                f"Valid fields: {sorted(self.__dataclass_fields__)}"
            )
        return replace(self, **dict(overrides))


@dataclass(frozen=True)
class RegimeConfig:
    """How the latent market state is estimated and mapped onto parameters."""

    detector: str = "hmm"
    """One of: none, hmm, gmm, volq, shuffled, oracle (see fpso.regime.registry)."""
    n_states: int = 3
    covariance_type: str = "diag"
    features: tuple[str, ...] = ("rv_21", "rv_ratio", "ret_63", "dd_252")
    refit_months: int = 12
    """Refit cadence. Inference still runs every rebalance; only fitting is periodic."""
    min_dwell: int = 2
    """A new regime must persist this many rebalances before the acting label switches."""
    burn_in_months: int = 36
    """Regime conditioning stays inactive until this much feature history exists."""
    overrides: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    """RegimeLabel name -> FPSOParams field overrides. Empty => static parameterization."""
    detector_seed: int = 0
    """Seed for detectors with their own randomness (EM init, label shuffling)."""
    constant_label: str = "CALM"
    """Label pinned by the `constant` detector; ignored by every other detector."""
    archive_seeds: int = 0
    """How many swarm particles to seed from the regime solution archive.

    0 disables the mechanism entirely, which is the default and reproduces every
    arm run before it existed. The archive is a *search accelerator*: it changes
    where the search starts, never the objective, the constraints, or the
    universe, so any effect it has is on time-to-converge rather than on the
    optimum being sought.
    """
    archive_capacity: int = 5
    """Solutions retained per state. See fpso.regime.archive."""
    archive_mode: str = "weights"
    """How an archived solution becomes a seed: "weights" replays the stored
    vector, "support" replays only the asset selection and applies it to current
    holdings. See fpso.regime.archive.RegimeSolutionArchive.seeds_for."""

    def overrides_for(self, label: RegimeLabel) -> Mapping[str, float]:
        """Parameter overrides for `label`, or an empty mapping if unspecified."""
        return self.overrides.get(label.name, {})


@dataclass(frozen=True)
class ScheduleConfig:
    """Rebalance cadence and the length of the trailing estimation window."""

    start: str = "2011-01-01"
    end: str = "2024-12-31"
    frequency: str = "monthly"
    """monthly (the paper's stated cadence) or annual (kept as a frequency ablation)."""
    estimation_window_days: int = 252
    """Trailing calendar window used to estimate mu and Sigma at each rebalance."""
    regime_window_days: int = 1260
    """Longer window used when the moment estimator is regime-weighted. Weighting
    discards the days that do not match the current regime, so the same 252-day
    window would leave far too few effective observations behind."""
    min_observations: int = 120
    """Assets with fewer observations in the estimation window are dropped."""


@dataclass(frozen=True)
class DataConfig:
    """Where the return panel comes from and how the universe is chosen."""

    source: str = "parquet"
    """parquet (cached CRSP panel) or synthetic (offline, deterministic)."""
    cache_dir: str = "cache"
    panel_start: str = "2005-01-01"
    """Panel starts well before the evaluation window to cover HMM burn-in."""
    panel_end: str = "2024-12-31"
    universe_size: int = 50
    universe_rule: str = "top_market_cap"
    """top_market_cap (point-in-time, unbiased) or first_permno (the legacy, biased rule)."""
    use_delisting_returns: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """A single experimental arm: everything needed to reproduce one result set."""

    name: str
    optimizer: str = "fpso"
    """fpso, equal_weight, or min_variance."""
    moment_estimator: str = "sample"
    """sample (flat trailing window), regime_weighted (the beliefs mechanism), or
    regime_tilt (the expected-return mechanism)."""
    tilt_scorer: str = "defensive"
    """Which cross-section the tilt points at: defensive (low volatility, derived
    from returns), net_issuance or size (both exogenous — see fpso.data.tilt)."""
    tilt_strength: Mapping[str, float] = field(default_factory=dict)
    """RegimeLabel name -> tilt strength on mu, in units of sd(mu). Empty leaves
    mu untouched, so `regime_tilt` with no strengths is exactly the baseline."""
    seeds: tuple[int, ...] = tuple(range(30))
    transaction_cost_rates: tuple[float, ...] = (0.0, 0.005, 0.01, 0.015)
    base_params: FPSOParams = field(default_factory=FPSOParams)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    data: DataConfig = field(default_factory=DataConfig)
    results_dir: str = "results"

    @property
    def is_regime_conditioned(self) -> bool:
        """True when the arm actually varies parameters by detected regime."""
        return self.regime.detector != "none" and bool(self.regime.overrides)

    @property
    def uses_regime_moments(self) -> bool:
        """True when the regime signal enters the optimizer's beliefs, not just
        its preferences."""
        return self.moment_estimator != "sample"
