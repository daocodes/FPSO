"""How uncertain is the optimizer about its own answer?

The swarm ends every rebalance holding `num_particles` feasible candidates and
returns exactly one. The rest are discarded, and with them the only direct
evidence the system produces about whether its answer was well determined.

Measured on this study's results, that evidence is informative: cross-run
disagreement predicts realised cross-run *outcome* dispersion (Spearman rho =
0.36 on the ex-ante quality spread, HAC t = 5.49) while carrying no information
about the *level* of returns (every test null, p >= 0.27). That is the profile
of an uncertainty quantifier, not a timing signal — so it is used here to decide
how much to trust the return forecasts, never to decide whether to be invested.

Two families of feature, both strictly ex ante:

*Swarm* features describe how much the search disagreed with itself. They are
available inside a single run, so nothing here requires the 30-seed matrix.

*Market* features describe how well determined the problem was to begin with:
if expected returns are indistinguishable across assets, or the covariance is
ill-conditioned, no optimizer can identify a portfolio and the forecast deserves
less weight.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from fpso.data.moments import Moments

__all__ = ["UncertaintyFeatures", "swarm_features", "market_features", "FEATURE_NAMES"]

_ACTIVE = 1e-8


@dataclass(frozen=True)
class UncertaintyFeatures:
    """One rebalance's worth of ex-ante uncertainty evidence."""

    # --- swarm: how much did the search disagree with itself? --------------
    weight_dispersion: float
    """Mean pairwise one-way distance between candidate portfolios."""
    support_disagreement: float
    """1 - mean pairwise Jaccard overlap of asset selections; 0 = unanimous."""
    fitness_spread: float
    """Spread of candidate objective values, scaled by the best. Large means
    the swarm could not separate good candidates from bad ones."""
    distinct_supports: float
    """Fraction of candidates with a unique asset selection."""

    # --- market: was the problem well determined at all? -------------------
    mu_dispersion: float
    """Cross-sectional SD of expected returns. Near zero, assets are
    indistinguishable on return and the selection is arbitrary."""
    mean_volatility: float
    avg_correlation: float
    condition_number: float
    """log10 condition number of Sigma; high means the risk model is degenerate."""

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in FEATURE_NAMES], dtype=float)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


FEATURE_NAMES: tuple[str, ...] = (
    "weight_dispersion",
    "support_disagreement",
    "fitness_spread",
    "distinct_supports",
    "mu_dispersion",
    "mean_volatility",
    "avg_correlation",
    "condition_number",
)


def swarm_features(population: np.ndarray, fitness: np.ndarray) -> dict[str, float]:
    """Disagreement statistics over a terminal swarm.

    Args:
        population: `(n_particles, n_assets)` array of feasible candidate weights.
        fitness: `(n_particles,)` objective value of each candidate.

    The fitness spread is scaled by the magnitude of the best candidate so it is
    comparable across dates; the objective is an annualised return net of risk
    and can legitimately be negative, so the absolute value is used rather than a
    ratio that would invert with the sign.
    """
    population = np.atleast_2d(np.asarray(population, dtype=float))
    fitness = np.asarray(fitness, dtype=float).ravel()
    n = len(population)
    if n < 2:
        return {
            "weight_dispersion": 0.0,
            "support_disagreement": 0.0,
            "fitness_spread": 0.0,
            "distinct_supports": 0.0,
        }

    supports = population > _ACTIVE
    pair_distance: list[float] = []
    pair_overlap: list[float] = []
    for a in range(n):
        for b in range(a + 1, n):
            pair_distance.append(0.5 * float(np.abs(population[a] - population[b]).sum()))
            union = int(np.sum(supports[a] | supports[b]))
            inter = int(np.sum(supports[a] & supports[b]))
            pair_overlap.append(inter / union if union else 1.0)

    scale = max(abs(float(np.max(fitness))), 1e-9)
    unique = {tuple(np.flatnonzero(row)) for row in supports}
    return {
        "weight_dispersion": float(np.mean(pair_distance)),
        "support_disagreement": 1.0 - float(np.mean(pair_overlap)),
        "fitness_spread": float(np.max(fitness) - np.min(fitness)) / scale,
        "distinct_supports": len(unique) / n,
    }


def market_features(moments: Moments) -> dict[str, float]:
    """How well determined the allocation problem is, before any search."""
    variances = np.diag(moments.sigma)
    volatility = np.sqrt(np.clip(variances, 0.0, None))
    denominator = np.outer(volatility, volatility)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.where(denominator > 0, moments.sigma / denominator, 0.0)
    off_diagonal = correlation[np.triu_indices_from(correlation, k=1)]

    eigenvalues = np.linalg.eigvalsh(moments.sigma)
    smallest = max(float(eigenvalues.min()), 1e-18)
    condition = float(eigenvalues.max()) / smallest

    return {
        "mu_dispersion": float(np.std(moments.mu, ddof=1)) if len(moments.mu) > 1 else 0.0,
        "mean_volatility": float(np.mean(volatility)),
        "avg_correlation": float(np.mean(off_diagonal)) if off_diagonal.size else 0.0,
        "condition_number": float(np.log10(max(condition, 1.0))),
    }


def build_features(
    population: np.ndarray, fitness: np.ndarray, moments: Moments
) -> UncertaintyFeatures:
    """Assemble the full ex-ante feature vector for one rebalance."""
    return UncertaintyFeatures(**swarm_features(population, fitness), **market_features(moments))
