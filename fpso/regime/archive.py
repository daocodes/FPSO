"""A regime-keyed archive of previously discovered portfolios.

The archive is the study's search-acceleration mechanism. It sits *outside* the
optimizer: the detector names the current market state, the archive returns the
best portfolios previously found while in that state, and those are handed to
FPSO as additional starting points. FPSO is never told that regimes exist — it
receives a list of weight vectors and searches from them.

**Why starting points rather than parameters or the universe.** Two other
injection points were tested in this study and neither accelerates anything.
Conditioning FPSO's hyperparameters changes where the search goes, not how long
it takes. Pruning the investable universe shrinks a term that is not the
bottleneck: measured end to end, cutting 50 assets to 20 collapses the
combinatorial space by ~10^9 and buys 8% of runtime, because cost is dominated
by `num_particles * max_iter` evaluations rather than by problem width. Solve
time is close to linear in the evaluation count, so the only lever that matters
is *reaching a good solution in fewer iterations* — which is what a good
starting point buys.

**Causality.** `record` is only ever called with solutions already computed at
earlier rebalances, and `seeds_for` reads that history. No future information can
enter, which the lookahead test in `tests/unit/test_lookahead.py` verifies for
the pipeline as a whole.

**Universe drift.** Holdings are stored by asset identifier, not by position, so
an archived portfolio survives reconstitution: retrieval reindexes onto today's
asset ordering and drops names that have left the universe. The caller repairs
the result, which restores the budget constraint after any such drop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ArchivedSolution:
    """One portfolio the optimizer found, tagged with the state it was found in."""

    weights: pd.Series
    """Holdings indexed by asset identifier, so the entry survives universe drift."""
    objective_value: float
    """Fitness at the time it was archived; used to keep only the best entries."""


@dataclass
class RegimeSolutionArchive:
    """Best-known portfolios per regime state, used to seed later searches.

    Args:
        capacity: How many solutions to retain per state. Small on purpose — the
            swarm has 30 particles and seeding too many of them with variations
            on the same past portfolio would collapse the diversity the firefly
            phase exists to maintain.
    """

    capacity: int = 5
    _by_state: dict[str, list[ArchivedSolution]] = field(default_factory=dict)

    def record(self, state: str, weights: pd.Series, objective_value: float) -> None:
        """Add a solved portfolio, keeping only the `capacity` best for that state."""
        if not np.isfinite(objective_value):
            return
        entries = self._by_state.setdefault(state, [])
        entries.append(ArchivedSolution(weights.copy(), float(objective_value)))
        entries.sort(key=lambda e: e.objective_value, reverse=True)
        del entries[self.capacity :]

    def seeds_for(
        self,
        state: str,
        assets: pd.Index,
        limit: int,
        weights_prev: np.ndarray | None = None,
        mode: str = "weights",
    ) -> list[np.ndarray]:
        """Starting points for `state`, on today's `assets` ordering.

        Two strategies, because the obvious one does not work:

        ``weights``
            Replay the archived weight vector directly. Measured on the real
            panel this seeds the swarm *worse* than a random Dirichlet draw
            (mean initial fitness 0.204 against 0.228). An archived portfolio was
            optimal for a different month's mu, Sigma and holdings, and the
            turnover term scores it against *today's* incumbent, so the staleness
            is charged straight to its fitness.

        ``support``
            Replay only which assets were held and apply that selection to the
            current portfolio: keep today's weight wherever the archived support
            agrees, open a small position in archived names not currently held.
            The seed stays near the incumbent, so it is not punished by the
            turnover term, while still carrying the archived selection. This
            transfers the part of a past solution that plausibly generalizes —
            which names suit this state — and discards the part that cannot.

        Absent names become zero weight; the caller repairs, which reinstates the
        budget constraint. An unseen state returns an empty list and the swarm
        falls back to random initialization, so the mechanism degrades to the
        baseline rather than failing.
        """
        seeds = []
        for entry in self._by_state.get(state, ())[:limit]:
            if mode == "support" and weights_prev is not None:
                seed = self._seed_from_support(entry, assets, weights_prev)
            else:
                seed = entry.weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
            if seed.sum() > 0:
                seeds.append(seed)
        return seeds

    @staticmethod
    def _seed_from_support(
        entry: ArchivedSolution, assets: pd.Index, weights_prev: np.ndarray
    ) -> np.ndarray:
        """Today's holdings restricted to an archived selection.

        Names in both keep their current weight, so turnover against the incumbent
        stays small. Archived names not currently held get the average incumbent
        weight, which is enough to put them in play without displacing the
        portfolio wholesale. Repair renormalizes.
        """
        in_support = assets.isin(entry.weights.index[entry.weights > 1e-8])
        seed = np.where(in_support, weights_prev, 0.0)
        opening = in_support & (weights_prev <= 1e-8)
        if opening.any():
            held = weights_prev[weights_prev > 1e-8]
            seed[opening] = float(held.mean()) if held.size else 1.0 / len(assets)
        return seed

    def size(self, state: str) -> int:
        """How many solutions are held for a state, for the diagnostics table."""
        return len(self._by_state.get(state, ()))

    @property
    def states_seen(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_state))
