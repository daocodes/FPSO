"""How many of its own positions can the optimizer defend?

A population method returns one portfolio and discards the rest of the evidence.
Run it again under a different seed and it returns a different one: across this
study's 30 matched seeds there are 30 distinct supports at all 168 rebalances,
and no two seeds ever select the same 20 names.

The fraction of runs holding a given asset is therefore a measurement, and an
unusually well-behaved one. It is **intrinsically faithful** — computed from the
decisions themselves rather than generated after the fact by a second model — so
unlike a post-hoc attribution it cannot disagree with what the optimizer did. And
it is the artifact a convex solver structurally cannot produce, because a solver
that returns a single point has no population to take a frequency over.

What it says here is not flattering, which is the point. Roughly four of twenty
held positions appear in 90% or more of runs; the rest are selections from a pool
of around forty contested candidates. A faithful account of *when a decision
system has no defensible reason for its choice* is worth more than a fluent
explanation of a choice that was arbitrary.

**Read this together with the non-convergence result.** Inclusion frequency does
not, on its own, distinguish "many portfolios are genuinely near-optimal" from
"the search stopped in a different place each time". The landscape probe settles
that separately — within 0.1% of the optimum there is essentially a unique
support — so in this study the dispersion is solver behaviour, not a property of
the problem. Anyone reusing this measure on another optimizer owes the same check
before interpreting it.

The cleaner object is an epsilon-band on *fitness* rather than agreement across
terminal points, since positional agreement conflates a flat objective with a
search that has not converged. That requires logging the terminal swarm; the
cross-run version here needs only the results already on disk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "ACTIVE_WEIGHT",
    "STRONG_CONVICTION",
    "CONTESTED_BAND",
    "ConvictionProfile",
    "inclusion_frequency",
    "conviction_profile",
]

ACTIVE_WEIGHT = 1e-8
"""Weight above which a position counts as held.

Matches the cardinality test used by the repair operator, so "held" means the
same thing here as it does inside the optimizer.
"""

STRONG_CONVICTION = 0.9
"""Fraction of runs that must agree before a holding counts as defensible."""

CONTESTED_BAND = (0.1, 0.9)
"""Frequencies at which runs genuinely disagree about a name.

Below the band a name is essentially rejected, above it essentially chosen; in
between the optimizer is not expressing a view, it is landing somewhere.
"""


def inclusion_frequency(weights_by_run: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Fraction of runs holding each asset at each rebalance.

    Args:
        weights_by_run: run identifier (seed) -> `(dates, assets)` target weights.
            Runs must share a rebalance index; they need not share columns, since
            a name absent from one run is simply not held in it.

    Returns:
        `(dates, assets)` frequencies in [0, 1], over the union of all columns.

    Raises:
        ValueError: if fewer than two runs are supplied, or their rebalance dates
            disagree. Both are silent-corruption risks rather than edge cases: a
            single run yields a frequency matrix of zeros and ones that looks like
            total conviction, and misaligned dates would average different
            decisions together.
    """
    if len(weights_by_run) < 2:
        raise ValueError(
            f"Inclusion frequency needs at least two runs; got {len(weights_by_run)}. "
            "One run would report every holding as unanimous."
        )

    frames = list(weights_by_run.values())
    index = frames[0].index
    for frame in frames[1:]:
        if not frame.index.equals(index):
            raise ValueError("Runs disagree on their rebalance dates.")

    columns = sorted(set().union(*(frame.columns for frame in frames)))
    held = np.stack([
        (frame.reindex(columns=columns).fillna(0.0).to_numpy() > ACTIVE_WEIGHT)
        for frame in frames
    ])
    return pd.DataFrame(held.mean(axis=0), index=index, columns=columns)


@dataclass(frozen=True)
class ConvictionProfile:
    """Per-rebalance summary of how much the runs agreed.

    Attributes:
        table: one row per rebalance with the candidate/unanimous/strong/
            contested counts.
        mean_strong: average number of holdings meeting `STRONG_CONVICTION`.
        mean_contested: average number of names inside `CONTESTED_BAND`.
        share_without_unanimity: fraction of rebalances at which not one name was
            held by every run.
    """

    table: pd.DataFrame
    mean_strong: float
    mean_contested: float
    share_without_unanimity: float

    def summary(self) -> str:
        """One line for a caption or a log."""
        return (
            f"{self.mean_strong:.1f} defensible holdings, "
            f"{self.mean_contested:.1f} contested candidates, "
            f"no unanimous name at {self.share_without_unanimity:.1%} of rebalances"
        )


def conviction_profile(frequency: pd.DataFrame) -> ConvictionProfile:
    """Summarise an inclusion-frequency matrix into per-rebalance counts.

    Only names some run actually held enter the counts. The point-in-time universe
    changes over the evaluation window, so counting across the full column set
    would silently mix "no run chose this" with "this was not investable yet" and
    make the denominator drift with the panel rather than with the decision.
    """
    low, high = CONTESTED_BAND
    rows, dates = [], []
    for date, row in frequency.iterrows():
        candidates = row[row > 0.0]
        if candidates.empty:
            # No run held anything here — a skipped rebalance, not a rebalance at
            # which the runs disagreed. Its date is dropped with it; carrying the
            # rows forward against a positional slice of the original index would
            # relabel every later row with the wrong date.
            continue
        dates.append(date)
        rows.append({
            "candidates": int(candidates.size),
            "unanimous": int((candidates >= 1.0).sum()),
            "strong": int((candidates >= STRONG_CONVICTION).sum()),
            "contested": int(((candidates >= low) & (candidates <= high)).sum()),
            "mean_frequency": float(candidates.mean()),
        })
    if not rows:
        raise ValueError("No rebalance had a single held name.")

    table = pd.DataFrame(rows, index=pd.Index(dates, name=frequency.index.name))
    return ConvictionProfile(
        table=table,
        mean_strong=float(table["strong"].mean()),
        mean_contested=float(table["contested"].mean()),
        share_without_unanimity=float((table["unanimous"] == 0).mean()),
    )
