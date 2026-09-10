"""Adaptive regularisation: learning how much the optimizer should trust itself.

Two findings motivate this package, both measured on the study's own results:

1. FPSO does not converge. A typical run lands ~17% below the achievable
   objective, and 200 restarts at the study's budget never match one run at 10x
   budget. The deficiency is the continuous weighting, not the stock selection —
   re-solving weights exactly on the swarm's own picks raises the objective by
   ~130% at 100% of rebalances (:mod:`fpso.optimizer.exact`).

2. Fixing it makes the portfolio worse gross of costs. Optimising harder against
   a trailing-mean `mu` amplifies its estimation error; the non-convergence was
   accidental regularisation. Deliberate shrinkage recovers the loss and more.

:mod:`.features` measures how uncertain the optimizer was about its own answer.
:mod:`.shrinkage` learns, from history only, how much to shrink given that
uncertainty — with an oracle ceiling, a signal-free permutation control, and a
best-constant baseline, so a positive result has to survive the same scrutiny
the study applied to regime conditioning.
"""

from fpso.adaptive.features import (
    FEATURE_NAMES,
    UncertaintyFeatures,
    build_features,
    market_features,
    swarm_features,
)
from fpso.adaptive.shrinkage import (
    RidgeShrinkagePolicy,
    ShrinkageGrid,
    oracle_shrinkage,
    shuffled_shrinkage,
    walk_forward_best_constant,
    walk_forward_learned,
)

__all__ = [
    "FEATURE_NAMES",
    "RidgeShrinkagePolicy",
    "ShrinkageGrid",
    "UncertaintyFeatures",
    "build_features",
    "market_features",
    "oracle_shrinkage",
    "shuffled_shrinkage",
    "swarm_features",
    "walk_forward_best_constant",
    "walk_forward_learned",
]
