"""Optimizers, the constraint set they respect, and the objective they maximize."""

from fpso.optimizer.base import OptimizationResult, PortfolioOptimizer
from fpso.optimizer.constraints import ConstraintSet, SimplexBoxCardinalityRepair
from fpso.optimizer.fpso import FPSOOptimizer, Particle, Swarm
from fpso.optimizer.objective import (
    ConstraintViolation,
    MeanVarianceTurnover,
    Objective,
    PenalizedObjective,
)

__all__ = [
    "ConstraintSet",
    "ConstraintViolation",
    "FPSOOptimizer",
    "MeanVarianceTurnover",
    "Objective",
    "OptimizationResult",
    "Particle",
    "PenalizedObjective",
    "PortfolioOptimizer",
    "SimplexBoxCardinalityRepair",
    "Swarm",
]


def build_optimizer(name: str, params):
    """Instantiate the allocator named in ``ExperimentConfig.optimizer``."""
    from fpso.baselines.allocators import (
        EqualWeightAllocator,
        LowVolatilityAllocator,
        MinimumVarianceAllocator,
    )
    from fpso.optimizer.fpso_cw import FPSOCWOptimizer

    allocators = {
        "fpso": FPSOOptimizer,
        "fpso_cw": FPSOCWOptimizer,
        "equal_weight": EqualWeightAllocator,
        "min_variance": MinimumVarianceAllocator,
        "low_volatility": LowVolatilityAllocator,
    }
    if name not in allocators:
        raise ValueError(
            f"Unknown optimizer '{name}'; expected one of {sorted(allocators)}."
        )
    return allocators[name](params)
