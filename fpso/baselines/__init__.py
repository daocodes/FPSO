"""Standard external baselines evaluated alongside FPSO."""

from fpso.baselines.allocators import EqualWeightAllocator, MinimumVarianceAllocator

__all__ = ["EqualWeightAllocator", "MinimumVarianceAllocator"]
