"""Regime detection, smoothing and the regime-to-parameter policy."""

from __future__ import annotations

import numpy as np

from fpso.config.schema import RegimeConfig
from fpso.regime.alternatives import GaussianMixtureDetector, VolatilityQuantileDetector
from fpso.regime.base import NullDetector, RegimeAssignment, RegimeDetector
from fpso.regime.controls import (
    ConstantLabelDetector,
    OracleOOSDetector,
    ShuffledLabelDetector,
)
from fpso.regime.features import FEATURE_LABELS, RegimeFeatureBuilder
from fpso.regime.hmm import GaussianHMMDetector
from fpso.regime.policy import RegimeDecision, RegimeParameterPolicy
from fpso.regime.smoothing import MinimumDwellSmoother

__all__ = [
    "FEATURE_LABELS",
    "ConstantLabelDetector",
    "GaussianHMMDetector",
    "GaussianMixtureDetector",
    "MinimumDwellSmoother",
    "NullDetector",
    "OracleOOSDetector",
    "RegimeAssignment",
    "RegimeDecision",
    "RegimeDetector",
    "RegimeFeatureBuilder",
    "RegimeParameterPolicy",
    "ShuffledLabelDetector",
    "VolatilityQuantileDetector",
    "build_detector",
]


def build_detector(config: RegimeConfig, rng: np.random.Generator) -> RegimeDetector:
    """Instantiate the detector named by `config.detector`.

    Only the shuffled control consumes `rng`; every other detector is a
    deterministic function of the data it is allowed to see.
    """
    if config.detector == "none":
        return NullDetector()
    if config.detector == "constant":
        return ConstantLabelDetector(config)
    if config.detector == "hmm":
        return GaussianHMMDetector(config)
    if config.detector == "gmm":
        return GaussianMixtureDetector(config)
    if config.detector == "volq":
        return VolatilityQuantileDetector(config)
    if config.detector == "shuffled":
        return ShuffledLabelDetector(config, rng)
    if config.detector == "oracle":
        return OracleOOSDetector(config)
    raise ValueError(
        f"Unknown regime.detector '{config.detector}'; expected one of: "
        "none, constant, hmm, gmm, volq, shuffled, oracle."
    )
