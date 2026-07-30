"""Shared fixtures. Every unit test runs offline against the synthetic panel."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fpso.config.schema import (
    DataConfig,
    ExperimentConfig,
    FPSOParams,
    RegimeConfig,
    ScheduleConfig,
)
from fpso.data.source import SyntheticPanelSource


@pytest.fixture(scope="session")
def panel():
    """A deterministic regime-switching market. Built once; treated as read-only."""
    return SyntheticPanelSource("2005-01-01", "2016-12-31", n_assets=25).load()


@pytest.fixture
def fast_params() -> FPSOParams:
    """A search budget small enough to run inside a test."""
    return FPSOParams(num_particles=8, max_iter=6, max_assets=8, max_weight=0.30)


@pytest.fixture
def base_config(fast_params) -> ExperimentConfig:
    """A short static-arm experiment over the synthetic panel."""
    return ExperimentConfig(
        name="test_static",
        seeds=(0,),
        transaction_cost_rates=(0.0, 0.005),
        base_params=fast_params,
        regime=RegimeConfig(detector="none", overrides={}),
        schedule=ScheduleConfig(start="2009-01-01", end="2012-12-31"),
        data=DataConfig(source="synthetic", panel_start="2005-01-01", panel_end="2016-12-31",
                        universe_size=20),
    )


@pytest.fixture
def regime_config(base_config) -> ExperimentConfig:
    """The same experiment with the HMM detector and a live parameter policy."""
    return replace(
        base_config,
        name="test_regime",
        regime=RegimeConfig(
            detector="hmm",
            burn_in_months=6,
            refit_months=12,
            overrides={
                "CALM": {"alpha": 0.10, "lambda_v": 0.60},
                "TURBULENT": {"alpha": 0.30, "lambda_v": 1.60},
                "CRISIS": {"alpha": 0.35, "lambda_v": 3.00, "max_assets": 5},
            },
        ),
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
