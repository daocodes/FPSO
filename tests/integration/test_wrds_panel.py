"""Live WRDS checks. Skipped without credentials; excluded from the CI contract.

These verify the *shape and semantics* of what CRSP returns, not performance.
They are the tests that would catch a WRDS schema change or an entitlement
problem before it silently corrupted a cached panel.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from dotenv import load_dotenv

from fpso.config.schema import DataConfig
from fpso.data.source import WRDSPanelSource

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def source() -> WRDSPanelSource:
    load_dotenv()
    if not os.getenv("WRDS_USERNAME") or not os.getenv("WRDS_PASSWORD"):
        pytest.skip("WRDS credentials are not set; see .env.example.")
    config = DataConfig(source="wrds", panel_start="2019-01-01", panel_end="2019-06-30")
    return WRDSPanelSource(config, universe_pool=20)


def test_panel_has_the_expected_shape(source):
    """A short two-quarter pull should give a dense, well-formed panel."""
    panel = source.load()

    assert not panel.returns.empty
    assert panel.returns.index.is_monotonic_increasing
    assert not panel.returns.index.has_duplicates
    assert panel.returns.index.equals(panel.market_cap.index)
    assert panel.market.index.equals(panel.returns.index)
    # 2019 H1 has ~124 trading days.
    assert 110 < len(panel.returns) < 135


def test_returns_are_plausible_daily_equity_returns(source):
    """Guards against a units error — e.g. CRSP returning percent, not decimal."""
    panel = source.load()
    returns = panel.returns.stack()

    assert returns.abs().median() < 0.05
    assert returns.min() > -1.0, "A daily return below -100% is impossible."
    assert returns.abs().max() < 2.0


def test_market_proxy_tracks_the_cross_section(source):
    """The CRSP value-weighted index must correlate strongly with the panel mean."""
    panel = source.load()
    cross_section = panel.returns.mean(axis=1)
    assert cross_section.corr(panel.market) > 0.85


def test_dtypes_are_plain_numpy_floats(source):
    """The WRDS driver returns nullable Float64, which numba cannot compile.

    `ReturnPanel` normalises it; if that ever regresses, vectorbt fails with an
    opaque typing error deep inside a numba kernel — so it is checked here.
    """
    panel = source.load()
    assert set(panel.returns.dtypes) == {pd.api.types.pandas_dtype("float64")}
    assert panel.market.dtype == pd.api.types.pandas_dtype("float64")


def test_cache_round_trip(source, tmp_path):
    """A pulled panel must reload from Parquet unchanged."""
    from fpso.data.panel import ReturnPanel

    panel = source.load()
    panel.to_parquet(tmp_path)
    reloaded = ReturnPanel.from_parquet(tmp_path)

    pd.testing.assert_frame_equal(panel.returns, reloaded.returns, check_freq=False)
    pd.testing.assert_series_equal(panel.market, reloaded.market, check_freq=False)
