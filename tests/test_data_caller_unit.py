import numpy as np
import pandas as pd

from algorithm.data_caller import normalize_wrds_data


def test_normalize_wrds_data_returns_expected_columns_and_sort_order():
    dates = pd.date_range("2021-01-01", periods=260, freq="D")
    first_asset = pd.DataFrame(
        {
            "permno": 10001,
            "date": dates,
            "prc": np.linspace(10.0, 20.0, len(dates)),
            "vol": np.linspace(1000.0, 2000.0, len(dates)),
            "ret": 0.0,
        }
    )
    second_asset = pd.DataFrame(
        {
            "permno": 10002,
            "date": dates,
            "prc": np.linspace(30.0, 40.0, len(dates)),
            "vol": np.linspace(3000.0, 4000.0, len(dates)),
            "ret": 0.0,
        }
    )
    input_frame = pd.concat([first_asset, second_asset], ignore_index=True).sample(
        frac=1, random_state=7
    )

    result = normalize_wrds_data(input_frame)

    assert list(result.columns) == [
        "permno",
        "date",
        "price_zscore",
        "volume_zscore",
        "volatility_zscore",
    ]
    assert result[["permno", "date"]].equals(
        result.sort_values(["permno", "date"])[["permno", "date"]]
    )
    assert np.isfinite(result["price_zscore"]).all()
    assert np.isfinite(result["volume_zscore"]).all()
    assert result["volatility_zscore"].notna().any()


def test_normalize_wrds_data_handles_zero_std_columns():
    dates = pd.date_range("2022-01-01", periods=10, freq="D")
    input_frame = pd.DataFrame(
        {
            "permno": 10003,
            "date": dates,
            "prc": 15.0,
            "vol": 1200.0,
            "ret": 0.0,
        }
    )

    result = normalize_wrds_data(input_frame)

    assert (result["price_zscore"] == 0.0).all()
    assert (result["volume_zscore"] == 0.0).all()
    assert (result["volatility_zscore"] == 0.0).all()
