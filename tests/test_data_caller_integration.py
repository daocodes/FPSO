import os

import pandas as pd
import pytest
from dotenv import load_dotenv

from algorithm.data_caller import get_wrds_data


@pytest.mark.integration
def test_get_wrds_data_returns_expected_schema():
    load_dotenv()
    username = os.getenv("WRDS_USERNAME")
    password = os.getenv("WRDS_PASSWORD")
    if not username or not password:
        pytest.skip("WRDS credentials are not set in environment.")

    try:
        data = get_wrds_data()
    except OSError as exc:
        if "reading from stdin" in str(exc):
            pytest.skip("WRDS prompted for interactive credentials; skipping integration test.")
        raise

    expected_columns = {"permno", "date", "prc", "vol", "ret"}
    assert expected_columns.issubset(set(data.columns))
    assert not data.empty
    assert pd.api.types.is_datetime64_any_dtype(data["date"])
