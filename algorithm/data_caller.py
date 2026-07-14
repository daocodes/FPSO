from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from algorithm.data_processor import create_wrds_connection, get_sp500_constituents


def _resolve_permnos(db_connection, tickers: Iterable[str], end_date: str) -> pd.DataFrame:
    ticker_list = [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
    if not ticker_list:
        return pd.DataFrame()

    ticker_sql = ",".join(f"'{ticker}'" for ticker in ticker_list)
    query = f"""
        SELECT ticker, permno
        FROM crsp.msenames
        WHERE UPPER(ticker) IN ({ticker_sql})
          AND namedt <= DATE '{end_date}'
          AND (nameendt IS NULL OR nameendt >= DATE '{end_date}')
        ORDER BY ticker, nameendt DESC, namedt DESC
    """
    raw = db_connection.raw_sql(query)
    if raw.empty:
        return raw
    return raw.drop_duplicates("ticker", keep="first")


def get_wrds_data(
    tickers: Iterable[str] | None = None,
    start_date: str = "2010-01-01",
    end_date: str = "2024-12-31",
    max_constituents: int = 50,
) -> pd.DataFrame:
    connection = create_wrds_connection()
    try:
        if tickers is not None:
            permno_frame = _resolve_permnos(connection, tickers, end_date)
            if permno_frame.empty:
                raise ValueError("No ticker mappings were found in CRSP msenames.")
            permnos = permno_frame["permno"].astype(int).tolist()
        else:
            permnos = get_sp500_constituents(connection, end_date)[:max_constituents]

        permno_sql = ",".join(str(permno) for permno in permnos)
        query = f"""
            SELECT
                permno,
                date,
                prc,
                vol,
                ret
            FROM crsp.dsf
            WHERE permno IN ({permno_sql})
              AND date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
            ORDER BY permno, date
        """
        data = connection.raw_sql(query, date_cols=["date"])
        if data.empty:
            raise ValueError("No WRDS rows were returned for the requested universe.")
        return data
    finally:
        connection.close()


def normalize_wrds_data(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized = normalized.sort_values(["permno", "date"]).reset_index(drop=True)

    def _zscore(series: pd.Series) -> pd.Series:
        series = pd.to_numeric(series, errors="coerce")
        std = float(series.std(ddof=0))
        if not np.isfinite(std) or std == 0.0:
            return pd.Series(0.0, index=series.index)
        return (series - float(series.mean())) / std

    normalized["price_zscore"] = normalized.groupby("permno", group_keys=False)["prc"].transform(_zscore)
    normalized["volume_zscore"] = normalized.groupby("permno", group_keys=False)["vol"].transform(_zscore)
    normalized["volatility_zscore"] = normalized.groupby("permno", group_keys=False)["ret"].transform(
        lambda series: series.rolling(window=20, min_periods=1).std(ddof=0).fillna(0.0)
    )

    return normalized[["permno", "date", "price_zscore", "volume_zscore", "volatility_zscore"]]


def get_data(*args, **kwargs) -> pd.DataFrame:
    return get_wrds_data(*args, **kwargs)
