import os
from typing import List

import wrds
from dotenv import load_dotenv
from sqlalchemy.exc import ProgrammingError
import numpy as np
import pandas as pd


def create_wrds_connection() -> wrds.Connection:
    load_dotenv()
    username = os.getenv("WRDS_USERNAME", "").strip()
    password = os.getenv("WRDS_PASSWORD", "").strip()
    if not username or not password:
        raise ValueError("Missing WRDS_USERNAME or WRDS_PASSWORD in environment.")
    return wrds.Connection(wrds_username=username, wrds_password=password)


def get_sp500_constituents(
    db_connection: wrds.Connection, target_date: str
) -> List[int]:
    official_query = f"""
        SELECT DISTINCT permno
        FROM crsp.msp500list
        WHERE start <= DATE '{target_date}'
          AND (ending IS NULL OR ending >= DATE '{target_date}')
        ORDER BY permno
    """
    try:
        official_df = db_connection.raw_sql(official_query)
        return official_df["permno"].astype(int).tolist()
    except ProgrammingError as exc:
        if "permission denied for schema crsp_a_indexes" not in str(exc).lower():
            raise

    # Research fallback when index-constituent entitlement is unavailable:
    # approximate S&P 500 using the largest 500 U.S. common stocks by market cap.
    fallback_query = f"""
        WITH latest_trading_day AS (
            SELECT MAX(date) AS date
            FROM crsp.dsf
            WHERE date <= DATE '{target_date}'
        )
        SELECT d.permno
        FROM crsp.dsf AS d
        JOIN latest_trading_day ltd
          ON d.date = ltd.date
        JOIN crsp.msenames AS n
          ON d.permno = n.permno
         AND n.namedt <= d.date
         AND d.date <= n.nameendt
        WHERE d.prc IS NOT NULL
          AND d.shrout IS NOT NULL
          AND n.shrcd IN (10, 11)
          AND n.exchcd IN (1, 2, 3)
        ORDER BY ABS(d.prc) * d.shrout DESC
        LIMIT 500
    """
    fallback_df = db_connection.raw_sql(fallback_query)
    if fallback_df.empty:
        raise ValueError(
            f"No proxy constituents available on or before {target_date}."
        )

    print(
        "Warning: official S&P 500 constituents unavailable for this WRDS account; "
        "using top-500 market-cap proxy from CRSP."
    )
    return fallback_df["permno"].astype(int).tolist()


def clean_and_pivot_data(window_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw long-format data from WRDS into a clean, wide-format matrix.
    
    Input: window_df (Columns: 'dlycaldt', 'permno', 'dlyret')
    Output: clean_df (Index: 'dlycaldt', Columns: 'permno', Values: 'dlyret')
    """
    # 1. Pivot the table from long format to wide format
    # This turns unique dates into rows and unique permnos into columns
    pivoted_df = window_df.pivot(
        index='dlycaldt', 
        columns='permno', 
        values='dlyret'
    )
    
    # 2. Sort the index to ensure dates are in chronological order
    pivoted_df = pivoted_df.sort_index()
    
    # 3. Clean missing values (NaNs)
    # If a stock has missing data for a day, forward-fill it with the previous day's return
    # If there's no previous day (missing at the start), fill it with 0.0 (no return)
    cleaned_df = pivoted_df.ffill().fillna(0.0)
    
    # 4. Optional: Filter out columns (assets) with too much missing data
    # (e.g., if a stock was missing for more than 10% of the days before filling, drop it)
    max_missing_pct = 0.10
    raw_missing_pct = pivoted_df.isna().mean()
    valid_columns = raw_missing_pct[raw_missing_pct <= max_missing_pct].index
    
    final_df = cleaned_df[valid_columns]
    
    return final_df

def compute_expected_returns(clean_df: pd.DataFrame) -> np.ndarray:
    """
    Collapses the historical daily returns table into a single 1D vector 
    representing the annualized expected return (mu) for each asset.
    """
    # 1. Calculate the arithmetic mean of daily returns for each column (asset)
    daily_means = clean_df.mean()
    
    # 2. Annualize the daily returns (multiplying by 252 standard trading days)
    annualized_returns = daily_means * 252
    
    # 3. Strip away Pandas labels and return as a raw 1D NumPy array
    return annualized_returns.to_numpy()


def compute_covariance_matrix(clean_df: pd.DataFrame) -> np.ndarray:
    """
    Collapses the historical daily returns table into a static 2D grid 
    representing the annualized variance and covariance (Sigma) between all assets.
    """
    # 1. Calculate the daily empirical covariance matrix across all columns
    daily_covariance = clean_df.cov()
    
    # 2. Annualize the covariance matrix (multiplying by 252 standard trading days)
    annualized_covariance = daily_covariance * 252
    
    # 3. Strip away Pandas labels and return as a raw 2D symmetric NumPy array
    return annualized_covariance.to_numpy()




if __name__ == "__main__":
    db = create_wrds_connection()
    try:
        target_date = "2024-01-31"
        start_date = "2023-02-01"

        permnos = get_sp500_constituents(db, target_date)
        print(f"constituents: {len(permnos)}")

        # Keep smoke test fast by using a smaller subset.
        sample_permnos = permnos[:50]
        permno_sql = ",".join(str(permno) for permno in sample_permnos)

        window_query = f"""
            SELECT
                date AS dlycaldt,
                permno,
                ret AS dlyret
            FROM crsp.dsf
            WHERE permno IN ({permno_sql})
              AND date BETWEEN DATE '{start_date}' AND DATE '{target_date}'
              AND ret IS NOT NULL
        """
        window_df = db.raw_sql(window_query, date_cols=["dlycaldt"])
        print(f"raw rows: {len(window_df)}")

        clean_df = clean_and_pivot_data(window_df)
        mu = compute_expected_returns(clean_df)
        sigma = compute_covariance_matrix(clean_df)

        print(f"clean shape: {clean_df.shape}")
        print(f"mu shape: {mu.shape}")
        print(f"sigma shape: {sigma.shape}")

        assert sigma.shape[0] == sigma.shape[1], "Sigma must be square."
        assert sigma.shape[0] == mu.shape[0], "mu and Sigma dimensions must align."
        print("smoke test passed")
    finally:
        db.close()
