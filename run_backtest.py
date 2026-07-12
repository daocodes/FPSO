import numpy as np
import pandas as pd

from algorithm.data_processor import (
    clean_and_pivot_data,
    compute_covariance_matrix,
    compute_expected_returns,
    create_wrds_connection,
    get_sp500_constituents,
)
from algorithm.fpso_optimizer import FPSO
from algorithm.utils import generate_naive_equal_weight


def _fetch_returns_matrix(
    db_connection,
    permnos: list[int],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if not permnos:
        return pd.DataFrame()

    permno_sql = ",".join(str(permno) for permno in permnos)
    query = f"""
        SELECT
            date AS dlycaldt,
            permno,
            ret AS dlyret
        FROM crsp.dsf
        WHERE permno IN ({permno_sql})
          AND date BETWEEN DATE '{start_date}' AND DATE '{end_date}'
          AND ret IS NOT NULL
        ORDER BY date, permno
    """
    raw = db_connection.raw_sql(query, date_cols=["dlycaldt"])
    if raw.empty:
        return pd.DataFrame()
    return clean_and_pivot_data(raw)


def _compute_basic_metrics(returns: np.ndarray) -> dict[str, float]:
    r = np.asarray(returns, dtype=float).reshape(-1)
    if r.size == 0:
        return {
            "days": 0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    equity = np.cumprod(1.0 + r)
    total_return = float(equity[-1] - 1.0)
    annual_return = float((1.0 + total_return) ** (252.0 / len(r)) - 1.0)

    daily_std = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    annual_vol = float(daily_std * np.sqrt(252.0))

    daily_mean = float(np.mean(r))
    sharpe = 0.0 if daily_std == 0 else float(np.sqrt(252.0) * daily_mean / daily_std)

    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1.0
    max_drawdown = float(np.min(drawdowns))

    return {
        "days": int(len(r)),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def main():
    # Keep this configuration block short and easy to edit.
    target_date = "2024-01-31"
    train_start = "2023-02-01"
    train_end = "2024-01-31"
    test_start = "2024-02-01"
    test_end = "2024-06-30"
    num_assets = 50

    # 1) Load universe and return matrices.
    db = create_wrds_connection()
    try:
        universe = get_sp500_constituents(db, target_date)[:num_assets]
        train_matrix = _fetch_returns_matrix(db, universe, train_start, train_end)
        test_matrix = _fetch_returns_matrix(db, universe, test_start, test_end)
    finally:
        db.close()

    if train_matrix.empty or test_matrix.empty:
        raise ValueError("Train/test return matrix is empty. Adjust dates or universe.")

    # Ensure both windows use the exact same asset ordering.
    common_assets = train_matrix.columns.intersection(test_matrix.columns)
    if len(common_assets) == 0:
        raise ValueError("No overlapping assets between train and test windows.")
    train_matrix = train_matrix[common_assets]
    test_matrix = test_matrix[common_assets]

    # 2) Build model inputs (mu, Sigma, and previous weights).
    mu = compute_expected_returns(train_matrix)
    sigma = compute_covariance_matrix(train_matrix)
    w_prev = generate_naive_equal_weight(len(common_assets))

    # 3) Run optimizer.
    optimizer = FPSO(
        num_particles=30,
        num_assets=len(common_assets),
        max_iter=60,
        beta_0=1.0,
        gamma=1.0,
        alpha=0.2,
        delta=1e-6,
    )
    optimizer.K = min(20, len(common_assets))
    optimizer.u = 0.15

    best_weights, fitness_history = optimizer.optimize(mu, sigma, w_prev)

    # 4) Evaluate on train and test windows.
    train_returns = train_matrix.to_numpy() @ best_weights
    test_returns = test_matrix.to_numpy() @ best_weights

    train_metrics = _compute_basic_metrics(train_returns)
    test_metrics = _compute_basic_metrics(test_returns)

    # 5) Print a compact summary.
    print("\n=== FPSO Backtest Summary ===")
    print(f"Assets used: {len(common_assets)}")
    print(f"Best fitness: {fitness_history[-1]:.6f}")
    print(f"Weight sum: {np.sum(best_weights):.6f}")
    print(f"Active positions: {int(np.sum(np.abs(best_weights) > 1e-8))}")

    print("\nTrain metrics")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\nTest metrics")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
