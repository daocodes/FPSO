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
from algorithm.portfolio_model import calculate_portfolio_return
from algorithm.utils import generate_naive_equal_weight
from backtester.get_performance import (
    path_performance,
    summarize_performance,
    to_1d_array,
    visualize_performance,
)


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


def _align_prev_weights(
    w_prev: pd.Series | None, assets: pd.Index
) -> np.ndarray:
    """Map last-period weights onto this window's asset list (missing => 0)."""
    if w_prev is None:
        return generate_naive_equal_weight(len(assets))
    aligned = w_prev.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    # If everything was new names, fall back to equal weight so TO is well-defined.
    if float(np.sum(aligned)) < 1e-12:
        return generate_naive_equal_weight(len(assets))
    return aligned


def _one_way_turnover(w_new: np.ndarray, w_prev: np.ndarray) -> float:
    """One-way turnover = 0.5 * L1 weight change (standard portfolio convention)."""
    return float(0.5 * np.sum(np.abs(w_new - w_prev)))


def _apply_transaction_cost(
    gross_returns: np.ndarray, one_way_turnover: float, tc_rate: float
) -> np.ndarray:
    """
    Subtract proportional trading cost on the rebalance day (first day of the window).

    cost = tc_rate * one_way_turnover
    Example: tc_rate=0.01 and turnover=0.40 => pay 0.4% of NAV once at rebalance.
    """
    net = to_1d_array(gross_returns).copy()
    if net.size == 0:
        return net
    net[0] -= tc_rate * one_way_turnover
    return net


def _run_window(
    db,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    num_assets: int,
    w_prev: pd.Series | None = None,
) -> dict:
    """Optimize on one 12-month train window; evaluate gross OOS returns + turnover."""
    universe = get_sp500_constituents(db, train_end)[:num_assets]
    train_matrix = _fetch_returns_matrix(db, universe, train_start, train_end)
    test_matrix = _fetch_returns_matrix(db, universe, test_start, test_end)

    if train_matrix.empty or test_matrix.empty:
        raise ValueError(
            f"Empty returns for train {train_start}:{train_end} "
            f"or test {test_start}:{test_end}."
        )

    common_assets = train_matrix.columns.intersection(test_matrix.columns)
    if len(common_assets) == 0:
        raise ValueError(f"No overlapping assets for window ending {train_end}.")
    train_matrix = train_matrix[common_assets]
    test_matrix = test_matrix[common_assets]

    w_prev_arr = _align_prev_weights(w_prev, common_assets)
    mu = compute_expected_returns(train_matrix)
    sigma = compute_covariance_matrix(train_matrix)

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

    best_weights, _ = optimizer.optimize(mu, sigma, w_prev_arr)
    turnover = _one_way_turnover(best_weights, w_prev_arr)

    # In-sample objective R(w*) from portfolio_model.py:
    # R(w) = mu^T w - lambda_v * sqrt(w^T Sigma w) - lambda_t * TO(w, w_prev)
    objective_R = calculate_portfolio_return(
        weights=best_weights,
        expected_returns=mu,
        covariance=sigma,
        weights_prev=w_prev_arr,
        lambda_v=optimizer.lambda_v,
        lambda_t=optimizer.lambda_t,
    )

    # OOS / train PnL paths use realized asset returns × weights (not R(w)).
    train_returns = to_1d_array(train_matrix.to_numpy() @ best_weights)
    test_returns_gross = to_1d_array(test_matrix.to_numpy() @ best_weights)
    w_star = pd.Series(best_weights, index=common_assets)

    return {
        "w_star": w_star,
        "turnover": turnover,
        "objective_R": objective_R,
        "train_returns": train_returns,
        "test_returns_gross": test_returns_gross,
        "train_metrics": summarize_performance(train_returns),
        "test_metrics_gross": summarize_performance(test_returns_gross),
    }


def main():
    # Full sample: 2010–2024 with rolling 12-month train → next-year hold.
    sample_start_year = 2010
    sample_end_year = 2024
    num_assets = 50
    # One-way proportional cost rates (same sweep style as the CNN-EF paper).
    tc_rates = [0.0, 0.005, 0.01, 0.015]
    # Set True to open the cumulative / Sharpe plots from get_performance.
    show_plots = False

    window_results: list[dict] = []
    w_prev: pd.Series | None = None

    print(
        f"Starting rolling backtest: train {sample_start_year}–"
        f"{sample_end_year - 1}, OOS {sample_start_year + 1}–{sample_end_year}",
        flush=True,
    )
    db = create_wrds_connection()
    try:
        for train_year in range(sample_start_year, sample_end_year):
            test_year = train_year + 1
            train_start = f"{train_year}-01-01"
            train_end = f"{train_year}-12-31"
            test_start = f"{test_year}-01-01"
            test_end = f"{test_year}-12-31"

            print(
                f"\nWindow: train {train_year} -> test {test_year} "
                f"(universe as of {train_end})",
                flush=True,
            )
            result = _run_window(
                db,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                num_assets=num_assets,
                w_prev=w_prev,
            )
            window_results.append(
                {
                    "test_year": test_year,
                    "turnover": result["turnover"],
                    "objective_R": result["objective_R"],
                    "test_returns_gross": result["test_returns_gross"],
                    "test_metrics_gross": result["test_metrics_gross"],
                }
            )
            # Next rebalance measures trading from this portfolio.
            w_prev = result["w_star"]

            print(
                f"  R(w*)={result['objective_R']:.4f} | "
                f"gross OOS ann={result['test_metrics_gross']['annual_return']:.4f} | "
                f"sharpe={result['test_metrics_gross']['sharpe']:.4f} | "
                f"one-way TO={result['turnover']:.4f}",
                flush=True,
            )
    finally:
        db.close()

    if not window_results:
        raise ValueError("No completed rolling windows.")

    print("\n=== FPSO Rolling Backtest Summary (2010–2024) ===")
    print(f"Windows: {len(window_results)}")
    print(f"OOS span: {sample_start_year + 1}–{sample_end_year}")
    print(f"Assets per window (cap): {num_assets}")

    print("\n--- Transaction-cost robustness ---")
    print(
        f"{'TC':>8} {'AnnRet':>10} {'AnnVol':>10} {'Sharpe':>10} {'MDD':>10} {'AvgTO':>10}"
    )
    net_by_tc: dict[float, np.ndarray] = {}
    for tc in tc_rates:
        net_chunks = [
            _apply_transaction_cost(row["test_returns_gross"], row["turnover"], tc)
            for row in window_results
        ]
        net_path = to_1d_array(np.concatenate(net_chunks))
        net_by_tc[tc] = net_path
        metrics = summarize_performance(net_path)
        avg_to = float(np.mean([row["turnover"] for row in window_results]))
        print(
            f"{tc:8.1%} {metrics['annual_return']:10.4f} "
            f"{metrics['annual_vol']:10.4f} {metrics['sharpe']:10.4f} "
            f"{metrics['max_drawdown']:10.4f} {avg_to:10.4f}"
        )

    print("\nPer-window gross OOS (TC = 0)")
    for row in window_results:
        m = row["test_metrics_gross"]
        print(
            f"  {row['test_year']}: "
            f"ann={m['annual_return']:.4f}, "
            f"sharpe={m['sharpe']:.4f}, "
            f"mdd={m['max_drawdown']:.4f}, "
            f"TO={row['turnover']:.4f}"
        )

    # Full OOS path diagnostics from get_performance helpers.
    baseline_path = net_by_tc[0.0]
    paths = path_performance(baseline_path)
    print("\n--- Full OOS path (TC = 0) via get_performance ---")
    print(f"  final cumulative return: {paths['cumulative'][-1]:.4f}")
    print(f"  final annualized return: {paths['annualized'][-1]:.4f}")
    print(f"  final expanding Sharpe:  {paths['sharpe'][-1]:.4f}")

    if show_plots:
        visualize_performance(
            baseline_path,
            sharpe=paths["sharpe"],
            annualized_performance=paths["annualized"],
        )


if __name__ == "__main__":
    main()
