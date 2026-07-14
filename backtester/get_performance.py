import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from algorithm.fpso_optimizer import firefly
from algorithm.data_caller import get_data, normalize_wrds_data

#takes the performance and plots the cumulative return of the strategy
#random perturbation epsilon is adjusted each iteration (n=50 here) to give us a robust understanding


def to_1d_array(result):
    array = np.asarray(result, dtype=float).reshape(-1)
    return array

def compute_sharpe(returns):
    sharpe_values = np.empty(len(returns), dtype=float)
    for index in range(len(returns)):
        window = returns[: index + 1]
        window_std = window.std(ddof=1)
        if window_std == 0 or np.isnan(window_std) or len(window) < 2:
            sharpe_values[index] = 0.0
        else:
            sharpe_values[index] = np.sqrt(252) * window.mean() / window_std
    return sharpe_values

def _build_random_portfolio_returns(data, asset_cap=30, timesteps=100, seed=None):
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "permno"]).dropna(subset=["ret"])

    dates = pd.Index(frame["date"].drop_duplicates().sort_values())
    rng = np.random.default_rng(seed)
    portfolio_returns = []

    for start_index in range(0, max(len(dates) - 1, 0), timesteps):
        rebalance_date = dates[start_index]
        holding_dates = dates[start_index + 1 : start_index + 1 + timesteps]

        if len(holding_dates) == 0:
            break

        universe = frame.loc[frame["date"] == rebalance_date, "permno"].dropna().unique()
        if len(universe) == 0:
            continue

        chosen_assets = rng.choice(universe, size=min(asset_cap, len(universe)), replace=False)
        weights = rng.dirichlet(np.ones(len(chosen_assets)))

        window = frame.loc[
            frame["date"].isin(holding_dates) & frame["permno"].isin(chosen_assets),
            ["date", "permno", "ret"],
        ]

        daily_matrix = (
            window.pivot_table(index="date", columns="permno", values="ret", aggfunc="last")
            .reindex(index=holding_dates, columns=chosen_assets)
            .fillna(0.0)
        )

        period_returns = daily_matrix.to_numpy() @ weights
        portfolio_returns.extend(period_returns.tolist())

    return np.asarray(portfolio_returns, dtype=float)

def track_random_performance(data=None, asset_cap=30, iterations=50, timesteps=100, seed=42):
    if data is None:
        data = get_data()

    simulated_returns = []

    for iteration in range(iterations):
        result = _build_random_portfolio_returns(
            data,
            asset_cap=asset_cap,
            timesteps=timesteps,
            seed=seed + iteration,
        )
        simulated_returns.append(to_1d_array(result))

    if not simulated_returns:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    return_matrix = np.vstack(simulated_returns)
    average_returns = return_matrix.mean(axis=0)
    cumulative_performance = np.cumprod(1.0 + average_returns) - 1.0

    periods = np.arange(1, len(average_returns) + 1, dtype=float)
    annualized_performance = np.where(
        periods > 0,
        np.power(1.0 + cumulative_performance, 252.0 / periods) - 1.0,
        0.0,
    )

    sharpe = compute_sharpe(average_returns)
    return cumulative_performance, annualized_performance, sharpe

def track_performance(data, epsilon=0.01, iterations=50, timesteps=100):
    simulated_returns = []

    for _ in range(iterations):
        data_copy = data.copy()
        result = firefly(data_copy, epsilon=epsilon, timesteps=timesteps, asset_cap=30)
        simulated_returns.append(to_1d_array(result))

    return_matrix = np.vstack(simulated_returns)
    average_returns = return_matrix.mean(axis=0)
    cumulative_performance = np.cumprod(1.0 + average_returns) - 1.0

    periods = np.arange(1, len(average_returns) + 1, dtype=float)
    annualized_performance = np.where(
        periods > 0,
        np.power(1.0 + cumulative_performance, 252.0 / periods) - 1.0,
        0.0,
    )

    sharpe = compute_sharpe(average_returns)
    return cumulative_performance, annualized_performance, sharpe

def track_oracle_performance(data, timesteps=100, transaction_cost=0.001):
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "permno"]).dropna(subset=["ret"])

    dates = pd.Index(frame["date"].drop_duplicates().sort_values())
    oracle_returns = []

    for start_index in range(timesteps, len(dates), timesteps):
        holding_dates = dates[start_index : start_index + timesteps]
        if len(holding_dates) == 0:
            break

        universe = frame.loc[frame["date"] == dates[start_index - 1], "permno"].dropna().unique()
        if len(universe) == 0:
            continue

        window = frame.loc[
            frame["date"].isin(holding_dates) & frame["permno"].isin(universe),
            ["date", "permno", "ret"],
        ]
        daily_matrix = (
            window.pivot_table(index="date", columns="permno", values="ret", aggfunc="last")
            .reindex(index=holding_dates, columns=universe)
            .sort_index()
            .ffill()
            .fillna(0.0)
        )
        if daily_matrix.empty:
            continue

        best_returns = daily_matrix.max(axis=1).to_numpy(dtype=float)
        holdings = daily_matrix.idxmax(axis=1).to_numpy()
        costs = np.zeros_like(best_returns)
        if len(holdings) > 1:
            switches = holdings[1:] != holdings[:-1]
            costs[1:][switches] = transaction_cost
        costs[-1] += transaction_cost
        oracle_returns.extend((best_returns - costs).tolist())

    oracle_returns = np.asarray(oracle_returns, dtype=float)
    if oracle_returns.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    cumulative_performance = np.cumprod(1.0 + oracle_returns) - 1.0
    periods = np.arange(1, len(oracle_returns) + 1, dtype=float)
    annualized_performance = np.where(
        periods > 0,
        np.power(1.0 + cumulative_performance, 252.0 / periods) - 1.0,
        0.0,
    )
    sharpe = compute_sharpe(oracle_returns)
    return cumulative_performance, annualized_performance, sharpe

def visualize_performance(returns, sharpe, annualized_performance):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(returns, color="blue", linewidth=1.5)
    axes[0].set_title("Returns over time")
    axes[0].set_ylabel("Returns")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(annualized_performance, color="green", linewidth=1.5)
    axes[1].set_title("Annualized performance over time")
    axes[1].set_ylabel("Annualized Return")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
    return fig, axes