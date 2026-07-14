"""Performance helpers used by the FPSO rolling backtest."""

import numpy as np
import matplotlib.pyplot as plt


def to_1d_array(result) -> np.ndarray:
    return np.asarray(result, dtype=float).reshape(-1)


def compute_sharpe(returns) -> np.ndarray:
    """Expanding-window annualized Sharpe (rf = 0) at each time step."""
    returns = to_1d_array(returns)
    sharpe_values = np.empty(len(returns), dtype=float)
    for index in range(len(returns)):
        window = returns[: index + 1]
        window_std = window.std(ddof=1)
        if window_std == 0 or np.isnan(window_std) or len(window) < 2:
            sharpe_values[index] = 0.0
        else:
            sharpe_values[index] = np.sqrt(252) * window.mean() / window_std
    return sharpe_values


def cumulative_and_annualized(returns) -> tuple[np.ndarray, np.ndarray]:
    """
    Full-path performance series:
      cumulative_performance[t] = prod(1+r)_0..t - 1
      annualized_performance[t] = (1 + cum[t])^(252/(t+1)) - 1
    """
    returns = to_1d_array(returns)
    if returns.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty

    cumulative_performance = np.cumprod(1.0 + returns) - 1.0
    periods = np.arange(1, len(returns) + 1, dtype=float)
    annualized_performance = np.power(1.0 + cumulative_performance, 252.0 / periods) - 1.0
    return cumulative_performance, annualized_performance


def summarize_performance(returns) -> dict[str, float]:
    """Scalar summary metrics for a daily return path (used by run_backtest)."""
    r = to_1d_array(returns)
    if r.size == 0:
        return {
            "days": 0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    cumulative, annualized = cumulative_and_annualized(r)
    equity = 1.0 + cumulative
    total_return = float(cumulative[-1])
    annual_return = float(annualized[-1])

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


def path_performance(returns) -> dict[str, np.ndarray]:
    """Return the expanding performance paths used for plotting / diagnostics."""
    r = to_1d_array(returns)
    cumulative, annualized = cumulative_and_annualized(r)
    return {
        "returns": r,
        "cumulative": cumulative,
        "annualized": annualized,
        "sharpe": compute_sharpe(r),
    }


def visualize_performance(returns, sharpe=None, annualized_performance=None):
    """Plot cumulative and annualized performance over time."""
    paths = path_performance(returns)
    if sharpe is None:
        sharpe = paths["sharpe"]
    if annualized_performance is None:
        annualized_performance = paths["annualized"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(paths["cumulative"], color="blue", linewidth=1.5)
    axes[0].set_title("Cumulative returns over time")
    axes[0].set_ylabel("Cumulative return")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(annualized_performance, color="green", linewidth=1.5, label="Ann. return")
    axes[1].plot(sharpe, color="orange", linewidth=1.2, alpha=0.85, label="Expanding Sharpe")
    axes[1].set_title("Annualized performance and expanding Sharpe")
    axes[1].set_ylabel("Value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()
    return fig, axes


# --- Legacy Monte Carlo baselines (optional; require long-format CRSP panel) ---

def _build_random_portfolio_returns(data, asset_cap=30, timesteps=100, seed=None):
    import pandas as pd

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

        chosen_assets = rng.choice(
            universe, size=min(asset_cap, len(universe)), replace=False
        )
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


def track_random_performance(data, asset_cap=30, iterations=50, timesteps=100, seed=42):
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
    cumulative_performance, annualized_performance = cumulative_and_annualized(
        average_returns
    )
    sharpe = compute_sharpe(average_returns)
    return cumulative_performance, annualized_performance, sharpe
