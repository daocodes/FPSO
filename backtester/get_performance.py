import numpy as np
import matplotlib.pyplot as plt
from algorithm.firefly import firefly

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

def track_performance(data, epsilon=0.01, iterations=50, timesteps=100):
    simulated_returns = []

    for _ in range(iterations):
        data_copy = data.copy()
        result = firefly(data_copy, epsilon=epsilon, timesteps=timesteps)
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