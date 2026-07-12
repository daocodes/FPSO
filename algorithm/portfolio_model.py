# This file will implement the mathematical environment, objectives, and penalization structures of the paper
import numpy as np


# This is a function of the weights
# lambda values will get tuned by the backtester
def calculate_portfolio_return(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    weights_prev: np.ndarray,
    lambda_v: float,
    lambda_t: float,
) -> float:
    # Ensure 1D vectors
    w = np.asarray(weights, dtype=float).reshape(-1)
    mu = np.asarray(expected_returns, dtype=float).reshape(-1)
    w_prev = np.asarray(weights_prev, dtype=float).reshape(-1)
    sigma = np.asarray(covariance, dtype=float)
    # Core terms from your equation
    expected_term = float(mu @ w)  # mu^T w
    risk_term = float(np.sqrt(np.clip(w @ sigma @ w, 0.0, None)))  # sqrt(w^T Sigma w)
    turnover_term = float(np.sum(np.abs(w - w_prev)))  # TO(w, w_prev), L1 turnover
    # R(w) = return - risk penalty - turnover penalty
    portfolio_return = expected_term - lambda_v * risk_term - lambda_t * turnover_term
    return portfolio_return


def evaluate_fitness(
    weights,
    expected_returns,
    covariance,
    weights_prev,
    lambda_v,
    lambda_t,
    rho_1,
    rho_2,
    rho_3,
    K,
    u,
):
    pass
