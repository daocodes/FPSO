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
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    weights_prev: np.ndarray,
    lambda_v: float,
    lambda_t: float,
    rho_1: float,
    rho_2: float,
    rho_3: float,
    K: int,
    u,
) -> float:
    # Base return-risk-turnover objective R(w)
    w = np.asarray(weights, dtype=float).reshape(-1)
    R_w = calculate_portfolio_return(
        w, expected_returns, covariance, weights_prev, lambda_v, lambda_t
    )

    # Constraint 1: budget (sum of weights equals 1)
    budget_penalty = abs(np.sum(w) - 1.0)

    # Constraint 2: box bounds [l_i, u_i]
    # We use long-only lower bounds l_i = 0 by default.
    lower_bounds = np.zeros_like(w)
    upper_bounds = np.asarray(u, dtype=float)
    if upper_bounds.ndim == 0:
        upper_bounds = np.full_like(w, float(upper_bounds))
    else:
        upper_bounds = upper_bounds.reshape(-1)

    if upper_bounds.shape != w.shape:
        raise ValueError("Upper bounds u must be scalar or same length as weights.")

    lower_viol = np.maximum(0.0, lower_bounds - w) ** 2
    upper_viol = np.maximum(0.0, w - upper_bounds) ** 2
    box_penalty = float(np.sum(lower_viol + upper_viol))

    # Constraint 3: cardinality (number of active positions equals K)
    active_positions = int(np.sum(np.abs(w) > 1e-8))
    cardinality_penalty = float((active_positions - K) ** 2)

    # Phi(w) = R(w) - weighted penalties
    fitness = (
        R_w
        - rho_1 * budget_penalty
        - rho_2 * box_penalty
        - rho_3 * cardinality_penalty
    )
    return float(fitness)
