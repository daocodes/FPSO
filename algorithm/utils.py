import numpy as np


def calculate_euclidean_distance(vector_a, vector_b) -> float:
    """
    Compute the L2 distance between two allocation vectors.
    """
    a = np.asarray(vector_a, dtype=float).reshape(-1)
    b = np.asarray(vector_b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("vector_a and vector_b must have the same shape.")
    return float(np.linalg.norm(a - b, ord=2))


def generate_naive_equal_weight(num_assets: int) -> np.ndarray:
    """
    Build a uniform 1/N portfolio weight vector.
    """
    n = int(num_assets)
    if n <= 0:
        raise ValueError("num_assets must be a positive integer.")
    return np.full(n, 1.0 / n, dtype=float)
