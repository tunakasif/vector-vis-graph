import numpy as np
from numba import njit


@njit(cache=True)
def inner_cosine(
    vector_a: np.ndarray, vector_b: np.ndarray, _time_a: float, _time_b: float, weighted: bool = False
) -> float:
    if weighted:
        magnitude = float(np.linalg.norm(vector_a)) * float(np.linalg.norm(vector_b))
        return np.dot(vector_a, vector_b) / magnitude
    else:
        return 1.0


@njit(cache=True)
def time_diff_inner_cosine(
    vector_a: np.ndarray, vector_b: np.ndarray, time_a: float, time_b: float, weighted: bool = False
) -> float:
    return inner_cosine(vector_a, vector_b, time_a, time_b) / np.abs(time_b - time_a) if weighted else 1.0
