from enum import Enum, auto
from typing import Callable

import numpy as np
from numba import njit

WeightFuncType = Callable[[np.ndarray, np.ndarray, float, float], float]


class WeightMethod(Enum):
    UNWEIGHTED = auto()
    COSINE_SIMILARITY = auto()
    TIME_DIFF_COSINE_SIMILARITY = auto()
    EUCLIDEAN_DISTANCE = auto()
    TIME_DIFF_EUCLIDEAN_DISTANCE = auto()
    NORMALIZED_EUCLIDEAN_DISTANCE = auto()
    TIME_DIFF_NORMALIZED_EUCLIDEAN_DISTANCE = auto()


def get_weight_calculation_func(weight_calculation: WeightMethod = WeightMethod.COSINE_SIMILARITY) -> WeightFuncType:
    match weight_calculation:
        case WeightMethod.UNWEIGHTED:
            return unweighted
        case WeightMethod.COSINE_SIMILARITY:
            return cosine_similarity
        case WeightMethod.TIME_DIFF_COSINE_SIMILARITY:
            return time_diff_cosine_similarity
        case WeightMethod.EUCLIDEAN_DISTANCE:
            return euclidean_distance
        case WeightMethod.TIME_DIFF_EUCLIDEAN_DISTANCE:
            return time_diff_euclidean_distance
        case WeightMethod.NORMALIZED_EUCLIDEAN_DISTANCE:
            return normalized_euclidean_distance
        case WeightMethod.TIME_DIFF_NORMALIZED_EUCLIDEAN_DISTANCE:
            return time_diff_normalized_euclidean_distance
        case _:
            raise ValueError(f"Unknown weight calculation: {weight_calculation}")


@njit
def unweighted(_vector_a: np.ndarray, _vector_b: np.ndarray, _time_a: float, _time_b: float) -> float:
    return 1.0


@njit
def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray, _time_a: float, _time_b: float) -> float:
    magnitude = float(np.linalg.norm(vector_a)) * float(np.linalg.norm(vector_b))
    return np.dot(vector_a, vector_b) / magnitude


@njit
def time_diff_cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray, time_a: float, time_b: float) -> float:
    return cosine_similarity(vector_a, vector_b, time_a, time_b) / np.abs(time_b - time_a)


@njit
def euclidean_distance(vector_a: np.ndarray, vector_b: np.ndarray, _time_a: float, _time_b: float) -> float:
    return float(np.linalg.norm(vector_a - vector_b))


@njit
def time_diff_euclidean_distance(vector_a: np.ndarray, vector_b: np.ndarray, time_a: float, time_b: float) -> float:
    return euclidean_distance(vector_a, vector_b, time_a, time_b) / np.abs(time_b - time_a)


@njit
def normalized_euclidean_distance(vector_a: np.ndarray, vector_b: np.ndarray, _time_a: float, _time_b: float) -> float:
    diff_norm = float(np.linalg.norm(vector_a - vector_b))
    norm_sum = float(np.linalg.norm(vector_a)) + float(np.linalg.norm(vector_b))
    return diff_norm / norm_sum


@njit
def time_diff_normalized_euclidean_distance(
    vector_a: np.ndarray, vector_b: np.ndarray, time_a: float, time_b: float
) -> float:
    return normalized_euclidean_distance(vector_a, vector_b, time_a, time_b) / np.abs(time_b - time_a)
