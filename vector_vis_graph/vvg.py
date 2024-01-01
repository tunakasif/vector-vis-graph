from typing import Callable, Optional

import numpy as np
from numba import njit, prange

from vector_vis_graph.weight_calculation import WeightFuncType, WeightMethod, get_weight_calculation_func

VisibilityFuncType = Callable[[np.ndarray, np.ndarray, int, int, int], bool]


def natural_vvg(
    multivariate: np.ndarray,
    *,
    timeline: Optional[np.ndarray] = None,
    weight_method: WeightMethod = WeightMethod.UNWEIGHTED,
    penetrable_limit: int = 0,
    directed: bool = False,
) -> np.ndarray:
    multivariate, timeline = _ensure_vvg_input(multivariate, timeline)
    weight_func = get_weight_calculation_func(weight_method)
    adj = _vvg_loop(multivariate, timeline, _is_visible_natural, weight_func, penetrable_limit)
    return adj if directed else adj + adj.T


def horizontal_vvg(
    multivariate: np.ndarray,
    *,
    timeline: Optional[np.ndarray] = None,
    weight_method: WeightMethod = WeightMethod.UNWEIGHTED,
    penetrable_limit: int = 0,
    directed: bool = False,
) -> np.ndarray:
    multivariate, timeline = _ensure_vvg_input(multivariate, timeline)
    weight_func = get_weight_calculation_func(weight_method)
    adj = _vvg_loop(multivariate, timeline, _is_visible_horizontal, weight_func, penetrable_limit)
    return adj if directed else adj + adj.T


@njit
def _is_visible_natural(
    curr_projection: np.ndarray, timeline: np.ndarray, i: int, j: int, penetrable_limit: int = 0
) -> bool:
    if i < j:
        first_value, middle_values, last_value = curr_projection[i], curr_projection[i + 1 : j], curr_projection[j]
        first_time, middle_times, last_time = timeline[i], timeline[i + 1 : j], timeline[j]

        lhs = np.divide(middle_values - last_value, last_time - middle_times)
        rhs = (first_value - last_value) / (last_time - first_time)
        return np.sum(lhs >= rhs) <= penetrable_limit
    else:
        return False


@njit
def _is_visible_horizontal(
    curr_projection: np.ndarray, _timeline: np.ndarray, i: int, j: int, penetrable_limit: int = 0
) -> bool:
    if i < j:
        first, middle, last = curr_projection[i], curr_projection[i + 1 : j], curr_projection[j]
        return np.sum(middle >= min(first, last)) <= penetrable_limit
    else:
        return False


@njit
def _unitarize(matrix: np.ndarray) -> np.ndarray:
    norms = np.sqrt(np.square(matrix).sum(axis=-1))  # njit does not support `keepdims`
    return matrix / norms.reshape(-1, 1)


@njit(parallel=True)
def _vvg_loop(
    multivariate: np.ndarray,
    timeline: np.ndarray,
    visibility_func: VisibilityFuncType,
    weight_func: WeightFuncType,
    penetrable_limit: int = 0,
) -> np.ndarray:
    projections = np.dot(_unitarize(multivariate), multivariate.T)
    time_length = timeline.shape[0]
    vvg_adjacency = np.zeros((time_length, time_length))
    for i in prange(time_length - 1):
        curr_projection = projections[i]
        for j in prange(i + 1, time_length):
            if visibility_func(curr_projection, timeline, i, j, penetrable_limit):
                vvg_adjacency[i, j] = weight_func(multivariate[i], multivariate[j], timeline[i], timeline[j])
    return vvg_adjacency


def _ensure_vvg_input(multivariate: np.ndarray, timeline: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    if timeline is None:
        timeline = np.arange(multivariate.shape[0])

    if timeline.ndim != 1:
        raise ValueError(f"timeline must be a 1D array, got {timeline.shape}")
    elif multivariate.ndim < 1 or multivariate.ndim > 2:
        raise ValueError(f"multivariate must be a 1D or 2D array, got {multivariate.shape}")
    elif multivariate.shape[0] != timeline.shape[0]:
        raise ValueError(
            "multivariate and timeline must have the same length, "
            f"got multivariate ({multivariate.shape[0]}) and timeline ({timeline.shape[0]})"
        )

    if multivariate.ndim == 1:
        multivariate = multivariate.reshape(-1, 1)
    return multivariate, timeline
