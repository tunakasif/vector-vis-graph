from typing import Callable, Optional

import numpy as np
from numba import njit, prange
from ts2vg.graph._horizontal import _compute_graph as _compute_graph_horizontal
from ts2vg.graph._natural import _compute_graph as _compute_graph_natural

from vector_vis_graph.weight_calculation import time_diff_inner_cosine

VisibilityFuncType = Callable[[np.ndarray, np.ndarray, int, int, int], bool]
WeigthFuncType = Callable[[np.ndarray, np.ndarray, float, float, bool], float]


def natural_vvg(
    multivariate: np.ndarray,
    *,
    timeline: Optional[np.ndarray] = None,
    weighted: bool = False,
    penetrable_limit: int = 0,
) -> np.ndarray:
    multivariate, timeline = _ensure_vvg_input(multivariate, timeline)
    return _vvg_loop(multivariate, timeline, weighted, _is_visible_natural, time_diff_inner_cosine, penetrable_limit)


def horizontal_vvg(
    multivariate: np.ndarray,
    *,
    timeline: Optional[np.ndarray] = None,
    weighted: bool = False,
    penetrable_limit: int = 0,
) -> np.ndarray:
    multivariate, timeline = _ensure_vvg_input(multivariate, timeline)
    return _vvg_loop(multivariate, timeline, weighted, _is_visible_horizontal, time_diff_inner_cosine, penetrable_limit)


def natural_vvg_ts2vg(multivariate: np.ndarray, timeline: Optional[np.ndarray] = None) -> np.ndarray:
    return _vvg_ts2vg(_compute_graph_natural, multivariate, timeline)


def horizontal_vvg_ts2vg(multivariate: np.ndarray, timeline: Optional[np.ndarray] = None) -> np.ndarray:
    return _vvg_ts2vg(_compute_graph_horizontal, multivariate, timeline)


def _vvg_ts2vg(
    compute_graph_func: Callable, multivariate: np.ndarray, timeline: Optional[np.ndarray] = None
) -> np.ndarray:
    multivariate, timeline = _ensure_vvg_input(multivariate, timeline)
    projections = np.dot(multivariate, multivariate.T)
    N = projections.shape[0]
    adj = np.array([np.pad(_adj_cg_first(projections[i, i:], compute_graph_func), (i, 0)) for i in range(N - 1)])
    adj = np.vstack([adj, np.zeros(N)])
    return adj


def _adj_cg_first(ts: np.ndarray, compute_graph_func: Callable) -> np.ndarray:
    xs = np.arange(len(ts), dtype=np.float64)
    edges, *_ = compute_graph_func(ts, xs, 1, 0, False, float("-inf"), float("inf"))
    edges_array = np.asarray(edges)
    idx = edges_array[:, 0] == 0
    first_row_adj = np.zeros(len(ts))
    first_row_adj[edges_array[idx, 1]] = 1
    return first_row_adj


@njit(cache=True)
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


@njit(cache=True)
def _is_visible_horizontal(
    curr_projection: np.ndarray, _timeline: np.ndarray, i: int, j: int, penetrable_limit: int = 0
) -> bool:
    if i < j:
        first, middle, last = curr_projection[i], curr_projection[i + 1 : j], curr_projection[j]
        return np.sum(middle >= min(first, last)) <= penetrable_limit
    else:
        return False


@njit(cache=True, parallel=True)
def _vvg_loop(
    multivariate: np.ndarray,
    timeline: np.ndarray,
    weighted: bool,
    visibility_func: VisibilityFuncType,
    weight_func: WeigthFuncType,
    penetrable_limit: int = 0,
) -> np.ndarray:
    projections = np.dot(multivariate, multivariate.T)
    time_length = timeline.shape[0]
    vvg_adjacency = np.zeros((time_length, time_length))
    for i in prange(time_length - 1):
        curr_projection = projections[i]
        for j in prange(i + 1, time_length):
            if visibility_func(curr_projection, timeline, i, j, penetrable_limit):
                vvg_adjacency[i, j] = weight_func(multivariate[i], multivariate[j], timeline[i], timeline[j], weighted)
    return vvg_adjacency


def _ensure_vvg_input(multivariate: np.ndarray, timeline: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    if timeline is None:
        timeline = np.arange(multivariate.shape[0])
    elif timeline.ndim != 1:
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
