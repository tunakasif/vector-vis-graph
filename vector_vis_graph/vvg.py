from typing import Callable, Optional

import numpy as np
from numba import njit, prange
from ts2vg.graph._horizontal import _compute_graph as _compute_graph_horizontal
from ts2vg.graph._natural import _compute_graph as _compute_graph_natural

VisibilityFuncType = Callable[[np.ndarray, np.ndarray, int, int, int], bool]
WeigthFuncType = Callable[[np.ndarray, np.ndarray, float, float, bool], float]


@njit(cache=True)
def calculate_weight(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    time_a: float,
    time_b: float,
    weighted: bool = False,
) -> float:
    if weighted:
        magnitude = float(np.linalg.norm(vector_a)) * float(np.linalg.norm(vector_b))
        inner = np.dot(vector_a, vector_b) / magnitude
        return inner / np.abs(time_b - time_a)
    else:
        return 1.0


@njit(cache=True)
def _is_visible_natural(
    curr_projection: np.ndarray, timeline: np.ndarray, a: int, b: int, penetrable_limit: int = 0
) -> bool:
    if a < b:
        x_aa = curr_projection[a]
        x_ab = curr_projection[b]
        t_a = timeline[a]
        t_b = timeline[b]

        x_acs = curr_projection[a + 1 : b]
        t_cs = timeline[a + 1 : b]

        lhs = np.divide(x_acs - x_ab, t_b - t_cs)
        rhs = (x_aa - x_ab) / (t_b - t_a)
        return np.sum(lhs >= rhs) <= penetrable_limit
    else:
        return False


@njit(cache=True)
def _is_visible_horizontal(
    curr_projection: np.ndarray, _timeline: np.ndarray, a: int, b: int, penetrable_limit: int = 0
) -> bool:
    if a < b:
        first = curr_projection[a]
        last = curr_projection[b]
        middle = curr_projection[a + 1 : b]
        return np.sum(middle >= min(first, last)) <= penetrable_limit
    else:
        return False


@njit(cache=True, parallel=True)
def _vvg_loop(
    multivariate_tensor: np.ndarray,
    timeline: np.ndarray,
    weighted: bool,
    visibility_func: VisibilityFuncType,
    weight_func: WeigthFuncType,
    penetrable_limit: int = 0,
) -> np.ndarray:
    projections = np.dot(multivariate_tensor, multivariate_tensor.T)
    time_length = timeline.shape[0]
    vvg_adjacency = np.zeros((time_length, time_length))
    for a in prange(time_length - 1):
        curr_projection = projections[a]
        for b in prange(a + 1, time_length):
            if visibility_func(curr_projection, timeline, a, b, penetrable_limit):
                vvg_adjacency[a, b] = weight_func(
                    multivariate_tensor[a],
                    multivariate_tensor[b],
                    timeline[a],
                    timeline[b],
                    weighted,
                )
    return vvg_adjacency


def _ensure_vvg_input(
    multivariate_tensor: np.ndarray, timeline: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    if timeline is None:
        timeline = np.arange(multivariate_tensor.shape[0])
    elif len(timeline.shape) != 1:
        raise ValueError(f"timeline must be a 1D tensor, got {timeline.shape}")
    elif multivariate_tensor.shape[0] != timeline.shape[0]:
        raise ValueError(
            "multivariate_tensor and timeline must have the same length, "
            f"got {multivariate_tensor.shape[0]} and {timeline.shape[0]}"
        )

    if multivariate_tensor.ndim == 1:
        multivariate_tensor = multivariate_tensor.reshape(-1, 1)
    return multivariate_tensor, timeline


def natural_vvg(
    multivariate_tensor: np.ndarray,
    *,
    timeline: Optional[np.ndarray] = None,
    weighted: bool = False,
    penetrable_limit: int = 0,
) -> np.ndarray:
    multivariate_tensor, timeline = _ensure_vvg_input(multivariate_tensor, timeline)
    return _vvg_loop(
        multivariate_tensor,
        timeline,
        weighted,
        _is_visible_natural,
        calculate_weight,
        penetrable_limit,
    )


def horizontal_vvg(
    multivariate_tensor: np.ndarray,
    *,
    timeline: Optional[np.ndarray] = None,
    weighted: bool = False,
    penetrable_limit: int = 0,
) -> np.ndarray:
    multivariate_tensor, timeline = _ensure_vvg_input(multivariate_tensor, timeline)
    return _vvg_loop(
        multivariate_tensor,
        timeline,
        weighted,
        _is_visible_horizontal,
        calculate_weight,
        penetrable_limit,
    )


def natural_vvg_ts2vg(
    multivariate_tensor: np.ndarray,
    timeline: Optional[np.ndarray] = None,
) -> np.ndarray:
    multivariate_tensor, timeline = _ensure_vvg_input(multivariate_tensor, timeline)
    projections = np.dot(multivariate_tensor, multivariate_tensor.T)
    N = projections.shape[0]
    adj = np.array(
        [np.pad(_adj_cg_first(projections[i, i:], _compute_graph_natural), (i, 0)) for i in range(N - 1)]
    )
    adj = np.vstack([adj, np.zeros(N)])
    return adj


def horizontal_vvg_ts2vg(
    multivariate_tensor: np.ndarray,
    timeline: Optional[np.ndarray] = None,
) -> np.ndarray:
    multivariate_tensor, timeline = _ensure_vvg_input(multivariate_tensor, timeline)
    projections = np.dot(multivariate_tensor, multivariate_tensor.T)
    N = projections.shape[0]
    adj = np.array(
        [np.pad(_adj_cg_first(projections[i, i:], _compute_graph_horizontal), (i, 0)) for i in range(N - 1)]
    )
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
