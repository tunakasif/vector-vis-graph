from typing import Optional

import numpy as np
from numba import njit

from vector_vis_graph.utils import project_onto_matrix


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
def _is_visible(
    curr_projection: np.ndarray,
    timeline: np.ndarray,
    a: int,
    b: int,
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
        return bool(np.all(lhs < rhs))
    else:
        return False


@njit(cache=True)
def _natural_vvg_loop(
    multivariate_tensor: np.ndarray,
    timeline: np.ndarray,
    projections: np.ndarray,
    weighted: bool,
) -> np.ndarray:
    time_length = timeline.shape[0]
    vvg_adjacency = np.zeros((time_length, time_length))
    for a in range(time_length - 1):
        curr_projection = projections[a]
        for b in range(a + 1, time_length):
            if _is_visible(curr_projection, timeline, a, b):
                vvg_adjacency[a, b] = calculate_weight(
                    multivariate_tensor[a],
                    multivariate_tensor[b],
                    timeline[a],
                    timeline[b],
                    weighted=weighted,
                )
    return vvg_adjacency


def _ensure_vvg_input(
    multivariate_tensor: np.ndarray,
    timeline: Optional[np.ndarray] = None,
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
) -> np.ndarray:
    multivariate_tensor, timeline = _ensure_vvg_input(multivariate_tensor, timeline)
    projections = project_onto_matrix(multivariate_tensor, multivariate_tensor)
    return _natural_vvg_loop(
        multivariate_tensor,
        timeline,
        projections,
        weighted,
    )
