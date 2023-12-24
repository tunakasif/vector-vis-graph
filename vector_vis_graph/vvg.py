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
def _natural_vvg_loop(
    multivariate_tensor: np.ndarray,
    timeline: np.ndarray,
    projections: np.ndarray,
    weighted: bool,
) -> np.ndarray:
    time_length = timeline.shape[0]
    vvg_adjacency = np.zeros((time_length, time_length))
    for a in range(time_length - 1):
        t_a = timeline[a]
        x_aa = projections[a, a]

        for b in range(a + 1, time_length):
            x_ab = projections[a, b]
            t_b = timeline[b]

            x_acs = projections[a, a + 1 : b]
            t_cs = timeline[a + 1 : b]

            lhs = np.divide(x_acs - x_ab, t_b - t_cs)
            rhs = np.divide(x_aa - x_ab, t_b - t_a)

            if np.all(lhs < rhs):
                vvg_adjacency[a, b] = calculate_weight(
                    multivariate_tensor[a],
                    multivariate_tensor[b],
                    t_a,
                    t_b,
                    weighted=weighted,
                )

    return vvg_adjacency


def natural_vvg(
    multivariate_tensor: np.ndarray,
    *,
    timeline: Optional[np.ndarray] = None,
    weighted: bool = False,
) -> np.ndarray:
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
    projections = project_onto_matrix(multivariate_tensor, multivariate_tensor)

    return _natural_vvg_loop(
        multivariate_tensor,
        timeline,
        projections,
        weighted,
    )
