from typing import Callable, Optional

import numpy as np
from ts2vg.graph._horizontal import _compute_graph as _compute_graph_horizontal
from ts2vg.graph._natural import _compute_graph as _compute_graph_natural

from vector_vis_graph.vvg import _ensure_vvg_input


def natural_vvg_ts2vg(
    multivariate: np.ndarray, timeline: Optional[np.ndarray] = None, directed: bool = False
) -> np.ndarray:
    adj = _vvg_ts2vg(_compute_graph_natural, multivariate, timeline)
    return adj if directed else adj + adj.T


def horizontal_vvg_ts2vg(
    multivariate: np.ndarray, timeline: Optional[np.ndarray] = None, directed: bool = False
) -> np.ndarray:
    adj = _vvg_ts2vg(_compute_graph_horizontal, multivariate, timeline)
    return adj if directed else adj + adj.T


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
