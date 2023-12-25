from typing import Any

import numpy as np
from hypothesis import given, settings  # noqa
from hypothesis.strategies import integers
from ts2vg import NaturalVG

from vector_vis_graph.vvg import natural_vvg, natural_vvg_dc


def compare_adj_with_ts2vg(adj: np.ndarray, X: np.ndarray, *args: Any, **kwargs: Any) -> None:
    vg = NaturalVG(*args, **kwargs)
    vg.build(
        X,
        xs=kwargs.get("xs", None),
        only_degrees=kwargs.get("only_degrees", False),
    )
    ts2vg_adj = vg.adjacency_matrix().astype(np.float64)
    assert np.allclose(adj, ts2vg_adj)


@settings(deadline=5000)
@given(integers(min_value=2, max_value=1024))
def test_natural_vvg_1d(time_length: int) -> None:
    np.random.seed(0)

    X = np.random.rand(time_length)
    adj = natural_vvg(X)

    vg = NaturalVG(directed="left_to_right")
    vg.build(X)
    ts2vg_adj = vg.adjacency_matrix().astype(np.float64)

    assert np.allclose(adj, ts2vg_adj)


@settings(deadline=5000)
@given(integers(min_value=2, max_value=1024))
def test_natural_vvg_dc_1d(time_length: int) -> None:
    np.random.seed(0)
    X = np.random.rand(time_length)
    adj = natural_vvg_dc(X)
    compare_adj_with_ts2vg(adj, X, directed="left_to_right")
