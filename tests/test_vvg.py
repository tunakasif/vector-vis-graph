import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import integers
from ts2vg import HorizontalVG, NaturalVG

from vector_vis_graph._ts2vg import horizontal_vvg_ts2vg, natural_vvg_ts2vg
from vector_vis_graph.vvg import horizontal_vvg, natural_vvg


@settings(deadline=10000, max_examples=20)
@given(integers(min_value=2, max_value=1024), integers(min_value=0, max_value=5))
def test_natural_vvg_1d(time_length: int, penetrable_limit: int) -> None:
    np.random.seed(0)

    X = np.random.rand(time_length)
    adj = natural_vvg(X, penetrable_limit=penetrable_limit)

    vg = NaturalVG(directed="left_to_right", penetrable_limit=penetrable_limit)
    vg.build(X)
    ts2vg_adj = vg.adjacency_matrix().astype(np.float64)
    assert np.allclose(adj, ts2vg_adj)


@settings(deadline=10000, max_examples=20)
@given(integers(min_value=2, max_value=1024), integers(min_value=0, max_value=5))
def test_horizontal_vvg_1d(time_length: int, penetrable_limit: int) -> None:
    np.random.seed(0)

    X = np.random.rand(time_length)
    adj = horizontal_vvg(X, penetrable_limit=penetrable_limit)

    vg = HorizontalVG(directed="left_to_right", penetrable_limit=penetrable_limit)
    vg.build(X)
    ts2vg_adj = vg.adjacency_matrix().astype(np.float64)
    assert np.allclose(adj, ts2vg_adj)


@settings(deadline=10000, max_examples=20)
@given(integers(min_value=2, max_value=1024), integers(min_value=1, max_value=1024))
def test_natural_vvg_equivalence(time_length: int, vec_length: int) -> None:
    np.random.seed(0)
    X = np.random.rand(time_length, vec_length)
    adj = natural_vvg(X)
    adj_ts2vg = natural_vvg_ts2vg(X)
    assert np.allclose(adj, adj_ts2vg)


@settings(deadline=10000, max_examples=20)
@given(integers(min_value=2, max_value=1024), integers(min_value=1, max_value=1024))
def test_horizontal_vvg_equivalence(time_length: int, vec_length: int) -> None:
    np.random.seed(0)
    X = np.random.rand(time_length, vec_length)
    adj = horizontal_vvg(X)
    adj_ts2vg = horizontal_vvg_ts2vg(X)
    assert np.allclose(adj, adj_ts2vg)
