import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import integers

from vector_vis_graph._ts2vg import horizontal_vvg_ts2vg, natural_vvg_ts2vg
from vector_vis_graph.vvg import horizontal_vvg, natural_vvg


@settings(deadline=10000, max_examples=20)
@given(integers(min_value=2, max_value=512), integers(min_value=1, max_value=512))
def test_natural_vvg_equivalence(time_length: int, vec_length: int) -> None:
    np.random.seed(0)
    X = np.random.randn(time_length, vec_length)
    adj = natural_vvg(X)
    adj_ts2vg = natural_vvg_ts2vg(X)
    assert np.allclose(adj, adj_ts2vg)


@settings(deadline=10000, max_examples=20)
@given(integers(min_value=2, max_value=512), integers(min_value=1, max_value=512))
def test_horizontal_vvg_equivalence(time_length: int, vec_length: int) -> None:
    np.random.seed(0)
    X = np.random.randn(time_length, vec_length)
    adj = horizontal_vvg(X)
    adj_ts2vg = horizontal_vvg_ts2vg(X)
    assert np.allclose(adj, adj_ts2vg)
