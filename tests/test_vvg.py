import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import integers

from tests import HYPOTHESIS_MAX_EXAMPLES, HYPOTHESIS_MAX_LENGTH
from vector_vis_graph._ts2vg import horizontal_vvg_ts2vg, natural_vvg_ts2vg
from vector_vis_graph.vvg import horizontal_vvg, natural_vvg


@settings(deadline=None, max_examples=HYPOTHESIS_MAX_EXAMPLES)
@given(integers(min_value=2, max_value=HYPOTHESIS_MAX_LENGTH), integers(min_value=2, max_value=HYPOTHESIS_MAX_LENGTH))
def test_natural_vvg_equivalence(time_length: int, vec_length: int) -> None:
    np.random.seed(0)
    X = np.random.randn(time_length, vec_length)
    adj = natural_vvg(X)
    adj_ts2vg = natural_vvg_ts2vg(X)
    assert np.allclose(adj, adj_ts2vg)


@settings(deadline=None, max_examples=HYPOTHESIS_MAX_EXAMPLES)
@given(integers(min_value=2, max_value=HYPOTHESIS_MAX_LENGTH), integers(min_value=2, max_value=HYPOTHESIS_MAX_LENGTH))
def test_horizontal_vvg_equivalence(time_length: int, vec_length: int) -> None:
    np.random.seed(0)
    X = np.random.randn(time_length, vec_length)
    adj = horizontal_vvg(X)
    adj_ts2vg = horizontal_vvg_ts2vg(X)
    assert np.allclose(adj, adj_ts2vg)
