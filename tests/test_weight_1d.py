import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import integers
from ts2vg import NaturalVG

from tests import HYPOTHESIS_MAX_EXAMPLES, HYPOTHESIS_MAX_LENGTH
from vector_vis_graph.vvg import natural_vvg
from vector_vis_graph.weight_calculation import WeightMethod


@settings(deadline=None, max_examples=HYPOTHESIS_MAX_EXAMPLES)
@given(integers(min_value=2, max_value=HYPOTHESIS_MAX_LENGTH), integers(min_value=0, max_value=5))
def test_natural_vvg_1d(time_length: int, penetrable_limit: int) -> None:
    np.random.seed(0)

    X = np.random.rand(time_length)
    adj = natural_vvg(
        X, directed=True, weight_method=WeightMethod.TIME_DIFF_EUCLIDEAN_DISTANCE, penetrable_limit=penetrable_limit
    )

    vg = NaturalVG(directed="left_to_right", weighted="abs_slope", penetrable_limit=penetrable_limit)
    vg.build(X)
    ts2vg_adj = vg.adjacency_matrix(use_weights=True, no_weight_value=0)
    assert np.allclose(adj, ts2vg_adj)
