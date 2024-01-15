# Vector Visibility Graph

[![Build](https://github.com/tunakasif/vector-vis-graph/actions/workflows/build.yml/badge.svg)](https://github.com/tunakasif/vector-vis-graph/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/tunakasif/vector-vis-graph/graph/badge.svg?token=1RQ1RDMT9G)](https://codecov.io/gh/tunakasif/vector-vis-graph)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Numba](https://img.shields.io/badge/numba-%23013243.svg?logo=numba&logoColor=white)](https://numba.pydata.org/)

This repository contains the `Numba` JIT-compiled implementation of the _Vector Visibility Graphs (VVGs)_, which are a generalization of the Visibility Graphs (VGs) for multivariate time series. For a single time series, `ts2vg` ([GitHub](https://github.com/CarlosBergillos/ts2vg), [PyPI](https://pypi.org/project/ts2vg/)) provides a detailed and thorough construction of VGs with a `Cython`-based approach for performance. However, this implementation is not directly applicable to multivariate time series. Therefore, in this package, we implement the construction of VVGs for multivariate time series using a `Numba`-based approach for performance.

## Installation

### For Usage

This GitHub repository is `pip`/`poetry` installable. You can install the package using `pip` or `poetry` as follows:

#### Using `pip`

```sh
pip install git+https://github.com/tunakasif/vector-vis-graph.git
```

#### Using `poetry`

Either run the following command:

```sh
poetry add git+https://github.com/tunakasif/vector-vis-graph.git
```

or add the following to your `pyproject.toml` file:

```toml
[tool.poetry.dependencies]
vector-vis-graph = { git = "https://github.com/tunakasif/vector-vis-graph.git" }
```

## Usage

Given a multivariate time series, the `vector_vis_graph` package can be used to construct a vector visibility graph (VVG). The package provides two functions `natural_vvg()` and `horizontal_vvg()` with the same input types for constructing Natural and Horizontal VVGs. They take a multivariate time series where the rows correspond to the time steps and the columns correspond to the vector components such that for a multivariate time series, `mts`, `mts[i]` is the vector at time step `i`. The functions also take the following optional arguments:

- `timeline`: The timeline of the multivariate time series. If not provided, the timeline is assumed to be `[0, 1, 2, ...]`.
- `weight_method`: The method used to calculate the weight of the edges. The default is `WeightMethod.UNWEIGHTED`. There are a few other options available in the `WeightMethod` enum.
- `penetrable_limit`: The penetrable limit of the "visibility" of the vectors. For two vectors at different time steps to be visible to each other, the vectors at in-between time steps must satisfy certain conditions. The penetrable limit is the number of in-between time steps that can violate the conditions. The default is `0`.
- `directed`: Whether the graph is directed or undirected. The visibility of the vectors is calculated in a `left-to-right` directed manner. If `directed`, the calculated graph adjacency matrix is returned, else its sum with its transpose is returned. The default is `False`, so undirected.

```python
import numpy as np
from vector_vis_graph import WeightMethod, horizontal_vvg, natural_vvg

# Multivariate Time Series
TIME_LENGTH = 100
VEC_SIZE = 64
multivariate_ts = np.random.rand(TIME_LENGTH, VEC_SIZE)

# Natural Vector Visibility Graph with Default Parameters
# Timeline: [0, 1, 2, ...]
# Weight Method: Unweighted
# Penetrable Limit: 0
# Undirected Graph
nvvg_adj = natural_vvg(multivariate_ts)

# Horizontal Vector Visibility Graph with All Custom Parameters
# Timeline: [0, 2, 4, ...]
# Weight Method: Cosine Similarity
# Penetrable Limit: 2
# Directed Graph
hvvg_adj = horizontal_vvg(
    multivariate_ts,
    timeline=np.arange(0, 2 * TIME_LENGTH, 2), # [0, 2, 4, ...]
    weight_method=WeightMethod.COSINE_SIMILARITY,
    penetrable_limit=2,
    directed=True,
)
```
