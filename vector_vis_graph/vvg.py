from typing import Callable, Optional

import numpy as np
from numba import njit, prange

from vector_vis_graph.weight_calculation import WeightFuncType, WeightMethod, get_weight_calculation_func

VisibilityFuncType = Callable[[np.ndarray, np.ndarray, int, int, int], bool]


def natural_vvg(
    multivariate: np.ndarray,
    *,
    timeline: Optional[np.ndarray] = None,
    weight_method: WeightMethod = WeightMethod.UNWEIGHTED,
    penetrable_limit: int = 0,
    directed: bool = False,
) -> np.ndarray:
    """Generate a vector visibility graph (VVG) from a multivariate time series using the natural visibility,
    and returns the adjacency matrix of the VVG as 2D ``np.ndarray``. It also irregularly sampled time series,
    provided through ``timeline``. By default, the VVG is unweighted, more weight options are available through
    ``WeightMethod``. The ``penetrable_limit`` is the number of allowances to disagree with visibility conditions.

    Parameters
    ----------
    multivariate : np.ndarray
        1D or 2D array of multivariate time series.
    timeline : Optional[np.ndarray], optional
        Time indices of `multivariate`. If not provided, `[0, 1, 2, ...]` is used obtained
        through `np.arange(multivariate.shape[0])`, by default None.
    weight_method : WeightMethod, optional
        By default the adjacency is unweighted ``{0, 1}``, other options are in ``WeightMethod``, by default
        ``WeightMethod.UNWEIGHTED``.
    penetrable_limit : int, optional
        Number of allowances to disagree with visibility conditions, e.g., still allows "visibility" even if some
        in-between points do not satisfy the visibility condition. This is useful for noisy data, by default 0.
    directed : bool, optional
        The adjacency matrix is generated in a ``left-to-right`` directed format, although by default undirected
        version is returned by ``A + A.T`` operation, by default False (undirected).

    Returns
    -------
    np.ndarray
        The adjacency matrix of the vector visibility graph (VVG) as 2D ``np.ndarray``.

    Examples
    --------
    >>> import numpy as np
    >>> from vector_vis_graph import WeightMethod, natural_vvg

    >>> TIME_LENGTH, VEC_SIZE = 100, 64
    >>> multivariate_ts = np.random.rand(TIME_LENGTH, VEC_SIZE)

    >>> nvvg_adj = natural_vvg(multivariate_ts)
    >>> nvvg_adj = natural_vvg(multivariate_ts, timeline=np.arange(0, 2 * TIME_LENGTH, 2), weight_method=WeightMethod.COSINE_SIMILARITY, penetrable_limit=2, directed=True)
    """
    multivariate, timeline = _ensure_vvg_input(multivariate, timeline)
    weight_func = get_weight_calculation_func(weight_method)
    adj = _vvg_loop(multivariate, timeline, _is_visible_natural, weight_func, penetrable_limit)
    return adj if directed else adj + adj.T


def horizontal_vvg(
    multivariate: np.ndarray,
    *,
    timeline: Optional[np.ndarray] = None,
    weight_method: WeightMethod = WeightMethod.UNWEIGHTED,
    penetrable_limit: int = 0,
    directed: bool = False,
) -> np.ndarray:
    """Generate a vector visibility graph (VVG) from a multivariate time series using the horizontal visibility,
    and returns the adjacency matrix of the VVG as 2D ``np.ndarray``. It also irregularly sampled time series,
    provided through ``timeline``. By default, the VVG is unweighted, more weight options are available through
    ``WeightMethod``. The ``penetrable_limit`` is the number of allowances to disagree with visibility conditions.

    Parameters
    ----------
    multivariate : np.ndarray
        1D or 2D array of multivariate time series.
    timeline : Optional[np.ndarray], optional
        Time indices of `multivariate`. If not provided, `[0, 1, 2, ...]` is used obtained
        through `np.arange(multivariate.shape[0])`, by default None.
    weight_method : WeightMethod, optional
        By default the adjacency is unweighted ``{0, 1}``, other options are in ``WeightMethod``, by default
        ``WeightMethod.UNWEIGHTED``.
    penetrable_limit : int, optional
        Number of allowances to disagree with visibility conditions, e.g., still allows "visibility" even if some
        in-between points do not satisfy the visibility condition. This is useful for noisy data, by default 0.
    directed : bool, optional
        The adjacency matrix is generated in a ``left-to-right`` directed format, although by default undirected
        version is returned by ``A + A.T`` operation, by default False (undirected).

    Returns
    -------
    np.ndarray
        The adjacency matrix of the vector visibility graph (VVG) as 2D ``np.ndarray``.

    Examples
    --------
    >>> import numpy as np
    >>> from vector_vis_graph import WeightMethod, horizontal_vvg

    >>> TIME_LENGTH, VEC_SIZE = 100, 64
    >>> multivariate_ts = np.random.rand(TIME_LENGTH, VEC_SIZE)

    >>> hvvg_adj = horizontal_vvg(multivariate_ts)
    >>> hvvg_adj = horizontal_vvg(multivariate_ts, timeline=np.arange(0, 2 * TIME_LENGTH, 2), weight_method=WeightMethod.COSINE_SIMILARITY, penetrable_limit=2, directed=True)
    """
    multivariate, timeline = _ensure_vvg_input(multivariate, timeline)
    weight_func = get_weight_calculation_func(weight_method)
    adj = _vvg_loop(multivariate, timeline, _is_visible_horizontal, weight_func, penetrable_limit)
    return adj if directed else adj + adj.T


@njit
def _is_visible_natural(
    curr_projection: np.ndarray,
    timeline: np.ndarray,
    i: int,
    j: int,
    penetrable_limit: int = 0,
) -> bool:
    if i < j:
        first_value, middle_values, last_value = (
            curr_projection[i],
            curr_projection[i + 1 : j],
            curr_projection[j],
        )
        first_time, middle_times, last_time = (
            timeline[i],
            timeline[i + 1 : j],
            timeline[j],
        )

        lhs = np.divide(middle_values - last_value, last_time - middle_times)
        rhs = (first_value - last_value) / (last_time - first_time)
        return np.sum(lhs >= rhs) <= penetrable_limit
    else:
        return False


@njit
def _is_visible_horizontal(
    curr_projection: np.ndarray,
    _timeline: np.ndarray,
    i: int,
    j: int,
    penetrable_limit: int = 0,
) -> bool:
    if i < j:
        first, middle, last = (
            curr_projection[i],
            curr_projection[i + 1 : j],
            curr_projection[j],
        )
        return np.sum(middle >= min(first, last)) <= penetrable_limit
    else:
        return False


@njit
def _unitarize(matrix: np.ndarray) -> np.ndarray:
    """Row normalize a matrix by dividing each row by its `l2`-norm. The `np.linalg.norm()`
    does not support `kwargs` like `axis=1` or `keepdims=True` for `njit` compilation. Therefore,
    implemented through `np.sqrt()`, `np.square()` and `np.sum()`.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix.

    Returns
    -------
    np.ndarray
        Row normalized matrix.
    """
    norms = np.sqrt(np.square(matrix).sum(axis=-1))  # njit does not support `keepdims`
    return matrix / norms.reshape(-1, 1)


@njit(parallel=True)
def _vvg_loop(
    multivariate: np.ndarray,
    timeline: np.ndarray,
    visibility_func: VisibilityFuncType,
    weight_func: WeightFuncType,
    penetrable_limit: int = 0,
) -> np.ndarray:
    projections = np.dot(_unitarize(multivariate), multivariate.T)
    time_length = timeline.shape[0]
    vvg_adjacency = np.zeros((time_length, time_length))
    for i in prange(time_length - 1):
        curr_projection = projections[i]
        for j in prange(i + 1, time_length):
            if visibility_func(curr_projection, timeline, i, j, penetrable_limit):
                vvg_adjacency[i, j] = weight_func(multivariate[i], multivariate[j], timeline[i], timeline[j])
    return vvg_adjacency


def _ensure_vvg_input(multivariate: np.ndarray, timeline: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Ensures that the input `multivariate` and `timeline` are valid for the VVG algorithm.

    Parameters
    ----------
    multivariate : np.ndarray
        1D or 2D array of multivariate time series.
    timeline : Optional[np.ndarray], optional
        Time indices of `multivariate`. If not provided, `[0, 1, 2, ...]` is used obtained
        through `np.arange(multivariate.shape[0])`, by default None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Suitable `multivariate` and `timeline` for the VVG algorithm.

    Raises
    ------
    ValueError
        If dimensions of `multivariate` and `timeline` are not valid.

        - `timeline` must be a 1D array.
        - `multivariate` must be a 1D or 2D array.
        - `multivariate` and `timeline` must have the same length.
    """
    if timeline is None:
        timeline = np.arange(multivariate.shape[0])

    if timeline.ndim != 1:
        raise ValueError(f"timeline must be a 1D array, got {timeline.shape}")
    elif multivariate.ndim < 1 or multivariate.ndim > 2:
        raise ValueError(f"multivariate must be a 1D or 2D array, got {multivariate.shape}")
    elif multivariate.shape[0] != timeline.shape[0]:
        raise ValueError(
            "multivariate and timeline must have the same length, "
            f"got multivariate ({multivariate.shape[0]}) and timeline ({timeline.shape[0]})"
        )

    if multivariate.ndim == 1:
        multivariate = multivariate.reshape(-1, 1)
    return multivariate, timeline
