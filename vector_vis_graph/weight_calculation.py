from enum import Enum, auto, unique
from typing import Callable

import numpy as np
from numba import njit

WeightFuncType = Callable[[np.ndarray, np.ndarray, float, float], float]


@unique
class WeightMethod(Enum):
    """
    Enumeration representing different weight calculation methods. The weight is calculated as a function of the
    two time instances, ``time_a`` :math:`(t_a)` and ``time_b`` :math:`(t_b)`, and the corresponding vectors in
    the multivariate time series ``vector_a`` :math:`(\\mathbf{x}_a)` and ``vector_b`` :math:`(\\mathbf{x}_b)`.
    Therefore, if nodes at time instances :math:`a` and :math:`b` are connected, then the weight of the edge
    between them is calculated as :math:`w_{a,b}`, with the following options:

    Attributes
    ----------
    COSINE_SIMILARITY :
        Cosine similarity between vectors.

            .. math:: w_{a, b} = \\frac{\\mathbf{x}_a \\cdot \\mathbf{x}_b}{\\lVert\\mathbf{x}_a\\rVert\\lVert\\mathbf{x}_b\\rVert}

    EUCLIDEAN_DISTANCE :
        Euclidean distance between vectors.

            .. math:: w_{a, b} = \\lVert\\mathbf{x}_b-\\mathbf{x}_a\\rVert

    NORMALIZED_EUCLIDEAN_DISTANCE :
        Normalized Euclidean distance between vectors.

            .. math:: w_{a, b} = \\frac{\\lVert\\mathbf{x}_b-\\mathbf{x}_a\\rVert}{\\lVert\\mathbf{x}_a\\rVert + \\lVert\\mathbf{x}_b\\rVert}

    TIME_DIFF_COSINE_SIMILARITY :
        Cosine similarity with time difference consideration.

            .. math:: w_{a, b} = \\frac{\\mathbf{x}_a \\cdot \\mathbf{x}_b}{\\lVert\\mathbf{x}_a\\rVert\\lVert\\mathbf{x}_b\\rVert\\lvert t_b-t_a\\rvert}

    TIME_DIFF_EUCLIDEAN_DISTANCE :
        Euclidean distance with time difference consideration.

            .. math:: w_{a, b} = \\frac{\\lVert\\mathbf{x}_b-\\mathbf{x}_a\\rVert}{\\lvert t_b-t_a\\rvert}

    TIME_DIFF_NORMALIZED_EUCLIDEAN_DISTANCE :
        Normalized Euclidean distance with time difference consideration.

            .. math:: w_{a, b} = \\frac{\\lVert\\mathbf{x}_b-\\mathbf{x}_a\\rVert}{\\left(\\lVert\\mathbf{x}_a\\rVert + \\lVert\\mathbf{x}_b\\rVert\\right)\\lvert t_b-t_a\\rvert}

    UNWEIGHTED :
        No specific weight calculation method, i.e., weight is always ``1.0``.

            .. math:: w_{a, b} = 1


    Examples
    --------
    >>> method = WeightMethod.COSINE_SIMILARITY
    >>> if method == WeightMethod.EUCLIDEAN_DISTANCE:
    ...     print("Using Euclidean distance for weight calculation.")
    """

    COSINE_SIMILARITY = auto()
    EUCLIDEAN_DISTANCE = auto()
    NORMALIZED_EUCLIDEAN_DISTANCE = auto()
    TIME_DIFF_COSINE_SIMILARITY = auto()
    TIME_DIFF_EUCLIDEAN_DISTANCE = auto()
    TIME_DIFF_NORMALIZED_EUCLIDEAN_DISTANCE = auto()
    UNWEIGHTED = auto()


def get_weight_calculation_func(weight_calculation: WeightMethod = WeightMethod.COSINE_SIMILARITY) -> WeightFuncType:
    """Given the weight calculation method in the form of ``WeightMethod``,
    returns the corresponding weight calculation function.

    Parameters
    ----------
    weight_calculation : WeightMethod, optional
        Desired calculation method as one of the options provided in ``WeightMethod``,
        by default ``WeightMethod.COSINE_SIMILARITY``.

    Returns
    -------
    WeightFuncType
        Returns the Nopython JIT Compiled weight calculation function that takes two
        vectors and their time instances and calculate the weight with function type
        ``Callable[[np.ndarray, np.ndarray, float, float], float]``.

    Raises
    ------
    ValueError
        If the ``weight_calculation`` is not one of the options provided in ``WeightMethod``.
    """
    match weight_calculation:
        case WeightMethod.UNWEIGHTED:
            return unweighted
        case WeightMethod.COSINE_SIMILARITY:
            return cosine_similarity
        case WeightMethod.TIME_DIFF_COSINE_SIMILARITY:
            return time_diff_cosine_similarity
        case WeightMethod.EUCLIDEAN_DISTANCE:
            return euclidean_distance
        case WeightMethod.TIME_DIFF_EUCLIDEAN_DISTANCE:
            return time_diff_euclidean_distance
        case WeightMethod.NORMALIZED_EUCLIDEAN_DISTANCE:
            return normalized_euclidean_distance
        case WeightMethod.TIME_DIFF_NORMALIZED_EUCLIDEAN_DISTANCE:
            return time_diff_normalized_euclidean_distance
        case _:
            raise ValueError(f"Unknown weight calculation: {weight_calculation}")


@njit
def unweighted(_vector_a: np.ndarray, _vector_b: np.ndarray, _time_a: float, _time_b: float) -> float:
    """Returns ``1.0``, where the parameters are not used, but are required for the function signature
    to match the other weight calculation functions
    ``WeightFuncType = Callable[[np.ndarray, np.ndarray, float, float], float]``.

    Parameters
    ----------
    vector_a : np.ndarray
        NOT USED vector of the first node, i.e., prior time instance.
    vector_b : np.ndarray
        NOT USED vector of the second node, i.e., latter time instance.
    time_a : float
        NOT USED value of the prior time instance.
    time_b : float
        NOT USED value of the latter time instance.

    Returns
    -------
    float
        Returns ``1.0``, i.e., the weight is always ``1.0``.
    """
    return 1.0


@njit
def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray, _time_a: float, _time_b: float) -> float:
    """Computes the cosine similarity between two vectors, by dividing their dot product by their magnitudes. Time
    instances are not used, but are required for the function signature to match the other weight calculation
    functions ``WeightFuncType = Callable[[np.ndarray, np.ndarray, float, float], float]``.

    Parameters
    ----------
    vector_a : np.ndarray
        The vector of the first node, i.e., prior time instance.
    vector_b : np.ndarray
        The vector of the second node, i.e., latter time instance.
    _time_a : float
        NOT USED value of the prior time instance.
    _time_b : float
        NOT USED value of the latter time instance.

    Returns
    -------
    float
        Cosine similarity between the two vectors.

    Notes
    -----
    For time indices :math:`a < b`, and input vectors ``vector_a`` and ``vector_b`` denoted by
    :math:`\\mathbf{x}_a` and :math:`\\mathbf{x}_b`, respecitvely. Then the function computes:
        .. math:: \\frac{\\mathbf{x}_a \\cdot \\mathbf{x}_b}{\\lVert\\mathbf{x}_a\\rVert\\lVert\\mathbf{x}_b\\rVert}
    """
    magnitude = float(np.linalg.norm(vector_a)) * float(np.linalg.norm(vector_b))
    return np.dot(vector_a, vector_b) / magnitude


@njit
def time_diff_cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray, time_a: float, time_b: float) -> float:
    """First computes the cosine similarity between two vectors, by dividing their dot product by their magnitude.
    Then, divides the result by the time difference between the two time instances.

    Parameters
    ----------
    vector_a : np.ndarray
        The vector of the first node, i.e., prior time instance.
    vector_b : np.ndarray
        The vector of the second node, i.e., latter time instance.
    time_a : float
        The value of the prior time instance.
    time_b : float
        The value of the latter time instance.

    Returns
    -------
    float
        Ratio of cosine similarity of the vectors and two time instances and time difference between the two time
        instances.

    Notes
    -----
    For input vectors ``vector_a`` and ``vector_b`` denoted by :math:`\\mathbf{x}_a` and :math:`\\mathbf{x}_b`,
    respecitvely, and the time instances ``time_a`` and ``time_b`` denoted by :math:`t_a` and :math:`t_b`.
    Then the function computes:
        .. math:: \\frac{\\mathbf{x}_a \\cdot \\mathbf{x}_b}{\\lVert\\mathbf{x}_a\\rVert\\lVert\\mathbf{x}_b\\rVert\\lvert t_b-t_a\\rvert}
    """
    return cosine_similarity(vector_a, vector_b, time_a, time_b) / np.abs(time_b - time_a)


@njit
def euclidean_distance(vector_a: np.ndarray, vector_b: np.ndarray, _time_a: float, _time_b: float) -> float:
    """Computes the Euclidean distance between two vectors, by calculating the norm of their difference. Time
    instances are not used, but are required for the function signature to match the other weight calculation
    functions ``WeightFuncType = Callable[[np.ndarray, np.ndarray, float, float], float]``.

    Parameters
    ----------
    vector_a : np.ndarray
        The vector of the first node, i.e., prior time instance.
    vector_b : np.ndarray
        The vector of the second node, i.e., latter time instance.
    _time_a : float
        NOT USED value of the prior time instance.
    _time_b : float
        NOT USED value of the latter time instance.

    Returns
    -------
    float
        Euclidean distance between the two vectors.

    Notes
    -----
    For time indices :math:`a < b`, and input vectors ``vector_a`` and ``vector_b`` denoted by
    :math:`\\mathbf{x}_a` and :math:`\\mathbf{x}_b`, respecitvely. Then the function computes:
        .. math:: \\lVert\\mathbf{x}_b - \\mathbf{x}_a\\rVert
    """
    return float(np.linalg.norm(vector_b - vector_a))


@njit
def time_diff_euclidean_distance(vector_a: np.ndarray, vector_b: np.ndarray, time_a: float, time_b: float) -> float:
    """First computes the Euclidean distance between two vectors, by calculating the norm of their difference.
    Then, divides the result by the time difference between the two time instances.

    Parameters
    ----------
    vector_a : np.ndarray
        The vector of the first node, i.e., prior time instance.
    vector_b : np.ndarray
        The vector of the second node, i.e., latter time instance.
    time_a : float
        The value of the prior time instance.
    time_b : float
        The value of the latter time instance.

    Returns
    -------
    float
        Ratio of Euclidean distance of the vectors and two time instances and time difference between the two time.

    Notes
    -----
    For time indices :math:`a < b`, and input vectors ``vector_a`` and ``vector_b`` denoted by
    :math:`\\mathbf{x}_a` and :math:`\\mathbf{x}_b`, respecitvely. Then the function computes:
        .. math:: \\frac{\\lVert\\mathbf{x}_b - \\mathbf{x}_a\\rVert}{\\lvert t_b-t_a\\rvert}
    """
    return euclidean_distance(vector_a, vector_b, time_a, time_b) / np.abs(time_b - time_a)


@njit
def normalized_euclidean_distance(vector_a: np.ndarray, vector_b: np.ndarray, _time_a: float, _time_b: float) -> float:
    """Computes the normalized Euclidean distance between two vectors, by calculating the ratio of their difference norm
    and sum their individual norms. Time instances are not used, but are required for the function signature to match
    the other weight calculation functions ``WeightFuncType = Callable[[np.ndarray, np.ndarray, float, float], float]``.

    Parameters
    ----------
    vector_a : np.ndarray
        The vector of the first node, i.e., prior time instance.
    vector_b : np.ndarray
        The vector of the second node, i.e., latter time instance.
    _time_a : float
        NOT USED value of the prior time instance.
    _time_b : float
        NOT USED value of the latter time instance.

    Returns
    -------
    float
        Normalized Euclidean distance between the two vectors.

    Notes
    -----
    For time indices :math:`a < b`, and input vectors ``vector_a`` and ``vector_b`` denoted by
    :math:`\\mathbf{x}_a` and :math:`\\mathbf{x}_b`, respecitvely. Then the function computes:
        .. math:: \\frac{\\lVert\\mathbf{x}_b - \\mathbf{x}_a\\rVert}{\\lVert\\mathbf{x}_a\\rVert + \\lVert\\mathbf{x}_b\\rVert}
    """
    norm_sum = float(np.linalg.norm(vector_a)) + float(np.linalg.norm(vector_b))
    return euclidean_distance(vector_a, vector_b, _time_a, _time_b) / norm_sum


@njit
def time_diff_normalized_euclidean_distance(
    vector_a: np.ndarray, vector_b: np.ndarray, time_a: float, time_b: float
) -> float:
    """First computes the normalized Euclidean distance between two vectors, by calculating the ratio of their
    difference norm and sum their individual norms. Then, divides the result by the time difference between
    the two time instances.

    Parameters
    ----------
    vector_a : np.ndarray
        The vector of the first node, i.e., prior time instance.
    vector_b : np.ndarray
        The vector of the second node, i.e., latter time instance.
    time_a : float
        The value of the prior time instance.
    time_b : float
        The value of the latter time instance.

    Returns
    -------
    float
        Ratio of normalized Euclidean distance between the two vectors and time difference between the two time.

    Notes
    -----
    For time indices :math:`a < b`, and input vectors ``vector_a`` and ``vector_b`` denoted by
    :math:`\\mathbf{x}_a` and :math:`\\mathbf{x}_b`, respecitvely. Then the function computes:
        .. math:: \\frac{\\lVert\\mathbf{x}_b - \\mathbf{x}_a\\rVert}{\\left(\\lVert\\mathbf{x}_a\\rVert + \\lVert\\mathbf{x}_b\\rVert\\right)\\lvert t_b-t_a\\rvert}
    """
    return normalized_euclidean_distance(vector_a, vector_b, time_a, time_b) / np.abs(time_b - time_a)
