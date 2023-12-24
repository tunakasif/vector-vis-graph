import numpy as np


def project_onto_vector(from_matrix: np.ndarray, onto_vector: np.ndarray) -> np.ndarray:
    return np.dot(from_matrix, onto_vector) / np.linalg.norm(onto_vector)


def project_onto_matrix(from_matrix: np.ndarray, onto_matrix: np.ndarray) -> np.ndarray:
    vectorized_project = np.vectorize(
        project_onto_vector,
        excluded="from_matrix",
        signature="(m,n),(n)->(m)",
    )
    return vectorized_project(from_matrix, onto_matrix)
