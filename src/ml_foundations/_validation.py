from __future__ import annotations

import numpy as np


def as_feature_matrix(values: object) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional feature matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("X must contain only finite values")
    return matrix


def as_target_vector(values: object, sample_count: int) -> np.ndarray:
    target = np.asarray(values)
    if target.ndim != 1 or target.shape[0] != sample_count:
        raise ValueError("y must be one-dimensional and match the number of samples")
    return target


def require_fitted(estimator: object, attribute: str) -> None:
    if not hasattr(estimator, attribute):
        raise RuntimeError("fit must be called before predict")

