"""Covariance estimation and inversion utilities."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

CovarianceMode = Literal["diagonal", "full"]


def estimate_covariance(
    X: np.ndarray,
    reg: float = 0.01,
    mode: CovarianceMode = "diagonal",
) -> NDArray[np.float64]:
    """Estimate covariance with diagonal or full form, always adding λI.

    For n < 2, returns λI (np.cov with one sample would be NaN).
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features), got shape {arr.shape}")
    n_samples, n_features = arr.shape
    eye = np.eye(n_features, dtype=np.float64)
    if n_samples < 2:
        return reg * eye

    if mode == "diagonal":
        var = np.var(arr, axis=0, ddof=1)
        var = np.where(np.isfinite(var), var, 0.0)
        return np.diag(var) + reg * eye

    if mode != "full":
        raise ValueError(f"mode must be 'diagonal' or 'full', got {mode!r}")

    cov = np.cov(arr.T)
    if cov.ndim == 1:
        cov = np.array([[cov[0]]], dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    if not np.all(np.isfinite(cov)):
        return reg * eye
    return cov + reg * eye


def invert_covariance(cov: np.ndarray, reg: float = 0.01, max_iter: int = 6) -> NDArray[np.float64]:
    """Invert covariance with escalating extra regularization on LinAlgError."""
    cov = np.asarray(cov, dtype=np.float64)
    current_reg = reg
    for _ in range(max_iter):
        try:
            cov_reg = cov + current_reg * np.eye(cov.shape[0])
            return np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            current_reg *= 10
    cov_reg = cov + (current_reg * 10) * np.eye(cov.shape[0])
    return np.linalg.inv(cov_reg)
