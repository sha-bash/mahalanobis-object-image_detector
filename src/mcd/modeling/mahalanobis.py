import numpy as np
from numpy.typing import NDArray

from mcd.modeling.covariance import invert_covariance


def mahalanobis_distance(
    x: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    reg: float = 0.01,
) -> float:
    """Compute Mahalanobis distance with stable covariance inversion."""
    diff = np.asarray(x, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    inv_cov = invert_covariance(np.asarray(cov, dtype=np.float64), reg=reg)
    quad = float(diff.T @ inv_cov @ diff)
    return float(np.sqrt(max(quad, 0.0)))


def mahalanobis_between(
    z1: np.ndarray,
    z2: np.ndarray,
    cov: np.ndarray,
    reg: float = 0.01,
) -> float:
    """Mahalanobis distance between two vectors under a given covariance (no fit)."""
    return mahalanobis_distance(z1, z2, cov, reg=reg)


def identity_covariance(dim: int) -> NDArray[np.float64]:
    return np.eye(int(dim), dtype=np.float64)


def scaled_identity_covariance(dim: int, scale: float) -> NDArray[np.float64]:
    return float(scale) * np.eye(int(dim), dtype=np.float64)
