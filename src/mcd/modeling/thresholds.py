from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2


class ThresholdStrategy(Protocol):
    def compute(self, distances: NDArray[np.float64] | Sequence[float], *, feature_dim: int) -> float:
        ...


@dataclass(frozen=True)
class QuantileThresholdStrategy:
    quantile: float = 0.95

    def compute(self, distances: NDArray[np.float64] | Sequence[float], *, feature_dim: int) -> float:
        if not 0.0 < self.quantile <= 1.0:
            raise ValueError(f"quantile must be in (0, 1], got {self.quantile}")
        arr = np.asarray(distances, dtype=float)
        if arr.ndim != 1:
            raise ValueError("distances must be a 1D array")
        if arr.size == 0:
            raise ValueError("distances must not be empty")
        if self.quantile == 1.0:
            return float(np.max(arr))
        return float(np.quantile(arr, self.quantile))


@dataclass(frozen=True)
class ChiSquareThresholdStrategy:
    alpha: float = 0.95

    def compute(self, distances: NDArray[np.float64] | Sequence[float], *, feature_dim: int) -> float:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        threshold_sq = chi2.ppf(self.alpha, df=feature_dim)
        if not np.isfinite(threshold_sq):
            raise ValueError("failed to compute finite chi-square quantile")
        return float(np.sqrt(threshold_sq))


@dataclass(frozen=True)
class FixedThresholdStrategy:
    value: float

    def compute(self, distances: NDArray[np.float64] | Sequence[float], *, feature_dim: int) -> float:
        if self.value < 0:
            raise ValueError(f"threshold value must be non-negative, got {self.value}")
        return float(self.value)


@dataclass(frozen=True)
class MaxPlusMarginThresholdStrategy:
    """Threshold = max(in-cluster Mahalanobis) + margin.

    With a single training point the max in-cluster distance is 0, so the margin
    (and optional floor) fully determines acceptance.
    """

    margin: float = 2.0
    floor: float = 0.0

    def compute(self, distances: NDArray[np.float64] | Sequence[float], *, feature_dim: int) -> float:
        arr = np.asarray(distances, dtype=float)
        if arr.ndim != 1:
            raise ValueError("distances must be a 1D array")
        if arr.size == 0:
            raise ValueError("distances must not be empty")
        if self.margin < 0:
            raise ValueError(f"margin must be non-negative, got {self.margin}")
        return float(max(float(np.max(arr)) + self.margin, self.floor))


def compute_thresholds(distances: Sequence[float], quantile: float = 0.95) -> float:
    strategy = QuantileThresholdStrategy(quantile=quantile)
    return strategy.compute(distances, feature_dim=1)
