from __future__ import annotations

from mcd.modeling.classifier import MahalanobisDriftDetector
from mcd.modeling.thresholds import QuantileThresholdStrategy


def calibrate_threshold(
    detector: MahalanobisDriftDetector,
    negative_captions: list[str],
    *,
    quantile: float = 0.05,
    floor: float = 0.0,
) -> float:
    """Set cluster threshold from negative sample distances (lower quantile = stricter)."""
    value = detector.calibrate_from_negatives(negative_captions, quantile=quantile)
    value = max(float(value), float(floor))
    detector.set_threshold(value)
    return value


def in_cluster_quantile(distances: list[float], quantile: float) -> float:
    return QuantileThresholdStrategy(quantile=quantile).compute(distances, feature_dim=1)
