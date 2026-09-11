from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

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


@dataclass(frozen=True)
class ThresholdCalibration:
    threshold: float
    precision: float
    recall: float
    f1: float


def optimize_threshold_by_f1(
    distances: list[float],
    labels: list[bool],
) -> ThresholdCalibration:
    """Select distance <= threshold using a labelled validation set."""
    if len(distances) != len(labels) or not distances:
        raise ValueError("distances and labels must be non-empty and have equal length")
    if not any(labels):
        raise ValueError(
            "No positive regions in the calibration set; refusing to fit a dummy threshold"
        )
    best = ThresholdCalibration(0.0, 0.0, 0.0, 0.0)
    candidates = sorted(set(float(value) for value in distances))
    for threshold in candidates:
        predictions = [distance <= threshold for distance in distances]
        tp = sum(prediction and label for prediction, label in zip(predictions, labels))
        fp = sum(prediction and not label for prediction, label in zip(predictions, labels))
        fn = sum(not prediction and label for prediction, label in zip(predictions, labels))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        candidate = ThresholdCalibration(threshold, precision, recall, f1)
        if (candidate.f1, candidate.recall, -candidate.threshold) > (
            best.f1,
            best.recall,
            -best.threshold,
        ):
            best = candidate
    return best


def optimize_threshold_from_csv(path: str | Path) -> ThresholdCalibration:
    """Read `distance,is_positive` validation rows and maximize F1."""
    distances: list[float] = []
    labels: list[bool] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("distance") in (None, "") or row.get("is_positive") in (None, ""):
                continue
            distances.append(float(row["distance"]))
            labels.append(str(row["is_positive"]).strip().casefold() in {"1", "true", "yes"})
    if not distances:
        raise ValueError(
            f"Calibration CSV {path} has no valid distance,is_positive rows"
        )
    return optimize_threshold_by_f1(distances, labels)
