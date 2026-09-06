"""Drift detection utilities."""


def detect_drift(distance: float, threshold: float) -> bool:
    """True if distance > threshold (out of cluster / background)."""
    return distance > threshold
