from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from moid.regions.base import BBox
from moid.scoring import ScoredBox, iou


@dataclass
class Track:
    track_id: int
    box: BBox
    misses: int = 0
    history: list[BBox] = field(default_factory=list)


class Tracker:
    """Greedy IoU tracker suitable for sparse, independently sampled frames."""

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 5) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.tracks: list[Track] = []
        self.completed: list[Track] = []
        self._next_id = 1

    def update(
        self,
        detections: list[ScoredBox],
        transform: np.ndarray | None = None,
    ) -> list[ScoredBox]:
        predicted = [
            _transform_box(track.box, transform) if transform is not None else track.box
            for track in self.tracks
        ]
        candidates = sorted(
            (
                (iou(predicted[track_index], detection.box), track_index, detection_index)
                for track_index in range(len(self.tracks))
                for detection_index, detection in enumerate(detections)
            ),
            reverse=True,
        )
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for overlap, track_index, detection_index in candidates:
            if overlap < self.iou_threshold:
                break
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            track = self.tracks[track_index]
            detection = detections[detection_index]
            track.box = detection.box
            track.history.append(detection.box)
            track.misses = 0
            detection.track_id = track.track_id
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)
        for index, track in enumerate(self.tracks):
            if index not in matched_tracks:
                track.misses += 1
        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            track = Track(self._next_id, detection.box, history=[detection.box])
            self._next_id += 1
            self.tracks.append(track)
            detection.track_id = track.track_id
        alive: list[Track] = []
        for track in self.tracks:
            if track.misses <= self.max_misses:
                alive.append(track)
            else:
                self.completed.append(track)
        self.tracks = alive
        return detections

    def stability(self, image_size: tuple[int, int]) -> dict[str, float | int]:
        width, height = image_size
        center_jitter: list[float] = []
        overlaps: list[float] = []
        for track in [*self.completed, *self.tracks]:
            for previous, current in zip(track.history, track.history[1:]):
                px = (previous.x1 + previous.x2) / 2.0
                py = (previous.y1 + previous.y2) / 2.0
                cx = (current.x1 + current.x2) / 2.0
                cy = (current.y1 + current.y2) / 2.0
                center_jitter.append(
                    float(np.hypot((cx - px) / width, (cy - py) / height))
                )
                overlaps.append(iou(previous, current))
        return {
            "tracks": len(self.completed) + len(self.tracks),
            "matched_transitions": len(center_jitter),
            "j_center": float(np.mean(center_jitter)) if center_jitter else 0.0,
            "s_iou": float(np.mean(overlaps)) if overlaps else 0.0,
        }


class CameraMotionEstimator:
    """Estimate previous-to-current homography using ORB feature matches."""

    def __init__(self, enabled: bool = False, max_features: int = 1000) -> None:
        self.enabled = enabled
        self.max_features = max_features
        self.previous_gray: np.ndarray | None = None

    def update(self, image: Image.Image) -> np.ndarray | None:
        if not self.enabled:
            return None
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Camera compensation requires the 'video' extra") from exc
        current = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        if self.previous_gray is None:
            self.previous_gray = current
            return None
        detector = cv2.ORB_create(nfeatures=self.max_features)
        previous_points, previous_descriptors = detector.detectAndCompute(
            self.previous_gray, None
        )
        current_points, current_descriptors = detector.detectAndCompute(current, None)
        self.previous_gray = current
        if previous_descriptors is None or current_descriptors is None:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(
            matcher.match(previous_descriptors, current_descriptors),
            key=lambda match: match.distance,
        )
        if len(matches) < 8:
            return None
        source = np.float32([previous_points[m.queryIdx].pt for m in matches[:200]])
        destination = np.float32([current_points[m.trainIdx].pt for m in matches[:200]])
        homography, _ = cv2.findHomography(source, destination, cv2.RANSAC, 3.0)
        return homography


def _transform_box(box: BBox, transform: np.ndarray) -> BBox:
    corners = np.asarray(
        [[[box.x1, box.y1], [box.x2, box.y1], [box.x2, box.y2], [box.x1, box.y2]]],
        dtype=np.float32,
    )
    try:
        import cv2

        transformed = cv2.perspectiveTransform(corners, transform)[0]
    except Exception:
        return box
    return BBox(
        int(np.floor(transformed[:, 0].min())),
        int(np.floor(transformed[:, 1].min())),
        int(np.ceil(transformed[:, 0].max())),
        int(np.ceil(transformed[:, 1].max())),
    )
