from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


BBoxTuple = tuple[float, float, float, float]


@dataclass(frozen=True)
class GroundTruth:
    frame: str
    bbox: BBoxTuple
    category_id: int = 1
    iscrowd: bool = False


@dataclass(frozen=True)
class Prediction:
    frame: str
    bbox: BBoxTuple
    confidence: float
    category_id: int = 1


@dataclass(frozen=True)
class MetricResult:
    precision: float
    recall: float
    f1: float
    map_50: float
    map_50_95: float
    tp: int
    fp: int
    fn: int
    ap_per_class: dict[int, float]
    pr_curve: list[dict[str, float]]
    unmatched_gt_frames: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def frame_stem(frame_index: int) -> str:
    return f"frame_{frame_index:08d}"


def _xywh_to_xyxy(box: Iterable[float]) -> BBoxTuple:
    x, y, width, height = (float(value) for value in box)
    return (x, y, x + width, y + height)


def load_ground_truth(
    path: str | Path,
    *,
    image_sizes: dict[str, tuple[int, int]] | None = None,
) -> tuple[list[GroundTruth], dict[int, str]]:
    """Load COCO JSON or a directory containing YOLO label files."""
    source = Path(path)
    if source.is_file():
        return _load_coco(source)
    if source.is_dir():
        return _load_yolo(source, image_sizes or {})
    raise FileNotFoundError(f"Ground-truth annotations not found: {source}")


def _load_coco(path: Path) -> tuple[list[GroundTruth], dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"images", "annotations", "categories"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Invalid COCO JSON; missing fields: {sorted(missing)}")
    images = {int(item["id"]): Path(item["file_name"]).stem for item in payload["images"]}
    categories = {
        int(item["id"]): str(item.get("name", item["id"])) for item in payload["categories"]
    }
    ground_truth: list[GroundTruth] = []
    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in images:
            raise ValueError(f"COCO annotation refers to unknown image_id={image_id}")
        ground_truth.append(
            GroundTruth(
                frame=images[image_id],
                bbox=_xywh_to_xyxy(annotation["bbox"]),
                category_id=int(annotation["category_id"]),
                iscrowd=bool(annotation.get("iscrowd", False)),
            )
        )
    return ground_truth, categories


def _load_yolo(
    labels_dir: Path,
    image_sizes: dict[str, tuple[int, int]],
) -> tuple[list[GroundTruth], dict[int, str]]:
    labels = sorted(labels_dir.glob("*.txt"))
    if not labels:
        raise FileNotFoundError(f"No YOLO .txt annotations found in {labels_dir}")
    ground_truth: list[GroundTruth] = []
    categories: dict[int, str] = {}
    for label_path in labels:
        stem = label_path.stem
        if stem not in image_sizes:
            raise ValueError(
                f"Image size is required for YOLO annotation {label_path.name}; "
                "its stem must match frame_XXXXXXXX"
            )
        width, height = image_sizes[stem]
        for line_number, raw in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"{label_path}:{line_number}: expected class cx cy w h")
            class_id, cx, cy, bw, bh = map(float, parts[:5])
            category_id = int(class_id) + 1
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
            ground_truth.append(GroundTruth(stem, (x1, y1, x2, y2), category_id))
            categories.setdefault(category_id, str(int(class_id)))
    return ground_truth, categories


def bbox_iou(a: BBoxTuple, b: BBoxTuple) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def distance_confidence(distance: float, threshold: float, temperature: float = 1.0) -> float:
    """Map a Mahalanobis margin to a stable probability-like ranking score."""
    scale = max(float(temperature), 1e-9)
    value = max(-60.0, min(60.0, (float(threshold) - float(distance)) / scale))
    return 1.0 / (1.0 + math.exp(-value))


def predictions_from_video_payload(
    payload: dict[str, Any], *, category_id: int = 1
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for video in payload.get("videos", []):
        for frame in video.get("frames", []):
            stem = frame_stem(int(frame["frame_index"]))
            for item in frame.get("detections", []):
                distance = item.get("distance")
                threshold = item.get("threshold")
                if distance is None or threshold is None:
                    continue
                predictions.append(
                    Prediction(
                        frame=stem,
                        bbox=tuple(float(v) for v in item["bbox"]),  # type: ignore[arg-type]
                        confidence=float(
                            item.get("confidence")
                            or distance_confidence(distance, threshold)
                        ),
                        category_id=category_id,
                    )
                )
    return predictions


def _match_flags(
    predictions: list[Prediction],
    ground_truth: list[GroundTruth],
    *,
    category_id: int,
    iou_threshold: float,
) -> tuple[list[int], list[int], int]:
    gt_by_frame: dict[str, list[GroundTruth]] = {}
    for item in ground_truth:
        if item.category_id == category_id and not item.iscrowd:
            gt_by_frame.setdefault(item.frame, []).append(item)
    used: dict[str, set[int]] = {frame: set() for frame in gt_by_frame}
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    ordered = sorted(
        (item for item in predictions if item.category_id == category_id),
        key=lambda item: item.confidence,
        reverse=True,
    )
    for prediction in ordered:
        candidates = gt_by_frame.get(prediction.frame, [])
        best_index = -1
        best_iou = iou_threshold
        for index, target in enumerate(candidates):
            if index in used.setdefault(prediction.frame, set()):
                continue
            overlap = bbox_iou(prediction.bbox, target.bbox)
            if overlap >= best_iou:
                best_iou, best_index = overlap, index
        matched = best_index >= 0
        if matched:
            used[prediction.frame].add(best_index)
        tp_flags.append(int(matched))
        fp_flags.append(int(not matched))
    total_gt = sum(len(items) for items in gt_by_frame.values())
    return tp_flags, fp_flags, total_gt


def _precision_recall(
    tp_flags: list[int], fp_flags: list[int], total_gt: int
) -> tuple[list[float], list[float]]:
    precisions: list[float] = []
    recalls: list[float] = []
    tp = fp = 0
    for tp_flag, fp_flag in zip(tp_flags, fp_flags):
        tp += tp_flag
        fp += fp_flag
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / max(total_gt, 1))
    return precisions, recalls


def _average_precision(precisions: list[float], recalls: list[float]) -> float:
    if not precisions:
        return 0.0
    total = 0.0
    for recall_level in (index / 100.0 for index in range(101)):
        candidates = [
            precision
            for precision, recall in zip(precisions, recalls)
            if recall >= recall_level
        ]
        total += max(candidates, default=0.0)
    return total / 101.0


def compute_metrics(
    ground_truth: list[GroundTruth],
    predictions: list[Prediction],
) -> MetricResult:
    category_ids = sorted({item.category_id for item in ground_truth if not item.iscrowd})
    if not category_ids:
        raise ValueError("Ground truth contains no evaluable annotations")
    thresholds = [0.5 + 0.05 * index for index in range(10)]
    ap_by_threshold: dict[float, list[float]] = {threshold: [] for threshold in thresholds}
    ap_per_class: dict[int, float] = {}
    operating_tp = operating_fp = operating_gt = 0
    pr_curve: list[dict[str, float]] = []
    for category_id in category_ids:
        class_aps: list[float] = []
        for threshold in thresholds:
            tp_flags, fp_flags, total_gt = _match_flags(
                predictions,
                ground_truth,
                category_id=category_id,
                iou_threshold=threshold,
            )
            precisions, recalls = _precision_recall(tp_flags, fp_flags, total_gt)
            ap = _average_precision(precisions, recalls)
            ap_by_threshold[threshold].append(ap)
            class_aps.append(ap)
            if threshold == 0.5:
                operating_tp += sum(tp_flags)
                operating_fp += sum(fp_flags)
                operating_gt += total_gt
                ordered = sorted(
                    (p for p in predictions if p.category_id == category_id),
                    key=lambda item: item.confidence,
                    reverse=True,
                )
                pr_curve.extend(
                    {
                        "category_id": float(category_id),
                        "confidence": ordered[index].confidence,
                        "precision": precision,
                        "recall": recall,
                    }
                    for index, (precision, recall) in enumerate(zip(precisions, recalls))
                )
        ap_per_class[category_id] = sum(class_aps) / len(class_aps)
    fn = max(operating_gt - operating_tp, 0)
    precision = operating_tp / max(operating_tp + operating_fp, 1)
    recall = operating_tp / max(operating_tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    gt_frames = {item.frame for item in ground_truth}
    pred_frames = {item.frame for item in predictions}
    return MetricResult(
        precision=precision,
        recall=recall,
        f1=f1,
        map_50=sum(ap_by_threshold[0.5]) / len(ap_by_threshold[0.5]),
        map_50_95=sum(sum(values) / len(values) for values in ap_by_threshold.values())
        / len(ap_by_threshold),
        tp=operating_tp,
        fp=operating_fp,
        fn=fn,
        ap_per_class=ap_per_class,
        pr_curve=pr_curve,
        unmatched_gt_frames=sorted(gt_frames - pred_frames),
    )
