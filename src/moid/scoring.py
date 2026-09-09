from __future__ import annotations

from dataclasses import dataclass
import math

from moid.captions import target_match_value
from moid.regions.base import BBox


@dataclass
class ScoredBox:
    box: BBox
    distance: float
    threshold: float
    accepted: bool
    caption: str
    failed: bool = False
    target_match: str = "unknown"
    raw_distance: float | None = None
    ocr_text: str | None = None
    visualization_only: bool = False
    proposal_confidence: float = 1.0
    proposal_source: str = "unknown"
    confidence: float | None = None
    track_id: int | None = None

    @property
    def score(self) -> float:
        """Lower is better. Negative when inside the cluster (d < threshold)."""
        return self.distance - self.threshold


def iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.area() + b.area() - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms(boxes: list[ScoredBox], iou_threshold: float = 0.5) -> list[ScoredBox]:
    """Keep lower-distance boxes; drop overlaps above IoU threshold."""
    ordered = sorted(boxes, key=lambda s: s.distance)
    kept: list[ScoredBox] = []
    for cand in ordered:
        if cand.failed:
            continue
        if all(iou(cand.box, k.box) < iou_threshold for k in kept):
            kept.append(cand)
    return kept


def mahalanobis_confidence(box: ScoredBox, temperature: float = 1.0) -> float:
    scale = max(float(temperature), 1e-9)
    margin = max(-60.0, min(60.0, (box.threshold - box.distance) / scale))
    semantic = 1.0 / (1.0 + math.exp(-margin))
    return float(max(0.0, min(1.0, semantic * box.proposal_confidence)))


def soft_nms(
    boxes: list[ScoredBox],
    *,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    min_confidence: float = 0.001,
    temperature: float = 1.0,
) -> list[ScoredBox]:
    """Gaussian Soft-NMS using the Mahalanobis margin as confidence."""
    remaining = list(boxes)
    for box in remaining:
        box.confidence = mahalanobis_confidence(box, temperature)
    kept: list[ScoredBox] = []
    while remaining:
        best = max(remaining, key=lambda item: item.confidence or 0.0)
        remaining.remove(best)
        if (best.confidence or 0.0) < min_confidence:
            break
        kept.append(best)
        next_remaining: list[ScoredBox] = []
        for candidate in remaining:
            overlap = iou(best.box, candidate.box)
            if overlap >= iou_threshold:
                candidate.confidence = (candidate.confidence or 0.0) * math.exp(
                    -(overlap * overlap) / max(sigma, 1e-9)
                )
            if (candidate.confidence or 0.0) >= min_confidence:
                next_remaining.append(candidate)
        remaining = next_remaining
    return kept


def weighted_boxes_fusion(
    boxes: list[ScoredBox],
    *,
    image_size: tuple[int, int],
    iou_threshold: float = 0.5,
    min_confidence: float = 0.001,
    temperature: float = 1.0,
) -> list[ScoredBox]:
    try:
        from ensemble_boxes import weighted_boxes_fusion as ensemble_wbf
    except ImportError as exc:
        raise RuntimeError(
            'WBF requires the optional dependency: pip install -e ".[fusion]"'
        ) from exc
    if not boxes:
        return []
    width, height = image_size
    scores = [mahalanobis_confidence(box, temperature) for box in boxes]
    normalized = [
        [
            box.box.x1 / width,
            box.box.y1 / height,
            box.box.x2 / width,
            box.box.y2 / height,
        ]
        for box in boxes
    ]
    fused_boxes, fused_scores, _ = ensemble_wbf(
        [normalized],
        [scores],
        [[0] * len(boxes)],
        iou_thr=iou_threshold,
        skip_box_thr=min_confidence,
    )
    output: list[ScoredBox] = []
    for coordinates, score in zip(fused_boxes, fused_scores):
        representative = min(
            boxes,
            key=lambda item: sum(
                abs(a - b)
                for a, b in zip(
                    coordinates,
                    (
                        item.box.x1 / width,
                        item.box.y1 / height,
                        item.box.x2 / width,
                        item.box.y2 / height,
                    ),
                )
            ),
        )
        representative.box = BBox(
            int(round(coordinates[0] * width)),
            int(round(coordinates[1] * height)),
            int(round(coordinates[2] * width)),
            int(round(coordinates[3] * height)),
        )
        representative.confidence = float(score)
        output.append(representative)
    return output


def effective_threshold(threshold: float, match: str, *, uncertain_scale: float, gated: bool) -> float:
    if not gated:
        return threshold
    if match == "no":
        return -1.0
    if match in {"uncertain", "unknown"}:
        return float(threshold) * float(uncertain_scale)
    return float(threshold)


def apply_target_match_gate(
    boxes: list[ScoredBox],
    *,
    uncertain_scale: float = 0.6,
    gated: bool = True,
) -> list[ScoredBox]:
    updated: list[ScoredBox] = []
    for box in boxes:
        match = box.target_match or target_match_value(box.caption)
        thr = effective_threshold(box.threshold, match, uncertain_scale=uncertain_scale, gated=gated)
        accepted = (not box.failed) and box.distance <= thr and match != "no"
        if gated and match == "no":
            accepted = False
        updated.append(
            ScoredBox(
                box=box.box,
                distance=box.distance,
                threshold=box.threshold,
                accepted=accepted,
                caption=box.caption,
                failed=box.failed,
                target_match=match,
                raw_distance=box.raw_distance,
                ocr_text=box.ocr_text,
                visualization_only=False,
                proposal_confidence=box.proposal_confidence,
                proposal_source=box.proposal_source,
                confidence=box.confidence,
                track_id=box.track_id,
            )
        )
    return updated


def select_frame_detections(
    boxes: list[ScoredBox],
    *,
    nms_iou: float = 0.5,
    min_positive_regions: int = 1,
    aggregate: str = "count",
    top_k: int = 3,
    include_best_if_none_accepted: bool = False,
    nms_before_accept: bool = True,
    nms_method: str = "hard",
    soft_sigma: float = 0.5,
    confidence_temperature: float = 1.0,
    min_confidence: float = 0.001,
    image_size: tuple[int, int] | None = None,
) -> tuple[list[ScoredBox], bool]:
    usable = [b for b in boxes if not b.failed]
    for box in usable:
        box.confidence = mahalanobis_confidence(box, confidence_temperature)

    def suppress(pool: list[ScoredBox]) -> list[ScoredBox]:
        if nms_method == "soft":
            return soft_nms(
                pool,
                iou_threshold=nms_iou,
                sigma=soft_sigma,
                min_confidence=min_confidence,
                temperature=confidence_temperature,
            )
        if nms_method == "wbf":
            if image_size is None:
                raise ValueError("image_size is required for WBF")
            return weighted_boxes_fusion(
                pool,
                image_size=image_size,
                iou_threshold=nms_iou,
                min_confidence=min_confidence,
                temperature=confidence_temperature,
            )
        return nms(pool, iou_threshold=nms_iou)

    if nms_before_accept:
        kept = suppress(usable)
        gated_pool = kept
    else:
        accepted_only = [b for b in usable if b.accepted]
        gated_pool = suppress(accepted_only if accepted_only else usable)

    accepted = [b for b in gated_pool if b.accepted]
    ranked = sorted(gated_pool, key=lambda s: s.distance)
    top = ranked[: max(1, top_k)]
    frame_positive = False
    if aggregate == "min_distance" and top:
        frame_positive = top[0].accepted
    elif aggregate == "mean_top_k" and top:
        mean_d = sum(s.distance for s in top) / len(top)
        mean_t = sum(s.threshold for s in top) / len(top)
        frame_positive = mean_d <= mean_t and len(accepted) >= min_positive_regions
    else:
        frame_positive = len(accepted) >= max(1, min_positive_regions)

    detections = list(accepted) if frame_positive else []
    if not detections and include_best_if_none_accepted and ranked:
        best = ranked[0]
        detections = [
            ScoredBox(
                box=best.box,
                distance=best.distance,
                threshold=best.threshold,
                accepted=False,
                caption=best.caption,
                failed=best.failed,
                target_match=best.target_match,
                raw_distance=best.raw_distance,
                ocr_text=best.ocr_text,
                visualization_only=True,
                proposal_confidence=best.proposal_confidence,
                proposal_source=best.proposal_source,
                confidence=best.confidence,
                track_id=best.track_id,
            )
        ]
    return detections, frame_positive
