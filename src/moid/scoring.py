from __future__ import annotations

from dataclasses import dataclass

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
) -> tuple[list[ScoredBox], bool]:
    usable = [b for b in boxes if not b.failed]
    if nms_before_accept:
        kept = nms(usable, iou_threshold=nms_iou)
        gated_pool = kept
    else:
        accepted_only = [b for b in usable if b.accepted]
        gated_pool = nms(accepted_only if accepted_only else usable, iou_threshold=nms_iou)

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
            )
        ]
    return detections, frame_positive
