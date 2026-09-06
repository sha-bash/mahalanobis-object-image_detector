from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from mcd.modeling.classifier import MahalanobisDriftDetector
from moid.adapters.base import VLMClient
from moid.adapters.ocr import NullOCR, adjust_distance
from moid.adapters.vlm import ContextualVLM, load_image
from moid.captions import parse_caption, target_match_value
from moid.config import MoidConfig
from moid.identity import IdentityProfile, build_identity_profile
from moid.prompts import build_reference_context, build_search_context
from moid.regions.base import BBox, RegionProposer, crop_box
from moid.scoring import ScoredBox, apply_target_match_gate, select_frame_detections

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class FewShotResult:
    detections: list[ScoredBox]
    all_regions: list[ScoredBox]
    reference_captions: list[str]
    failed_regions: int
    frame_positive: bool = False
    profile: IdentityProfile | None = None

    def to_dict(self) -> dict:
        def box_dict(s: ScoredBox) -> dict:
            return {
                "bbox": list(s.box.as_tuple()),
                "distance": s.distance,
                "threshold": s.threshold,
                "score": s.score,
                "accepted": s.accepted,
                "failed": s.failed,
                "caption": s.caption,
                "target_match": s.target_match,
                "raw_distance": s.raw_distance,
                "ocr_text": s.ocr_text,
                "visualization_only": s.visualization_only,
            }

        return {
            "detections": [box_dict(s) for s in self.detections],
            "all_regions": [box_dict(s) for s in self.all_regions],
            "reference_captions": self.reference_captions,
            "failed_regions": self.failed_regions,
            "frame_positive": self.frame_positive,
            "profile": None if self.profile is None else self.profile.fields,
        }


def list_images(folder: str | Path) -> list[Path]:
    path = Path(folder)
    if path.is_file():
        return [path]
    files = [p for p in sorted(path.iterdir()) if p.suffix.lower() in _IMAGE_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"No images found in {path}")
    return files


def caption_references(
    refs: str | Path,
    vlm: VLMClient,
    config: MoidConfig | None = None,
) -> tuple[list[str], IdentityProfile]:
    cfg = config or MoidConfig()
    context = build_reference_context(cfg.hints.text, cfg.hints.known_traits)
    ref_vlm = ContextualVLM(vlm, context) if not isinstance(vlm, ContextualVLM) else vlm
    captions: list[str] = []
    for path in list_images(refs):
        text = (ref_vlm.describe(path) or "").strip()
        if text:
            captions.append(text)
    if not captions:
        raise RuntimeError("VLM produced no captions for reference images")
    profile = build_identity_profile(captions, cfg.hints.text, cfg.hints.known_traits)
    return captions, profile


def fit_detector(captions: list[str], embedder, config: MoidConfig | None = None) -> MahalanobisDriftDetector:
    cfg = config or MoidConfig()
    detector = MahalanobisDriftDetector(
        embedder=embedder,
        min_cluster_size=cfg.detector.min_cluster_size,
        regularization=cfg.detector.regularization,
        covariance_mode=cfg.detector.covariance_mode,
        threshold_strategy=cfg.threshold_strategy(),
        normalize_tickets=False,
        embedder_model_name=cfg.detector.sbert_model,
    )
    labels = [cfg.target_label] * len(captions)
    detector.fit(captions, labels)
    return detector


def run_few_shot(
    refs: str | Path,
    target: str | Path,
    vlm: VLMClient,
    embedder,
    config: MoidConfig | None = None,
    proposer: RegionProposer | None = None,
    ocr=None,
    captions: list[str] | None = None,
    profile: IdentityProfile | None = None,
) -> FewShotResult:
    cfg = config or MoidConfig()
    if captions is None or profile is None:
        captions, profile = caption_references(refs, vlm, cfg)
    detector = fit_detector(captions, embedder, cfg)
    return detect_on_image(
        target,
        detector,
        vlm,
        config=cfg,
        proposer=proposer,
        ocr=ocr,
        profile=profile,
        reference_captions=captions,
    )


def detect_on_image(
    target: str | Path,
    detector: MahalanobisDriftDetector,
    vlm: VLMClient,
    config: MoidConfig | None = None,
    proposer: RegionProposer | None = None,
    ocr=None,
    profile: IdentityProfile | None = None,
    reference_captions: list[str] | None = None,
) -> FewShotResult:
    from moid.factory import build_proposer

    cfg = config or MoidConfig()
    if profile is None:
        profile = build_identity_profile(reference_captions or [])
    search_vlm = ContextualVLM(
        vlm,
        build_search_context(profile.as_line(), cfg.hints.text),
    )
    if proposer is None:
        prompt = cfg.regions.text_prompt or profile.object
        proposer = build_proposer(cfg, text_prompt=prompt)
    target_image = load_image(target)
    boxes = proposer.propose(target_image)
    scored = _score_regions(
        target_image,
        boxes,
        search_vlm,
        detector,
        ocr=ocr or NullOCR(),
        profile=profile,
        config=cfg,
    )
    failed = sum(1 for s in scored if s.failed)
    gated = apply_target_match_gate(
        scored,
        uncertain_scale=cfg.decision.uncertain_scale,
        gated=cfg.decision.target_match_gate,
    )
    detections, frame_positive = select_frame_detections(
        gated,
        nms_iou=cfg.grid.nms_iou,
        min_positive_regions=cfg.decision.min_positive_regions,
        aggregate=cfg.decision.aggregate,
        top_k=cfg.decision.top_k,
        include_best_if_none_accepted=cfg.grid.include_best_if_none_accepted,
        nms_before_accept=cfg.decision.nms_before_accept,
    )
    return FewShotResult(
        detections=detections,
        all_regions=gated,
        reference_captions=reference_captions or profile.captions,
        failed_regions=failed,
        frame_positive=frame_positive,
        profile=profile,
    )


def _score_regions(
    image: Image.Image,
    boxes: list[BBox],
    vlm: VLMClient,
    detector: MahalanobisDriftDetector,
    *,
    ocr,
    profile: IdentityProfile,
    config: MoidConfig,
) -> list[ScoredBox]:
    captions: list[str] = []
    valid_idx: list[int] = []
    results: list[ScoredBox | None] = [None] * len(boxes)
    crops: dict[int, Image.Image] = {}
    for i, box in enumerate(boxes):
        crop = crop_box(image, box)
        text = (vlm.describe(crop) or "").strip()
        if not text:
            results[i] = ScoredBox(
                box=box,
                distance=float("inf"),
                threshold=float("nan"),
                accepted=False,
                caption="",
                failed=True,
            )
            continue
        captions.append(text)
        valid_idx.append(i)
        crops[i] = crop

    if captions:
        preds = detector.predict_batch(captions)
        for i, pred, caption in zip(valid_idx, preds, captions):
            _label, dist, thr, _is_drift = pred
            fields = parse_caption(caption)
            raw = float(dist)
            observed = _ocr_if_needed(fields, crops[i], ocr, profile, config)
            if observed:
                dist = adjust_distance(
                    dist,
                    observed,
                    profile.markings,
                    match_scale=config.ocr.match_scale,
                    mismatch_scale=config.ocr.mismatch_scale,
                )
            results[i] = ScoredBox(
                box=boxes[i],
                distance=float(dist),
                threshold=thr,
                accepted=float(dist) <= float(thr),
                caption=caption,
                failed=False,
                target_match=target_match_value(fields),
                raw_distance=raw,
                ocr_text=observed,
            )
    return [s for s in results if s is not None]


def _ocr_if_needed(fields, crop, ocr, profile, config) -> str | None:
    if config.ocr.backend == "none" or ocr is None:
        return None
    if profile.domain != "vehicle":
        return None
    parts = fields.get("parts", "").lower()
    if "plate" not in parts and "license" not in parts:
        return None
    read = ocr.read_text(crop)
    return None if read is None else read.text
