from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time

import numpy as np
from PIL import Image

from mcd.modeling.classifier import MahalanobisDriftDetector
from moid.calibration import optimize_threshold_from_csv
from moid.adapters.base import VLMClient
from moid.adapters.ocr import NullOCR, adjust_distance
from moid.adapters.vlm import ContextualVLM, load_image
from moid.captions import parse_caption, target_match_value
from moid.config import MoidConfig
from moid.identity import IdentityProfile, build_identity_profile
from moid.prompts import build_reference_context, build_search_context
from moid.performance import StageTimer
from moid.regions.base import BBox, ProposedBox, RegionProposer, crop_box
from moid.scoring import ScoredBox, apply_target_match_gate, select_frame_detections
from moid.adapters.visual_encoder import VisualEncoder

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
logger = logging.getLogger(__name__)


@dataclass
class FewShotResult:
    detections: list[ScoredBox]
    all_regions: list[ScoredBox]
    reference_captions: list[str]
    failed_regions: int
    frame_positive: bool = False
    profile: IdentityProfile | None = None
    timings: StageTimer | None = None

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
                "proposal_confidence": s.proposal_confidence,
                "proposal_source": s.proposal_source,
                "confidence": s.confidence,
                "track_id": s.track_id,
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


def filter_reference_paths(paths: list[Path], config: MoidConfig) -> list[Path]:
    excluded = tuple(term.casefold() for term in config.references.excluded_terms)
    return [
        path
        for path in paths
        if not any(term in path.name.casefold() for term in excluded)
    ]


def caption_references(
    refs: str | Path,
    vlm: VLMClient,
    config: MoidConfig | None = None,
) -> tuple[list[str], IdentityProfile]:
    captions, profile, _ = caption_references_with_paths(refs, vlm, config)
    return captions, profile


def caption_references_with_paths(
    refs: str | Path,
    vlm: VLMClient,
    config: MoidConfig | None = None,
) -> tuple[list[str], IdentityProfile, list[Path]]:
    cfg = config or MoidConfig()
    context = build_reference_context(cfg.hints.text, cfg.hints.known_traits)
    ref_vlm = ContextualVLM(vlm, context) if not isinstance(vlm, ContextualVLM) else vlm
    candidates = list_images(refs)
    excluded = tuple(term.casefold() for term in cfg.references.excluded_terms)
    allowed = tuple(term.casefold() for term in cfg.references.allowed_terms)
    paths = filter_reference_paths(candidates, cfg)
    captions: list[str] = []
    used_paths: list[Path] = []
    for path in paths:
        text = (ref_vlm.describe(path) or "").strip()
        searchable = f"{path.name} {text}".casefold()
        if any(term in searchable for term in excluded):
            logger.info("Excluded reference %s by caption filter", path)
            continue
        if allowed and not any(term in searchable for term in allowed):
            logger.info("Excluded reference %s: no allowed target term", path)
            continue
        if text:
            captions.append(text)
            used_paths.append(path)
    if not captions:
        raise RuntimeError("No valid target references remain after filtering")
    if len(captions) < cfg.references.min_references_warning:
        logger.warning(
            "Only %d valid reference(s) remain; covariance and threshold may be unstable",
            len(captions),
        )
    profile = build_identity_profile(captions, cfg.hints.text, cfg.hints.known_traits)
    return captions, profile, used_paths


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
    if cfg.detector.manual_threshold is not None:
        detector.set_threshold(cfg.detector.manual_threshold)
    elif cfg.detector.calibration_csv:
        calibrated = optimize_threshold_from_csv(cfg.detector.calibration_csv)
        detector.set_threshold(calibrated.threshold)
        logger.info(
            "Calibrated Mahalanobis threshold %.6f (validation F1 %.4f)",
            calibrated.threshold,
            calibrated.f1,
        )
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
    visual_encoder: VisualEncoder | None = None,
    reference_visual_embeddings: np.ndarray | list | None = None,
    timer: StageTimer | None = None,
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
    timer = timer or StageTimer(enabled=cfg.performance.enabled)
    frame_started = time.perf_counter()
    target_image = load_image(target)
    with timer.measure("proposals"):
        boxes = proposer.propose(target_image)
    scored = _score_regions(
        target_image,
        boxes,
        search_vlm,
        detector,
        ocr=ocr or NullOCR(),
        profile=profile,
        config=cfg,
        visual_encoder=visual_encoder,
        reference_visual_embeddings=reference_visual_embeddings,
        timer=timer,
    )
    failed = sum(1 for s in scored if s.failed)
    with timer.measure("postprocess"):
        gated = apply_target_match_gate(
            scored,
            uncertain_scale=cfg.decision.uncertain_scale,
            gated=cfg.decision.target_match_gate,
        )
        detections, frame_positive = select_frame_detections(
            gated,
            nms_iou=(
                cfg.nms.iou_threshold
                if cfg.nms.iou_threshold is not None
                else cfg.grid.nms_iou
            ),
            min_positive_regions=cfg.decision.min_positive_regions,
            aggregate=cfg.decision.aggregate,
            top_k=cfg.decision.top_k,
            include_best_if_none_accepted=cfg.grid.include_best_if_none_accepted,
            nms_before_accept=cfg.decision.nms_before_accept,
            nms_method=cfg.nms.method,
            soft_sigma=cfg.nms.soft_sigma,
            confidence_temperature=cfg.nms.confidence_temperature,
            min_confidence=cfg.nms.min_confidence,
            image_size=target_image.size,
        )
    timer.add("frame_total", (time.perf_counter() - frame_started) * 1000.0)
    return FewShotResult(
        detections=detections,
        all_regions=gated,
        reference_captions=reference_captions or profile.captions,
        failed_regions=failed,
        frame_positive=frame_positive,
        profile=profile,
        timings=timer,
    )


def _score_regions(
    image: Image.Image,
    boxes: list[BBox | ProposedBox],
    vlm: VLMClient,
    detector: MahalanobisDriftDetector,
    *,
    ocr,
    profile: IdentityProfile,
    config: MoidConfig,
    visual_encoder: VisualEncoder | None = None,
    reference_visual_embeddings: np.ndarray | list | None = None,
    timer: StageTimer | None = None,
) -> list[ScoredBox]:
    timer = timer or StageTimer(enabled=False)
    proposals = [
        item if isinstance(item, ProposedBox) else ProposedBox(item, 1.0, "grid")
        for item in boxes
    ]
    plain_boxes = [item.box for item in proposals]
    captions: list[str] = []
    valid_idx: list[int] = []
    results: list[ScoredBox | None] = [None] * len(plain_boxes)
    crops: dict[int, Image.Image] = {}

    # Precompute crops for all boxes (needed for visual filtering too)
    for i, box in enumerate(plain_boxes):
        crop = crop_box(image, box)
        crops[i] = crop

    # Determine which boxes to process with VLM
    selected_indices = list(range(len(plain_boxes)))

    if visual_encoder is not None and reference_visual_embeddings is not None and len(plain_boxes) > 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Visual pre-filter: %d boxes -> selecting top %d (min_sim=%.2f)",
                    len(boxes), config.visual.top_k_before_vlm, config.visual.min_similarity)
        # Compute visual embeddings for all crops
        visual_started = time.perf_counter()
        crop_embeddings = [visual_encoder.encode(crops[i]) for i in selected_indices]
        ref_emb = np.asarray(reference_visual_embeddings)  # shape (n_refs, dim)

        # Compute cosine similarity between each crop and each reference
        # Normalize embeddings for cosine similarity
        crop_emb = np.asarray(crop_embeddings)
        crop_norm = np.linalg.norm(crop_emb, axis=1, keepdims=True)
        crop_norm[crop_norm == 0] = 1e-9
        crop_emb_norm = crop_emb / crop_norm

        ref_norm = np.linalg.norm(ref_emb, axis=1, keepdims=True)
        ref_norm[ref_norm == 0] = 1e-9
        ref_emb_norm = ref_emb / ref_norm

        sims = np.dot(crop_emb_norm, ref_emb_norm.T)  # shape (n_crops, n_refs)
        mean_sim = sims.mean(axis=1)

        # Select top_k_before_vlm regions (and optionally those above min_similarity)
        top_k = config.visual.top_k_before_vlm
        min_sim = config.visual.min_similarity

        if top_k > 0 and top_k < len(plain_boxes):
            # Get indices of top_k highest mean_sim
            top_indices = np.argsort(mean_sim)[-top_k:]
            selected_indices = sorted(top_indices.tolist())
        if min_sim > 0.0:
            # Additional filter: only keep indices with mean_sim >= min_sim
            selected_indices = [i for i in selected_indices if mean_sim[i] >= min_sim]
        timer.add("clip_filter", (time.perf_counter() - visual_started) * 1000.0)

    # Now process selected indices with VLM
    vlm_started = time.perf_counter()
    for i in selected_indices:
        crop = crops[i]
        text = (vlm.describe(crop) or "").strip()
        if not text:
            results[i] = ScoredBox(
                box=plain_boxes[i],
                distance=float("inf"),
                threshold=float("nan"),
                accepted=False,
                caption="",
                failed=True,
                proposal_confidence=proposals[i].confidence,
                proposal_source=proposals[i].source,
            )
            continue
        captions.append(text)
        valid_idx.append(i)
    timer.add("vlm", (time.perf_counter() - vlm_started) * 1000.0)

    # For non-selected indices, mark as failed with some default distance
    for i in range(len(plain_boxes)):
        if i not in selected_indices and results[i] is None:
            results[i] = ScoredBox(
                box=plain_boxes[i],
                distance=float("inf"),
                threshold=float("nan"),
                accepted=False,
                caption="",
                failed=True,
                proposal_confidence=proposals[i].confidence,
                proposal_source=proposals[i].source,
            )

    if captions:
        embed_started = time.perf_counter()
        cleaned = [detector._clean_text(caption) for caption in captions]
        embeddings = detector.embedder.embed(cleaned)
        timer.add("sbert", (time.perf_counter() - embed_started) * 1000.0)
        mahalanobis_started = time.perf_counter()
        preds = detector.predict_embeddings(embeddings)
        timer.add("mahalanobis", (time.perf_counter() - mahalanobis_started) * 1000.0)
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
                box=plain_boxes[i],
                distance=float(dist),
                threshold=thr,
                accepted=float(dist) <= float(thr),
                caption=caption,
                failed=False,
                target_match=target_match_value(fields),
                raw_distance=raw,
                ocr_text=observed,
                proposal_confidence=proposals[i].confidence,
                proposal_source=proposals[i].source,
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
