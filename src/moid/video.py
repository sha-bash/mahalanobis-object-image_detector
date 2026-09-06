from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Any

import cv2
import numpy as np
from PIL import Image

from moid.adapters.vlm import load_image
from moid.config import MoidConfig
from moid.factory import build_vlm, build_embedder, build_ocr, build_proposer, build_visual_encoder
from moid.pipeline.few_shot import caption_references, fit_detector, detect_on_image

logger = logging.getLogger(__name__)

def extract_frames(video_path: str | Path, frame_step: int = 10) -> Iterator[tuple[int, float, Image.Image]]:
    """Yield (frame_index, timestamp_seconds, PIL Image) for every frame_step-th frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            timestamp = idx / fps
            yield idx, timestamp, pil_image
        idx += 1
    cap.release()

def process_video(
    video_path: str | Path,
    refs: str | Path,
    config: MoidConfig,
    frame_step: int = 10,
    output_json: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Run few-shot detection on selected frames of a video.
    Returns a list of per-frame results with timestamps.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    logger.info("Processing video: %s (frame_step=%d)", video_path, frame_step)

    # Build components once
    vlm = build_vlm(config)
    embedder = build_embedder(config)
    ocr = build_ocr(config)
    proposer = build_proposer(config, text_prompt=config.regions.text_prompt or "object")
    visual_encoder = build_visual_encoder(config)

    # Prepare references and detector
    ref_captions, profile = caption_references(refs, vlm, config)
    detector = fit_detector(ref_captions, embedder, config)

    # If visual encoder is used, compute reference visual embeddings once
    ref_visual_embeddings = None
    if visual_encoder is not None:
        from moid.pipeline.few_shot import list_images
        ref_images = [load_image(p) for p in list_images(refs)]
        ref_visual_embeddings = [visual_encoder.encode(img) for img in ref_images]
        ref_visual_embeddings = np.stack(ref_visual_embeddings, axis=0)  # shape (n_refs, dim)

    if visual_encoder is not None and ref_visual_embeddings is not None:
        logger.info("Reference visual embeddings computed: shape=%s", ref_visual_embeddings.shape)

    results = []
    for frame_idx, timestamp, pil_image in extract_frames(video_path, frame_step):
        # Call existing detect_on_image; target is PIL Image
        frame_result = detect_on_image(
            target=pil_image,
            detector=detector,
            vlm=vlm,
            config=config,
            proposer=proposer,
            ocr=ocr,
            profile=profile,
            reference_captions=ref_captions,
            visual_encoder=visual_encoder,
            reference_visual_embeddings=ref_visual_embeddings,
        )
        # Convert to a simple dict for storage
        frame_dict = {
            "frame_index": frame_idx,
            "timestamp": timestamp,
            "frame_positive": frame_result.frame_positive,
            "detections": [s.to_dict() for s in frame_result.detections],
            "all_regions": [s.to_dict() for s in frame_result.all_regions],
        }
        results.append(frame_dict)

    # Save to JSON if requested
    if output_json:
        Path(output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Results saved to %s (%d frames)", output_json, len(results))

    logger.info("Video processing complete: %d frames analyzed", len(results))
    return results