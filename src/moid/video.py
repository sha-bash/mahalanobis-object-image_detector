from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import cv2
from PIL import Image

from moid.config import MoidConfig
from moid.factory import build_embedder, build_ocr, build_proposer, build_vlm
from moid.pipeline.few_shot import caption_references, detect_on_image, fit_detector


def list_videos(path: str | Path, extensions: list[str] | None = None) -> list[Path]:
    """Return supported video files in stable order."""
    source = Path(path)
    allowed = {
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in (extensions or [".mp4", ".avi", ".mov", ".mkv", ".webm"])
    }
    if source.is_file():
        if source.suffix.lower() not in allowed:
            raise ValueError(f"Unsupported video format: {source.suffix or '(none)'}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Video path not found: {source}")
    videos = sorted(
        (item for item in source.iterdir() if item.is_file() and item.suffix.lower() in allowed),
        key=lambda item: item.name.casefold(),
    )
    if not videos:
        raise FileNotFoundError(f"No supported videos found in {source}")
    return videos


def extract_frames(
    video_path: str | Path,
    sample_fps: float = 1.0,
) -> Iterator[tuple[int, float, Image.Image]]:
    """Yield frames selected on a time grid at up to ``sample_fps``."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    effective_fps = min(float(sample_fps), source_fps)
    interval = 1.0 / effective_fps
    next_sample = 0.0
    idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            timestamp = idx / source_fps
            if timestamp + 1e-9 >= next_sample:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield idx, timestamp, Image.fromarray(frame_rgb)
                while next_sample <= timestamp + 1e-9:
                    next_sample += interval
            idx += 1
    finally:
        cap.release()

def process_video(
    video_path: str | Path,
    refs: str | Path,
    config: MoidConfig,
    sample_fps: float | None = None,
    output_json: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Run few-shot detection on selected frames of a video.
    Returns a list of per-frame results with timestamps.
    """
    # Build components once
    vlm = build_vlm(config)
    embedder = build_embedder(config)
    ocr = build_ocr(config)
    proposer = build_proposer(config, text_prompt=config.regions.text_prompt or "object")

    # Prepare references and detector
    ref_captions, profile = caption_references(refs, vlm, config)
    detector = fit_detector(ref_captions, embedder, config)

    results = []
    actual_sample_fps = sample_fps if sample_fps is not None else config.video.sample_fps
    for frame_idx, timestamp, pil_image in extract_frames(video_path, actual_sample_fps):
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
        )
        frame_dict = {
            "frame_index": frame_idx,
            "timestamp": timestamp,
            "frame_positive": frame_result.frame_positive,
            "detections": frame_result.to_dict()["detections"],
            "all_regions": frame_result.to_dict()["all_regions"],
        }
        results.append(frame_dict)

    # Save to JSON if requested
    if output_json:
        Path(output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results