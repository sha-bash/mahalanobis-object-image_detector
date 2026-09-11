from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from moid.adapters.visual_encoder import VisualEncoder

from moid.adapters.llm import GigaChatLLM, OllamaLLM, StubLLM
from moid.adapters.ocr import EasyOCRClient, NullOCR, StubOCR
from moid.adapters.vlm import GigaChatVLM, OllamaVLM, StubVLM
from moid.config import MoidConfig
from mcd.embedding.projector import ProjectedEmbedder
from mcd.embedding.sbert import SBERT
from moid.regions.dino import DinoProposer, HybridProposer
from moid.regions.dinov2 import Dinov2Proposer
from moid.regions.grid import GridProposer
from moid.regions.union import UnionProposer
from moid.regions.yolo import YoloProposalGenerator

logger = logging.getLogger(__name__)


def build_vlm(config: MoidConfig, stub: bool = False):
    if stub or config.adapters.vlm == "stub":
        return StubVLM()
    if config.adapters.vlm == "ollama":
        return OllamaVLM(
            host=config.adapters.ollama_host,
            model=config.adapters.ollama_vlm_model,
            timeout_sec=config.adapters.ollama_timeout_sec,
            keep_alive=config.adapters.ollama_keep_alive,
            num_ctx=config.adapters.ollama_num_ctx,
            num_predict=config.adapters.ollama_num_predict,
            max_image_side=config.adapters.ollama_max_image_side,
            jpeg_quality=config.adapters.ollama_jpeg_quality,
            cache=config.adapters.ollama_cache,
            cache_dir=config.adapters.ollama_cache_dir,
        )
    return GigaChatVLM(model=config.adapters.gigachat_model)


def build_llm(config: MoidConfig, stub: bool = False):
    if stub or config.adapters.llm == "stub":
        return StubLLM()
    if config.adapters.llm == "ollama":
        return OllamaLLM(
            host=config.adapters.ollama_host,
            model=config.adapters.ollama_llm_model,
            timeout_sec=config.adapters.ollama_timeout_sec,
            keep_alive=config.adapters.ollama_keep_alive,
            num_ctx=config.adapters.ollama_num_ctx,
            num_predict=config.adapters.ollama_num_predict,
        )
    return GigaChatLLM(model=config.adapters.gigachat_model)


def build_embedder(config: MoidConfig, embedder=None):
    if embedder is not None:
        base = embedder
    else:
        base = SBERT(model_name=config.detector.sbert_model)
    path = config.detector.projector_path
    if path:
        return ProjectedEmbedder(base, path)
    return base


def build_ocr(config: MoidConfig):
    backend = config.ocr.backend
    if backend == "stub":
        return StubOCR()
    if backend == "easyocr":
        return EasyOCRClient(min_confidence=config.ocr.min_confidence)
    return NullOCR()


def build_grid(config: MoidConfig) -> GridProposer:
    extra = tuple((int(a), int(b)) for a, b in config.grid.extra_scales)
    fine = (
        tuple((int(a), int(b)) for a, b in config.grid.fine_scales)
        if config.grid.use_fine_scales
        else ()
    )
    return GridProposer(
        rows=config.grid.rows,
        cols=config.grid.cols,
        overlap=config.grid.overlap,
        extra_scales=extra,
        fine_scales=fine,
        fine_overlap=config.grid.fine_overlap,
    )


def _build_yolo(config: MoidConfig) -> YoloProposalGenerator:
    return YoloProposalGenerator(
        model_name=config.regions.yolo_model,
        confidence=config.regions.yolo_confidence,
        iou_threshold=config.regions.yolo_iou,
        max_boxes=config.regions.max_boxes,
        vehicle_only=config.regions.yolo_vehicle_only,
    )


def build_proposer(config: MoidConfig, *, text_prompt: str | None = None):
    grid = build_grid(config)
    prompt = text_prompt if text_prompt is not None else config.regions.text_prompt
    prompt = (prompt or "object").strip() or "object"
    backend = config.regions.backend

    if backend == "grid":
        return grid
    if backend == "yolo":
        return _build_yolo(config)
    if backend == "yolo_grid":
        return UnionProposer(
            _build_yolo(config),
            grid,
            max_yolo_area_ratio=config.regions.max_yolo_area_ratio,
        )
    if backend == "dinov2":
        return Dinov2Proposer(
            model_name=config.visual.model_name or "facebook/dinov2-small",
            min_area_ratio=config.regions.min_area_ratio,
            max_boxes=config.regions.max_boxes,
            device=config.visual.device,
        )
    dino = DinoProposer(
        model_id=config.regions.model_id,
        text_prompt=prompt,
        box_threshold=config.regions.box_threshold,
        text_threshold=config.regions.text_threshold,
        max_boxes=config.regions.max_boxes,
        min_area_ratio=config.regions.min_area_ratio,
    )
    if backend == "hybrid":
        return HybridProposer(dino, fallback=grid)
    if backend == "dino":
        return dino
    raise ValueError(f"Unknown regions.backend: {backend}")


def build_visual_encoder(config: MoidConfig) -> "VisualEncoder | None":
    if config.visual.model_name:
        from moid.adapters.visual_encoder import VisualEncoder
        return VisualEncoder(config.visual.model_name, device=config.visual.device)
    return None
