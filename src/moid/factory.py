from __future__ import annotations

from moid.adapters.llm import GigaChatLLM, StubLLM
from moid.adapters.ocr import EasyOCRClient, NullOCR, StubOCR
from moid.adapters.vlm import GigaChatVLM, StubVLM
from moid.config import MoidConfig
from mcd.embedding.projector import ProjectedEmbedder
from mcd.embedding.sbert import SBERT
from moid.regions.dino import DinoProposer, HybridProposer
from moid.regions.grid import GridProposer


def build_vlm(config: MoidConfig, stub: bool = False):
    if stub or config.adapters.vlm == "stub":
        return StubVLM()
    return GigaChatVLM(model=config.adapters.gigachat_model)


def build_llm(config: MoidConfig, stub: bool = False):
    if stub or config.adapters.llm == "stub":
        return StubLLM()
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
    return GridProposer(
        rows=config.grid.rows,
        cols=config.grid.cols,
        overlap=config.grid.overlap,
        extra_scales=extra,
    )


def build_proposer(config: MoidConfig, *, text_prompt: str | None = None):
    grid = build_grid(config)
    prompt = text_prompt if text_prompt is not None else config.regions.text_prompt
    prompt = (prompt or "object").strip() or "object"
    backend = config.regions.backend
    if backend == "grid":
        return grid
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
    return dino
