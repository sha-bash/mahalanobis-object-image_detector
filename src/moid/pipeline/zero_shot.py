from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mcd.modeling.classifier import MahalanobisDriftDetector
from mcd.modeling.mahalanobis import mahalanobis_between, scaled_identity_covariance
from moid.adapters.base import LLMClient, VLMClient
from moid.config import MoidConfig
from moid.pipeline.few_shot import list_images


@dataclass
class ImageScore:
    path: str
    caption: str
    distance: float
    failed: bool = False


@dataclass
class ZeroShotResult:
    best: ImageScore | None
    scores: list[ImageScore]
    query_normalized: str
    rejected: bool

    def to_dict(self) -> dict:
        def as_dict(s: ImageScore) -> dict:
            return {
                "path": s.path,
                "caption": s.caption,
                "distance": None if s.failed else s.distance,
                "failed": s.failed,
            }

        return {
            "best": as_dict(self.best) if self.best else None,
            "scores": [as_dict(s) for s in self.scores],
            "query_normalized": self.query_normalized,
            "rejected": self.rejected,
        }


def run_zero_shot(
    query: str,
    images: str | Path,
    vlm: VLMClient,
    llm: LLMClient,
    embedder,
    config: MoidConfig | None = None,
) -> ZeroShotResult:
    cfg = config or MoidConfig()
    normalized = (llm.normalize_query(query) or "").strip()
    if not normalized:
        raise RuntimeError("LLM produced an empty normalized query")

    paths = list_images(images)
    captions: list[str] = []
    valid_paths: list[Path] = []
    failed: list[ImageScore] = []
    for path in paths:
        text = (vlm.describe(path) or "").strip()
        if not text:
            failed.append(ImageScore(path=str(path), caption="", distance=float("inf"), failed=True))
            continue
        captions.append(text)
        valid_paths.append(path)

    if not captions:
        return ZeroShotResult(best=None, scores=failed, query_normalized=normalized, rejected=True)

    if cfg.zero_shot.mode == "one_shot":
        detector = MahalanobisDriftDetector(
            embedder=embedder,
            min_cluster_size=1,
            regularization=cfg.detector.regularization,
            covariance_mode=cfg.detector.covariance_mode,
            threshold_strategy=cfg.threshold_strategy(),
            normalize_tickets=False,
            embedder_model_name=cfg.detector.sbert_model,
        )
        detector.fit([normalized], [cfg.target_label])
        preds = detector.predict_batch(captions)
        distances = [p[1] for p in preds]
    elif cfg.zero_shot.mode == "pairwise":
        q = np.asarray(embedder.embed([normalized]), dtype=np.float64)[0]
        zs = np.asarray(embedder.embed(captions), dtype=np.float64)
        cov = scaled_identity_covariance(q.shape[0], cfg.zero_shot.pairwise_scale)
        distances = [mahalanobis_between(q, z, cov, reg=cfg.detector.regularization) for z in zs]
    else:
        raise ValueError(f"Unknown zero-shot mode: {cfg.zero_shot.mode}")

    scored = [
        ImageScore(path=str(p), caption=c, distance=float(d), failed=False)
        for p, c, d in zip(valid_paths, captions, distances)
    ]
    scored.sort(key=lambda s: s.distance)
    best = scored[0]
    reject_thr = cfg.zero_shot.reject_threshold
    rejected = reject_thr is not None and best.distance > reject_thr
    return ZeroShotResult(
        best=None if rejected else best,
        scores=scored + failed,
        query_normalized=normalized,
        rejected=rejected,
    )
