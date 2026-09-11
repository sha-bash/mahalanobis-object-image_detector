from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

from mcd.modeling.thresholds import (
    ChiSquareThresholdStrategy,
    FixedThresholdStrategy,
    MaxPlusMarginThresholdStrategy,
    QuantileThresholdStrategy,
    ThresholdStrategy,
)

CovarianceMode = Literal["diagonal", "full"]
ThresholdKind = Literal["max_margin", "chi2", "fixed", "quantile", "negative_quantile"]
ZeroShotMode = Literal["pairwise", "one_shot"]
VlmKind = Literal["stub", "gigachat", "ollama"]
LlmKind = Literal["stub", "gigachat", "ollama"]
RegionBackend = Literal["grid", "dino", "dinov2", "hybrid", "yolo", "yolo_grid"]
AggregateKind = Literal["count", "min_distance", "mean_top_k"]
OcrBackend = Literal["none", "stub", "easyocr"]
PromptSchema = Literal["generic", "generic+vehicle"]


def _from_dict(cls, data: dict[str, Any] | None):
    raw = dict(data or {})
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in raw.items() if key in allowed})


@dataclass
class DetectorConfig:
    sbert_model: str = "all-MiniLM-L6-v2"
    min_cluster_size: int = 1
    regularization: float = 0.01
    covariance_mode: CovarianceMode = "diagonal"
    threshold: ThresholdKind = "max_margin"
    threshold_margin: float = 0.8
    threshold_floor: float = 0.0
    chi2_alpha: float = 0.95
    fixed_threshold: float = 10.0
    quantile: float = 0.99
    negative_quantile: float = 0.05
    projector_path: str | None = None
    manual_threshold: float | None = None
    calibration_csv: str | None = None


@dataclass
class GridConfig:
    rows: int = 3
    cols: int = 3
    overlap: float = 0.2
    extra_scales: list[list[int]] = field(default_factory=list)
    fine_scales: list[list[int]] = field(default_factory=lambda: [[4, 4], [5, 5]])
    use_fine_scales: bool = False
    fine_overlap: float = 0.1
    nms_iou: float = 0.5
    include_best_if_none_accepted: bool = False


@dataclass
class RegionsConfig:
    backend: RegionBackend = "grid"
    text_prompt: str | None = None
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    max_boxes: int = 20
    min_area_ratio: float = 0.002
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.25
    yolo_iou: float = 0.45
    yolo_vehicle_only: bool = True
    max_yolo_area_ratio: float = 0.12


@dataclass
class DecisionConfig:
    min_positive_regions: int = 1
    top_k: int = 3
    aggregate: AggregateKind = "count"
    uncertain_scale: float = 0.6
    target_match_gate: bool = True
    nms_before_accept: bool = True


@dataclass
class OcrConfig:
    backend: OcrBackend = "none"
    match_scale: float = 0.5
    mismatch_scale: float = 1.5
    min_confidence: float = 0.3


@dataclass
class QueryHintsConfig:
    text: str = "вид сверху, аэрофотосъёмка, объект на асфальте"
    known_traits: list[str] = field(default_factory=list)


@dataclass
class PromptConfig:
    language: str = "en"
    few_shot_examples: bool = True
    schema: PromptSchema = "generic+vehicle"


@dataclass
class PathsConfig:
    refs: str = "data/refs"
    search: str = "data/search"
    videos: str = "data/videos"
    reports: str = "reports"


@dataclass
class VideoConfig:
    sample_fps: float = 1.0
    extensions: list[str] = field(
        default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    )


@dataclass
class ReferenceFilterConfig:
    excluded_terms: list[str] = field(default_factory=lambda: ["sedan"])
    allowed_terms: list[str] = field(default_factory=list)
    min_references_warning: int = 2


@dataclass
class NmsConfig:
    method: Literal["hard", "soft", "wbf"] = "hard"
    iou_threshold: float | None = None
    soft_sigma: float = 0.5
    confidence_temperature: float = 1.0
    min_confidence: float = 0.001


@dataclass
class TrackingConfig:
    enabled: bool = False
    iou_threshold: float = 0.3
    max_misses: int = 5
    camera_compensation: bool = False


@dataclass
class EvaluationConfig:
    gt_annotations: str | None = None
    save_metrics: bool = False


@dataclass
class PerformanceConfig:
    enabled: bool = True


@dataclass
class VisualConfig:
    model_name: str | None = None          # e.g., "openai/clip-vit-base-patch32"
    similarity_metric: Literal["cosine", "mahalanobis"] = "cosine"
    top_k_before_vlm: int = 5
    min_similarity: float = 0.0
    use_visual_scores: bool = False
    device: str = "cpu"
    similarity_reduce: Literal["mean", "max"] = "max"
    clip_batch_size: int = 4
    fusion_weight: float = 0.5
    skip_vlm: bool = False


@dataclass
class RefineConfig:
    enabled: bool = False
    scales: list[float] = field(default_factory=lambda: [0.7, 1.0, 1.3])
    shifts: list[float] = field(default_factory=lambda: [-0.15, 0.0, 0.15])
    max_seeds: int = 8


@dataclass
class ZeroShotConfig:
    mode: ZeroShotMode = "pairwise"
    reject_threshold: float | None = None
    pairwise_scale: float = 1.0


@dataclass
class AdapterConfig:
    vlm: VlmKind = "stub"
    llm: LlmKind = "stub"
    gigachat_model: str = "GigaChat-2-Pro"
    ollama_host: str = "http://localhost:11434"
    ollama_vlm_model: str = "qwen2.5vl:3b"
    ollama_llm_model: str = "qwen2.5vl:3b"
    ollama_timeout_sec: float = 180.0
    ollama_keep_alive: str = "5m"
    ollama_num_ctx: int = 2048
    ollama_num_predict: int = 512
    ollama_max_image_side: int = 768
    ollama_crop_max_side: int = 512
    ollama_jpeg_quality: int = 85
    ollama_cache: bool = True
    ollama_cache_dir: str = ".cache/ollama_captions"


@dataclass
class MoidConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    regions: RegionsConfig = field(default_factory=RegionsConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    hints: QueryHintsConfig = field(default_factory=QueryHintsConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    zero_shot: ZeroShotConfig = field(default_factory=ZeroShotConfig)
    adapters: AdapterConfig = field(default_factory=AdapterConfig)
    target_label: str = "target"
    visual: VisualConfig = field(default_factory=VisualConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    references: ReferenceFilterConfig = field(default_factory=ReferenceFilterConfig)
    nms: NmsConfig = field(default_factory=NmsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    def threshold_strategy(self) -> ThresholdStrategy:
        d = self.detector
        if d.threshold == "max_margin":
            return MaxPlusMarginThresholdStrategy(margin=d.threshold_margin, floor=d.threshold_floor)
        if d.threshold == "chi2":
            return ChiSquareThresholdStrategy(alpha=d.chi2_alpha)
        if d.threshold in {"fixed", "negative_quantile"}:
            return FixedThresholdStrategy(value=d.fixed_threshold)
        if d.threshold == "quantile":
            return QuantileThresholdStrategy(quantile=d.quantile)
        raise ValueError(f"Unknown threshold kind: {d.threshold}")

def load_config(path: str | Path | None) -> MoidConfig:
    if path is None:
        return MoidConfig()
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    detector = dict(raw.get("detector", {}))
    if detector.get("projector_path") in ("", None):
        detector["projector_path"] = None
    return MoidConfig(
        detector=_from_dict(DetectorConfig, detector),
        grid=_grid_from_dict(raw.get("grid", {})),
        regions=_from_dict(RegionsConfig, raw.get("regions")),
        decision=_from_dict(DecisionConfig, raw.get("decision")),
        ocr=_from_dict(OcrConfig, raw.get("ocr")),
        hints=_from_dict(QueryHintsConfig, raw.get("hints")),
        prompt=_from_dict(PromptConfig, raw.get("prompt")),
        paths=_from_dict(PathsConfig, raw.get("paths")),
        video=_from_dict(VideoConfig, raw.get("video")),
        zero_shot=_from_dict(ZeroShotConfig, raw.get("zero_shot")),
        adapters=_from_dict(AdapterConfig, raw.get("adapters")),
        target_label=str(raw.get("target_label", "target")),
        visual=_from_dict(VisualConfig, raw.get("visual", {})),
        refine=_from_dict(RefineConfig, raw.get("refine", {})),
        references=_from_dict(ReferenceFilterConfig, raw.get("references")),
        nms=_from_dict(NmsConfig, raw.get("nms")),
        tracking=_from_dict(TrackingConfig, raw.get("tracking")),
        evaluation=_from_dict(EvaluationConfig, raw.get("evaluation")),
        performance=_from_dict(PerformanceConfig, raw.get("performance")),
    )


def _grid_from_dict(data: dict[str, Any]) -> GridConfig:
    extra = data.get("extra_scales") or []
    normalized = []
    for pair in extra:
        if len(pair) != 2:
            raise ValueError("grid.extra_scales entries must be [rows, cols]")
        normalized.append([int(pair[0]), int(pair[1])])
    kwargs = dict(data)
    kwargs["extra_scales"] = normalized
    fine = data.get("fine_scales") or [[4, 4], [5, 5]]
    kwargs["fine_scales"] = [[int(pair[0]), int(pair[1])] for pair in fine]
    return _from_dict(GridConfig, kwargs)
