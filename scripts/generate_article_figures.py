"""Generate publication-ready figures for the 20260910 van experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "20260910_005553_van"
DEFAULT_ANNOTATIONS = ROOT / "annotations" / "instances.json"
DEFAULT_OUTPUT = ROOT / "docs" / "paper" / "figures"

COLORS = {
    "blue": "#276FBF",
    "cyan": "#2A9D8F",
    "green": "#2E8B57",
    "orange": "#E07A2D",
    "red": "#C94141",
    "purple": "#6C5AA7",
    "gray": "#6B7280",
    "light": "#E8EDF3",
    "dark": "#1F2937",
    "white": "#FFFFFF",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": COLORS["gray"],
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def add_source(fig: Figure, text: str) -> None:
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=7.5, color=COLORS["gray"])


def box(
    ax: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    color: str,
    subtitle: str | None = None,
    linestyle: str = "-",
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=color,
        facecolor=COLORS["white"],
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * (0.58 if subtitle else 0.5),
        text,
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["dark"],
        weight="bold",
    )
    if subtitle:
        ax.text(
            x + width / 2,
            y + height * 0.25,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.1,
            color=COLORS["gray"],
        )


def arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["gray"],
    linestyle: str = "-",
    label: str | None = None,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.25,
        color=color,
        linestyle=linestyle,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(patch)
    if label:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.018,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
        )


def plot_pipeline_architecture(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Архитектура few-shot поиска объектов", loc="left", weight="bold", pad=14)

    ref_y, search_y = 0.72, 0.34
    w, h = 0.115, 0.13
    ref_nodes = [
        (0.02, "Референсные\nизображения", "2 описания в results"),
        (0.17, "VLM", "атрибутивные описания"),
        (0.32, "Sentence-BERT", "эмбеддинги 768-D"),
        (0.47, "Профиль цели", "μ, Σ; diagonal"),
    ]
    for index, (x, title, subtitle) in enumerate(ref_nodes):
        box(ax, (x, ref_y), w, h, title, color=COLORS["blue"], subtitle=subtitle)
        if index:
            arrow(ax, (x - 0.035, ref_y + h / 2), (x, ref_y + h / 2), color=COLORS["blue"])

    search_nodes = [
        (0.02, "Видеокадр", "0.5 кадра/с"),
        (0.17, "Grid proposer", "517 регионов"),
        (0.32, "CLIP", "визуальный top-5"),
        (0.47, "VLM + SBERT", "описание → 768-D"),
        (0.62, "Mahalanobis", "d ≤ 1.8"),
        (0.77, "Gate + WBF", "target_match + IoU"),
        (0.90, "Результат", "bbox, track_id, отчёт"),
    ]
    search_widths = [w, w, w, w, w, w, 0.09]
    for index, ((x, title, subtitle), width) in enumerate(zip(search_nodes, search_widths)):
        box(ax, (x, search_y), width, h, title, color=COLORS["green"], subtitle=subtitle)
        if index:
            arrow(ax, (x - 0.035, search_y + h / 2), (x, search_y + h / 2), color=COLORS["green"])

    arrow(
        ax,
        (0.527, ref_y),
        (0.677, search_y + h),
        color=COLORS["purple"],
        label="семантическое распределение",
    )

    alt_y = 0.08
    for x, label in ((0.17, "YOLOv8"), (0.29, "Grounding DINO"), (0.43, "DINOv2")):
        box(ax, (x, alt_y), 0.10, 0.08, label, color=COLORS["orange"], linestyle="--")
        arrow(
            ax,
            (x + 0.05, alt_y + 0.08),
            (0.225, search_y),
            color=COLORS["orange"],
            linestyle="--",
        )
    ax.text(
        0.17,
        0.025,
        "Альтернативные proposer-ы (в данном прогоне итоговые регионы имеют source=grid)",
        fontsize=8,
        color=COLORS["orange"],
    )

    ax.text(0.02, 0.91, "Формирование профиля", color=COLORS["blue"], weight="bold")
    ax.text(0.02, 0.54, "Обработка каждого выбранного кадра", color=COLORS["green"], weight="bold")
    ax.text(
        0.72,
        0.19,
        "CLIP и Sentence-BERT — разные пространства.\n"
        "Трекинг выполняется после решения по кадру\n"
        "и не изменяет frame_positive.",
        fontsize=8.2,
        color=COLORS["gray"],
        va="center",
    )
    add_source(fig, "Источник: фактический pipeline run_video_search_session / detect_on_image.")
    save_figure(fig, output_dir, "01_pipeline_architecture")


def plot_software_architecture(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Программная архитектура прототипа MOID", loc="left", weight="bold", pad=14)

    layers = [
        (
            0.80,
            "Интерфейс и оркестрация",
            COLORS["blue"],
            [
                (0.06, 0.18, "moid.cli", "CLI-команды"),
                (0.30, 0.24, "moid.session", "video/search sessions"),
                (0.60, 0.17, "moid.video", "сэмплинг кадров"),
                (0.82, 0.13, "MoidConfig", "YAML"),
            ],
        ),
        (
            0.57,
            "Ядро обработки кадра",
            COLORS["green"],
            [
                (0.06, 0.23, "pipeline.few_shot", "detect_on_image"),
                (0.36, 0.18, "moid.factory", "сборка компонентов"),
                (0.60, 0.16, "regions.*", "grid / YOLO / DINO"),
                (0.82, 0.13, "adapters.*", "VLM / CLIP / OCR"),
            ],
        ),
        (
            0.34,
            "Семантический скоринг",
            COLORS["purple"],
            [
                (0.06, 0.22, "mcd.embedding", "Sentence-BERT"),
                (0.35, 0.23, "mcd.modeling", "μ, Σ, Mahalanobis"),
                (0.65, 0.17, "moid.scoring", "gate + NMS/WBF"),
                (0.87, 0.09, "tracking", "IoU"),
            ],
        ),
        (
            0.11,
            "Артефакты и оценка",
            COLORS["orange"],
            [
                (0.06, 0.20, "moid.reporting", "JSON / MD / overlay"),
                (0.33, 0.18, "moid.metrics", "IoU / mAP"),
                (0.58, 0.19, "StageTimer", "latency / FPS"),
                (0.84, 0.13, "reports/", "артефакты"),
            ],
        ),
    ]
    for y, title, color, nodes in layers:
        ax.text(0.015, y + 0.105, title, color=color, fontsize=9, weight="bold", va="center")
        ax.plot([0.015, 0.98], [y + 0.08, y + 0.08], color=color, alpha=0.2, linewidth=1)
        for x, width, label, subtitle in nodes:
            box(ax, (x, y), width, 0.105, label, color=color, subtitle=subtitle)

    connectors = [
        ((0.39, 0.80), (0.175, 0.675)),
        ((0.69, 0.80), (0.175, 0.675)),
        ((0.475, 0.57), (0.44, 0.445)),
        ((0.68, 0.57), (0.175, 0.445)),
        ((0.90, 0.57), (0.735, 0.445)),
        ((0.48, 0.34), (0.16, 0.215)),
        ((0.735, 0.34), (0.42, 0.215)),
        ((0.915, 0.34), (0.675, 0.215)),
        ((0.16, 0.11), (0.905, 0.11)),
    ]
    for start, end in connectors:
        arrow(ax, start, end, color=COLORS["gray"])

    add_source(
        fig,
        "Источник: src/moid, src/mcd. Центральная точка повторного использования — pipeline.few_shot.detect_on_image.",
    )
    save_figure(fig, output_dir, "02_software_architecture")


def finite_distances(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["distance"] = pd.to_numeric(result["distance"], errors="coerce")
    result["threshold"] = pd.to_numeric(result["threshold"], errors="coerce")
    return result[np.isfinite(result["distance"])].copy()


def plot_distance_over_time(distances: pd.DataFrame, output_dir: Path) -> None:
    finite = finite_distances(distances)
    grouped = finite.groupby("timestamp")["distance"]
    summary = grouped.agg(["min", "median", "max"]).reset_index()
    threshold = float(finite["threshold"].dropna().median())

    fig, ax = plt.subplots(figsize=(12, 6.5))
    yes = finite[finite["target_match"].eq("yes")]
    no = finite[finite["target_match"].eq("no")]
    ax.scatter(
        yes["timestamp"],
        yes["distance"],
        color=COLORS["green"],
        alpha=0.55,
        s=34,
        label=f"target_match=yes (n={len(yes)})",
    )
    ax.scatter(
        no["timestamp"],
        no["distance"],
        color=COLORS["red"],
        marker="x",
        s=52,
        linewidth=1.5,
        label=f"target_match=no (n={len(no)})",
    )
    ax.plot(
        summary["timestamp"],
        summary["min"],
        color=COLORS["blue"],
        linewidth=2,
        marker="o",
        markersize=3.5,
        label="Минимум по кадру",
    )
    ax.axhline(threshold, color=COLORS["orange"], linestyle="--", linewidth=2, label=f"Порог τ={threshold:.1f}")
    ax.fill_between(
        summary["timestamp"],
        summary["min"],
        summary["max"],
        color=COLORS["blue"],
        alpha=0.08,
        label="Диапазон top-5",
    )
    ax.set(
        title="Семантическое расстояние по времени",
        xlabel="Время видео, с",
        ylabel="Расстояние Махаланобиса (меньше — ближе)",
        xticks=summary["timestamp"],
    )
    ax.legend(ncol=2, loc="upper left")
    ax.text(
        0.99,
        0.03,
        f"Конечных оценок: {len(finite)} из {len(distances)} регионов",
        transform=ax.transAxes,
        ha="right",
        color=COLORS["gray"],
        fontsize=8,
    )
    add_source(fig, "Источник: region_distances.csv; показаны только конечные distance после CLIP top-5 и VLM.")
    save_figure(fig, output_dir, "03_distance_over_time")


def detection_counts(payload: dict[str, Any]) -> dict[int, int]:
    return {
        int(frame["frame_index"]): len(frame.get("detections", []))
        for video in payload.get("videos", [])
        for frame in video.get("frames", [])
    }


def plot_candidate_funnel(
    distances: pd.DataFrame, payload: dict[str, Any], output_dir: Path
) -> None:
    finite = finite_distances(distances)
    totals = distances.groupby("frame_index").size()
    described = finite.groupby("frame_index").size()
    accepted = finite[finite["accepted"].astype(str).str.lower().eq("true")].groupby("frame_index").size()
    post = pd.Series(detection_counts(payload), dtype=float)
    frame_ids = sorted(int(value) for value in totals.index)
    x = np.arange(len(frame_ids))

    fig, ax = plt.subplots(figsize=(13, 7.1))
    width = 0.2
    series = [
        ("Grid-кандидаты", totals, COLORS["gray"]),
        ("После CLIP/VLM", described, COLORS["blue"]),
        ("Приняты Mahalanobis/gate", accepted, COLORS["green"]),
        ("После WBF", post, COLORS["orange"]),
    ]
    for offset, (label, values, color) in enumerate(series):
        heights = [float(values.get(frame_id, 0)) for frame_id in frame_ids]
        bars = ax.bar(x + (offset - 1.5) * width, heights, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=7, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(0.8, max(totals) * 1.8)
    ax.set_xticks(x, [str(frame_id) for frame_id in frame_ids])
    ax.set(
        title="Редукция кандидатов на этапах конвейера",
        xlabel="Индекс кадра",
        ylabel="Число регионов (логарифмическая шкала)",
    )
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    add_source(
        fig,
        "Источник: region_distances.csv и results.json; 13 кадров, шаг 60; на каждом кадре CLIP сокращает 517 регионов до 5.",
    )
    save_figure(fig, output_dir, "04_candidate_funnel")


def load_ground_truth(path: Path) -> tuple[dict[str, list[tuple[float, float, float, float]]], dict[str, tuple[int, int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = {
        int(item["id"]): (Path(item["file_name"]).stem, (int(item["width"]), int(item["height"])))
        for item in payload["images"]
    }
    boxes: dict[str, list[tuple[float, float, float, float]]] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in images or bool(annotation.get("iscrowd", False)):
            continue
        stem, size = images[image_id]
        x, y, width, height = (float(value) for value in annotation["bbox"])
        boxes.setdefault(stem, []).append((x, y, x + width, y + height))
        sizes[stem] = size
    return boxes, sizes


def iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    union += max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union else 0.0


def frame_predictions(payload: dict[str, Any]) -> dict[str, list[tuple[float, float, float, float]]]:
    result: dict[str, list[tuple[float, float, float, float]]] = {}
    for video in payload.get("videos", []):
        for frame in video.get("frames", []):
            stem = f"frame_{int(frame['frame_index']):08d}"
            result[stem] = [
                tuple(float(value) for value in detection["bbox"])
                for detection in frame.get("detections", [])
            ]
    return result


def best_iou_by_gt(
    ground_truth: dict[str, list[tuple[float, float, float, float]]],
    predictions: dict[str, list[tuple[float, float, float, float]]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for stem in sorted(set(ground_truth).intersection(predictions)):
        overlaps = [
            iou(prediction, target)
            for prediction in predictions[stem]
            for target in ground_truth[stem]
        ]
        result[stem] = max(overlaps, default=0.0)
    return result


def plot_localization_quality(
    payload: dict[str, Any],
    metrics: dict[str, Any],
    ground_truth: dict[str, list[tuple[float, float, float, float]]],
    output_dir: Path,
) -> dict[str, float]:
    overlaps = best_iou_by_gt(ground_truth, frame_predictions(payload))
    labels = [stem.removeprefix("frame_").lstrip("0") or "0" for stem in overlaps]
    values = list(overlaps.values())

    fig, (ax_iou, ax_metrics) = plt.subplots(
        1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.7, 1]}
    )
    bars = ax_iou.bar(labels, values, color=COLORS["blue"])
    ax_iou.axhline(0.5, color=COLORS["red"], linestyle="--", linewidth=1.8, label="IoU=0.5")
    ax_iou.set_ylim(0, 0.55)
    ax_iou.set(
        title="Лучший IoU с GT для размеченных кадров",
        xlabel="Индекс кадра",
        ylabel="Intersection over Union",
    )
    ax_iou.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax_iou.legend(loc="upper right")

    metric_names = ["Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"]
    metric_values = [
        float(metrics.get("precision", 0)),
        float(metrics.get("recall", 0)),
        float(metrics.get("f1", 0)),
        float(metrics.get("map_50", 0)),
        float(metrics.get("map_50_95", 0)),
    ]
    metric_bars = ax_metrics.barh(metric_names, metric_values, color=COLORS["red"])
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set(
        title="Детекционные метрики",
        xlabel="Значение метрики",
        ylabel="Метрика",
    )
    ax_metrics.bar_label(metric_bars, fmt="%.2f", padding=4)
    ax_metrics.text(
        0.5,
        0.12,
        f"TP={metrics.get('tp', 0)}   FP={metrics.get('fp', 0)}   FN={metrics.get('fn', 0)}",
        transform=ax_metrics.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color=COLORS["dark"],
    )
    fig.suptitle(
        "Семантическое принятие 13/13 кадров не обеспечивает точную локализацию",
        fontsize=14,
        weight="bold",
    )
    add_source(fig, "Источник: results.json, metrics.json и annotations/instances.json; operating point IoU=0.5.")
    save_figure(fig, output_dir, "05_localization_quality")
    return overlaps


def plot_runtime_profile(payload: dict[str, Any], output_dir: Path) -> None:
    performance = payload["performance"]
    stages = performance["stages"]
    names = [
        name
        for name in ("clip_filter", "vlm", "mahalanobis", "sbert", "postprocess", "proposals", "tracking")
        if name in stages
    ]
    labels = {
        "clip_filter": "CLIP-фильтр",
        "vlm": "VLM",
        "mahalanobis": "Mahalanobis",
        "sbert": "Sentence-BERT",
        "postprocess": "Постобработка",
        "proposals": "Генерация регионов",
        "tracking": "Трекинг",
    }
    means = np.array([float(stages[name]["mean_ms"]) / 1000 for name in names])
    p50 = np.array([float(stages[name]["p50_ms"]) / 1000 for name in names])
    p95 = np.array([float(stages[name]["p95_ms"]) / 1000 for name in names])
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    bars = ax.barh(y, means, color=COLORS["blue"], alpha=0.85, label="Среднее")
    ax.scatter(p50, y, color=COLORS["green"], marker="|", s=180, linewidth=2.2, label="p50")
    ax.scatter(p95, y, color=COLORS["orange"], marker="D", s=34, label="p95")
    ax.set_yticks(y, [labels[name] for name in names])
    ax.invert_yaxis()
    ax.set(
        title="Время выполнения этапов на один кадр",
        xlabel="Время, с",
        ylabel="Этап конвейера",
    )
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in means], padding=4, fontsize=8)
    ax.legend(loc="lower right")
    ax.text(
        0.99,
        0.93,
        f"Полное время: {performance['mean_frame_ms'] / 1000:.2f} с/кадр\n"
        f"Пропускная способность: {performance['throughput_fps']:.3f} FPS",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=COLORS["dark"],
        bbox={"facecolor": "white", "edgecolor": COLORS["light"], "pad": 6},
    )
    add_source(fig, "Источник: results.json → performance; 13 обработанных кадров, CPU для CLIP.")
    save_figure(fig, output_dir, "06_runtime_profile")


def find_frame(payload: dict[str, Any], frame_index: int) -> dict[str, Any] | None:
    for video in payload.get("videos", []):
        for frame in video.get("frames", []):
            if int(frame["frame_index"]) == frame_index:
                return frame
    return None


def plot_qualitative_overlay(
    report_dir: Path,
    payload: dict[str, Any],
    ground_truth: dict[str, list[tuple[float, float, float, float]]],
    overlaps: dict[str, float],
    output_dir: Path,
) -> None:
    available = [(stem, value) for stem, value in overlaps.items() if find_frame(payload, int(stem[-8:]))]
    if not available:
        return
    stem, best_overlap = max(available, key=lambda item: item[1])
    frame_index = int(stem[-8:])
    frame = find_frame(payload, frame_index)
    if frame is None or not frame.get("overlay"):
        return
    source = report_dir / frame["overlay"]
    if not source.exists():
        return

    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(4, image.width // 500)
    for target in ground_truth.get(stem, []):
        draw.rectangle(target, outline=(210, 35, 35), width=line_width)
    canvas = np.asarray(image)

    fig, ax = plt.subplots(figsize=(14, 7.9))
    ax.imshow(canvas)
    ax.axis("off")
    ax.set_title(
        f"Качественный пример локализации: кадр {frame_index}, лучший IoU={best_overlap:.3f}",
        loc="left",
        weight="bold",
    )
    ax.legend(
        handles=[
            Patch(facecolor="none", edgecolor=COLORS["green"], linewidth=2, label="Предсказанные боксы"),
            Patch(facecolor="none", edgecolor=COLORS["red"], linewidth=2, label="Ground truth"),
        ],
        loc="lower right",
    )
    add_source(fig, f"Источник: {frame['overlay']} + annotations/instances.json.")
    save_figure(fig, output_dir, "07_qualitative_localization")


def write_summary(
    output_dir: Path,
    distances: pd.DataFrame,
    payload: dict[str, Any],
    metrics: dict[str, Any],
    overlaps: dict[str, float],
) -> None:
    finite = finite_distances(distances)
    accepted = finite[finite["accepted"].astype(str).str.lower().eq("true")]
    rejected = finite[~finite.index.isin(accepted.index)]
    summary = {
        "frames_checked": int(payload["summary"]["frames_checked"]),
        "positive_frames": int(payload["summary"]["positive_frames"]),
        "requested_sample_fps": float(payload["sampling"]["requested_fps"]),
        "regions_total": int(len(distances)),
        "regions_per_frame": int(distances.groupby("frame_index").size().median()),
        "regions_with_finite_distance": int(len(finite)),
        "accepted_regions_before_wbf": int(len(accepted)),
        "detections_after_wbf": int(sum(detection_counts(payload).values())),
        "accepted_distance_min": float(accepted["distance"].min()),
        "accepted_distance_max": float(accepted["distance"].max()),
        "rejected_distance_min": float(rejected["distance"].min()),
        "rejected_distance_max": float(rejected["distance"].max()),
        "threshold": float(finite["threshold"].dropna().median()),
        "mean_frame_seconds": float(payload["performance"]["mean_frame_ms"]) / 1000,
        "throughput_fps": float(payload["performance"]["throughput_fps"]),
        "temporal_stability": payload.get("temporal_stability", {}),
        "metrics": {
            key: metrics.get(key)
            for key in ("precision", "recall", "f1", "map_50", "map_50_95", "tp", "fp", "fn")
        },
        "best_iou_by_gt_frame": overlaps,
        "evaluated_gt_frames": int(len(overlaps)),
    }
    (output_dir / "source_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.report.resolve()
    output_dir = args.output.resolve()
    configure_style()

    payload = json.loads((report_dir / "results.json").read_text(encoding="utf-8"))
    metrics_path = report_dir / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists()
        else payload.get("metrics", {})
    )
    distances = pd.read_csv(report_dir / "region_distances.csv")
    ground_truth, _ = load_ground_truth(args.annotations.resolve())

    plot_pipeline_architecture(output_dir)
    plot_software_architecture(output_dir)
    plot_distance_over_time(distances, output_dir)
    plot_candidate_funnel(distances, payload, output_dir)
    overlaps = plot_localization_quality(payload, metrics, ground_truth, output_dir)
    plot_runtime_profile(payload, output_dir)
    plot_qualitative_overlay(report_dir, payload, ground_truth, overlaps, output_dir)
    write_summary(output_dir, distances, payload, metrics, overlaps)

    generated = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    print(f"Generated {len(generated)} files in {output_dir}")
    for name in generated:
        print(f"  {name}")


if __name__ == "__main__":
    main()
