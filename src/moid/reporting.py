from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from moid.adapters.vlm import load_image
from moid.identity import IdentityProfile
from moid.pipeline.few_shot import FewShotResult
from moid.prompts import REPORT_PROMPT
from moid.viz import draw_overlays


def new_report_dir(root: str | Path, slug: str = "search") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in slug)[:40] or "search"
    path = Path(root) / f"{stamp}_{safe}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "overlays").mkdir(exist_ok=True)
    return path


def template_report(
    profile: IdentityProfile,
    clarification: str,
    per_image: list[dict[str, Any]],
) -> str:
    lines = [
        "# Отчёт поиска",
        "",
        f"- Объект: **{profile.object}**",
        f"- Домен: {profile.domain}",
        f"- Уточнение пользователя: {clarification.strip() or 'нет'}",
        "",
        "## Кадры",
        "",
        "| Файл | Найден | Лучшая дистанция | target_match |",
        "|---|---|---|---|",
    ]
    for row in per_image:
        found = "да" if row.get("frame_positive") else "нет"
        dist = row.get("best_distance")
        dist_s = f"{dist:.3f}" if isinstance(dist, (int, float)) else "—"
        lines.append(
            f"| `{row.get('file', '')}` | {found} | {dist_s} | {row.get('best_match', '—')} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_run_dir(
    out_dir: Path,
    *,
    profile: IdentityProfile,
    clarification: str,
    per_image: list[dict[str, Any]],
    llm: Any | None = None,
) -> Path:
    payload = {
        "profile": profile.fields,
        "domain": profile.domain,
        "clarification": clarification,
        "captions": profile.captions,
        "images": per_image,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = ""
    if llm is not None:
        prompt = REPORT_PROMPT.format(
            profile=profile.as_line(),
            clarification=clarification or "(none)",
            results=json.dumps(payload, ensure_ascii=False)[:8000],
        )
        markdown = (llm.complete(prompt) if hasattr(llm, "complete") else "") or ""
    if not markdown.strip():
        markdown = template_report(profile, clarification, per_image)
    report_path = out_dir / "report.md"
    report_path.write_text(markdown, encoding="utf-8")
    return report_path


def template_video_report(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Отчёт поиска в видео",
        "",
        f"- Объект: **{payload.get('profile', {}).get('object', 'unknown')}**",
        f"- Проверено видео: {summary.get('videos_processed', 0)}",
        f"- Проверено кадров: {summary.get('frames_checked', 0)}",
        f"- Кадров с объектом: {summary.get('positive_frames', 0)}",
        "",
        "## Видео",
        "",
    ]
    for video in payload.get("videos", []):
        stats = video.get("summary", {})
        lines.append(
            f"- `{video.get('file', '')}`: "
            f"{stats.get('positive_frames', 0)} из {stats.get('frames_checked', 0)} кадров, "
            f"лучший кадр: {video.get('best_frame_overlay') or 'нет'}"
        )
    lines.append("")
    return "\n".join(lines)


def write_video_run_dir(
    out_dir: Path,
    *,
    payload: dict[str, Any],
    profile: IdentityProfile,
    clarification: str,
    llm: Any | None = None,
) -> Path:
    """Write the canonical video JSON and an LLM-generated markdown report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_payload = _json_safe(payload)
    json_text = json.dumps(safe_payload, ensure_ascii=False, indent=2, allow_nan=False)
    (out_dir / "results.json").write_text(json_text, encoding="utf-8")
    markdown = ""
    if llm is not None:
        prompt = REPORT_PROMPT.format(
            profile=profile.as_line(),
            clarification=clarification or "(none)",
            results=json_text[:32000],
        )
        markdown = (llm.complete(prompt) if hasattr(llm, "complete") else "") or ""
    if not markdown.strip():
        markdown = template_video_report(safe_payload)
    report_path = out_dir / "report.md"
    report_path.write_text(markdown, encoding="utf-8")
    return report_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_overlay(image: str | Path | Image.Image, result: FewShotResult, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    boxes = result.detections or [s for s in result.all_regions if not s.failed][:1]
    draw_overlays(load_image(image), boxes, dest)
