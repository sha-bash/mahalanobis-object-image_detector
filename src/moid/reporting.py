from __future__ import annotations

import json
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


def save_overlay(image: str | Path | Image.Image, result: FewShotResult, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    boxes = result.detections or [s for s in result.all_regions if not s.failed][:1]
    draw_overlays(load_image(image), boxes, dest)
