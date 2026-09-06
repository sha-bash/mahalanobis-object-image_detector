from __future__ import annotations

from pathlib import Path
from typing import Any, TextIO

from moid.config import MoidConfig
from moid.factory import build_embedder, build_llm, build_ocr, build_proposer, build_vlm
from moid.identity import apply_profile_to_captions, propose_refine_questions, refine_profile
from moid.pipeline.few_shot import caption_references, detect_on_image, fit_detector, list_images
from moid.reporting import new_report_dir, save_overlay, write_run_dir


def _ask(prompt: str, stdin: TextIO, stdout: TextIO, default: str = "") -> str:
    stdout.write(prompt)
    stdout.flush()
    line = stdin.readline()
    if line == "":
        return default
    text = line.strip()
    return text if text else default


def _yes(value: str) -> bool:
    return value.strip().lower() in {"y", "yes", "д", "да"}


def run_search_session(
    config: MoidConfig,
    *,
    refs: str | None = None,
    search: str | None = None,
    out: str | None = None,
    stub: bool = False,
    no_refine: bool = False,
    stdin: TextIO,
    stdout: TextIO,
    interactive: bool = True,
) -> Path:
    vlm = build_vlm(config, stub=stub)
    llm = build_llm(config, stub=stub)
    ocr = build_ocr(config)
    embedder = build_embedder(config)

    default_refs = refs or config.paths.refs
    stdout.write("0 этап: укажите директорию референсных фото\n")
    refs_dir = default_refs
    if interactive and refs is None:
        refs_dir = _ask(f"[{default_refs}]: ", stdin, stdout, default_refs)
    stdout.write(f"Референсы: {refs_dir}\n")

    paths = list_images(refs_dir)
    stdout.write(f"1 этап: обработка референсных фото ({len(paths)} файлов)\n")
    for path in paths:
        stdout.write(f"  - {path.name}\n")

    stdout.write("2 этап: описание изображения на референсах\n")
    captions, profile = caption_references(refs_dir, vlm, config)
    stdout.write(f"Профиль: {profile.object} ({profile.domain})\n")
    stdout.write(f"{profile.as_line()}\n")

    clarification = ""
    if not no_refine and interactive:
        stdout.write("3 этап: желаете дополнить описание? yes/no\n")
        answer = _ask("> ", stdin, stdout, "no")
        if _yes(answer):
            questions = propose_refine_questions(profile, llm)
            if questions:
                stdout.write("Вопросы для уточнения:\n")
                for i, q in enumerate(questions, 1):
                    stdout.write(f"  {i}. {q}\n")
            stdout.write("4 этап: введите уточнение\n")
            clarification = _ask("> ", stdin, stdout, "")
            stdout.write("5 этап: уточнение получено\n")
            if clarification:
                profile = refine_profile(profile, clarification, llm)
                captions = apply_profile_to_captions(captions, profile)
                stdout.write(f"Обновлённый профиль: {profile.as_line()}\n")
        else:
            stdout.write("Уточнение пропущено.\n")
    else:
        stdout.write("3–5 этапы: уточнение пропущено.\n")

    default_search = search or config.paths.search
    stdout.write("6 этап: передайте директорию с фото для поиска\n")
    search_dir = default_search
    if interactive and search is None:
        search_dir = _ask(f"[{default_search}]: ", stdin, stdout, default_search)
    stdout.write(f"Поиск в: {search_dir}\n")

    stdout.write("7 этап: поиск целевого объекта на изображениях\n")
    detector = fit_detector(captions, embedder, config)
    prompt = config.regions.text_prompt or profile.object
    proposer = build_proposer(config, text_prompt=prompt)
    search_paths = list_images(search_dir)
    per_image: list[dict[str, Any]] = []
    results = []
    for path in search_paths:
        result = detect_on_image(
            path,
            detector,
            vlm,
            config=config,
            proposer=proposer,
            ocr=ocr,
            profile=profile,
            reference_captions=captions,
        )
        results.append((path, result))
        best = None
        if result.detections:
            best = result.detections[0]
        elif result.all_regions:
            best = min((s for s in result.all_regions if not s.failed), key=lambda s: s.distance, default=None)
        stdout.write(
            f"  {path.name}: {'найдено' if result.frame_positive else 'не найдено'}"
            + (f" (d={best.distance:.3f})" if best is not None else "")
            + "\n"
        )
        per_image.append(
            {
                "file": path.name,
                "frame_positive": result.frame_positive,
                "best_distance": None if best is None else best.distance,
                "best_match": None if best is None else best.target_match,
                "caption": None if best is None else best.caption,
                "detections": result.to_dict()["detections"],
            }
        )

    stdout.write("8 этап: подготовка отчета (LLM формирует отчет по эксперименту)\n")
    slug = profile.object.replace(" ", "-")[:24] or "search"
    out_root = Path(out) if out else Path(config.paths.reports)
    if out and Path(out).suffix:
        out_dir = Path(out).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        report_name = Path(out).stem
    else:
        out_dir = new_report_dir(out_root, slug)
        report_name = out_dir.name
    overlays = out_dir / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    for path, result in results:
        save_overlay(path, result, overlays / path.name)
    write_run_dir(out_dir, profile=profile, clarification=clarification, per_image=per_image, llm=llm)
    stdout.write(f"9 этап: отчет {report_name} расположен в директории отчетов {out_dir}\n")
    return out_dir
