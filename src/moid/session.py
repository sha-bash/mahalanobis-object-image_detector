from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from moid.config import MoidConfig
from moid.factory import build_embedder, build_llm, build_ocr, build_proposer, build_vlm, build_visual_encoder
from moid.identity import apply_profile_to_captions, propose_refine_questions, refine_profile
from moid.metrics import compute_metrics, frame_stem, load_ground_truth, predictions_from_video_payload
from moid.performance import StageTimer
from moid.pipeline.few_shot import (
    caption_references,
    caption_references_with_paths,
    detect_on_image,
    filter_reference_paths,
    fit_detector,
    list_images,
)
from moid.reporting import new_report_dir, save_overlay, write_run_dir, write_video_run_dir
from moid.video import extract_frames, list_videos
from moid.tracking import CameraMotionEstimator, Tracker


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


def _run_video_search_session_legacy(
    config: MoidConfig,
    *,
    refs: str | None = None,
    video: str | None = None,
    out: str | None = None,
    stub: bool = False,
    no_refine: bool = False,
    frame_step: int = 10,
    stdin: TextIO,
    stdout: TextIO,
    interactive: bool = True,
) -> Path:
    """Interactive search: reference image → optional text hint → video → detection."""
    vlm = build_vlm(config, stub=stub)
    llm = build_llm(config, stub=stub)
    embedder = build_embedder(config)
    visual_encoder = build_visual_encoder(config)

    # 0. Ask for reference image
    default_refs = refs or config.paths.refs
    stdout.write("0 этап: укажите путь к референсному фото (изображение или папка)\n")
    refs_path = default_refs
    if interactive and refs is None:
        refs_path = _ask(f"[{default_refs}]: ", stdin, stdout, default_refs)
    stdout.write(f"Референсы: {refs_path}\n")

    # 1. Caption references and build profile
    paths = list_images(refs_path)
    stdout.write(f"1 этап: обработка референсных фото ({len(paths)} файлов)\n")
    for path in paths:
        stdout.write(f"  - {path.name}\n")

    stdout.write("2 этап: описание изображения на референсах\n")
    captions, profile = caption_references(refs_path, vlm, config)
    stdout.write(f"Профиль: {profile.object} ({profile.domain})\n")
    stdout.write(f"{profile.as_line()}\n")

    # 2. Optional text clarification
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

    # 3. Ask for video file
    stdout.write("6 этап: укажите путь к видеофайлу\n")
    video_path = video
    if interactive and video is None:
        video_path = _ask("Путь к видео: ", stdin, stdout, "")
    if not video_path:
        stdout.write("Путь к видео не указан. Завершение.\n")
        return Path()
    video_path = Path(video_path)
    if not video_path.exists():
        stdout.write(f"Файл не найден: {video_path}\n")
        return Path()
    stdout.write(f"Видео: {video_path}\n")

    # 4. Search in video
    stdout.write("7 этап: поиск целевого объекта на видео\n")
    stdout.write(f"   frame_step={frame_step} (обрабатывается каждый {frame_step}-й кадр)\n")

    detector = fit_detector(captions, embedder, config)
    prompt = config.regions.text_prompt or profile.object
    proposer = build_proposer(config, text_prompt=prompt)
    ocr = build_ocr(config)

    # Compute reference visual embeddings if visual encoder is available
    ref_visual_embeddings = None
    if visual_encoder is not None:
        stdout.write("   Вычисление визуальных эмбеддингов референсов...\n")
        from moid.pipeline.few_shot import list_images as _list_images
        from moid.adapters.vlm import load_image
        import numpy as np
        ref_images = [load_image(p) for p in _list_images(refs_path)]
        ref_visual_embeddings = [visual_encoder.encode(img) for img in ref_images]
        ref_visual_embeddings = np.stack(ref_visual_embeddings, axis=0)
        stdout.write(f"   Визуальные эмбеддинги: {ref_visual_embeddings.shape}\n")

    # Run per-frame detection
    results = []
    for frame_idx, timestamp, pil_image in extract_frames(video_path, frame_step):
        frame_result = detect_on_image(
            pil_image,
            detector,
            vlm,
            config=config,
            proposer=proposer,
            ocr=ocr,
            profile=profile,
            reference_captions=captions,
            visual_encoder=visual_encoder,
            reference_visual_embeddings=ref_visual_embeddings,
        )
        results.append({
            "frame_index": frame_idx,
            "timestamp": timestamp,
            "frame_positive": frame_result.frame_positive,
            "detections": [s.to_dict() for s in frame_result.detections],
            "all_regions": [s.to_dict() for s in frame_result.all_regions],
        })

    # 5. Report
    positive_frames = [r for r in results if r["frame_positive"]]
    stdout.write("\n8 этап: результаты\n")
    stdout.write(f"   Всего кадров: {len(results)}\n")
    stdout.write(f"   Кадров с обнаружением: {len(positive_frames)}\n")
    for r in results[:10]:  # show first 10 for brevity
        status = "НАЙДЕН" if r["frame_positive"] else "—"
        stdout.write(f"   Кадр {r['frame_index']} (t={r['timestamp']:.1f}s): {status}\n")
    if len(results) > 10:
        stdout.write(f"   ... и ещё {len(results) - 10} кадров\n")

    # Save overlays and report
    slug = profile.object.replace(" ", "-")[:24] or "video-search"
    out_root = Path(out) if out and not Path(out).suffix else Path(out or config.paths.reports)
    if out and Path(out).suffix:
        out_dir = Path(out).parent
        report_name = Path(out).stem
    else:
        out_dir = new_report_dir(out_root, slug)
        report_name = out_dir.name
    overlays = out_dir / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)

    # Save per-frame overlays (only positive frames)
    frame_map = {}
    for frame_idx, timestamp, pil_image in extract_frames(video_path, frame_step):
        frame_map[frame_idx] = pil_image

    for r in results:
        if r["frame_positive"] and r["frame_index"] in frame_map:
            # Reconstruct FewShotResult for overlay
            from moid.pipeline.few_shot import FewShotResult
            from moid.scoring import ScoredBox
            from moid.regions.base import BBox
            detections = []
            all_regions = []
            for d in r["detections"]:
                bbox = tuple(d["bbox"])
                box = BBox(bbox[0], bbox[1], bbox[2], bbox[3])
                detections.append(ScoredBox(
                    box=box,
                    distance=d["distance"],
                    threshold=d["threshold"],
                    accepted=d["accepted"],
                    caption=d["caption"],
                    failed=d["failed"],
                    target_match=d.get("target_match"),
                    raw_distance=d.get("raw_distance"),
                    ocr_text=d.get("ocr_text"),
                    visualization_only=d.get("visualization_only", False),
                ))
            for a in r["all_regions"]:
                bbox = tuple(a["bbox"])
                box = BBox(bbox[0], bbox[1], bbox[2], bbox[3])
                all_regions.append(ScoredBox(
                    box=box,
                    distance=a["distance"],
                    threshold=a["threshold"],
                    accepted=a["accepted"],
                    caption=a["caption"],
                    failed=a["failed"],
                    target_match=a.get("target_match"),
                    raw_distance=a.get("raw_distance"),
                    ocr_text=a.get("ocr_text"),
                    visualization_only=a.get("visualization_only", False),
                ))
            fake_result = FewShotResult(
                detections=detections,
                all_regions=all_regions,
                reference_captions=captions,
                failed_regions=sum(1 for s in all_regions if s.failed),
                frame_positive=r["frame_positive"],
            )
            frame_img = frame_map[r["frame_index"]]
            save_overlay(frame_img, fake_result, overlays / f"frame_{r['frame_index']}.png")

    per_image = results
    write_run_dir(out_dir, profile=profile, clarification=clarification, per_image=per_image, llm=llm)
    stdout.write(f"9 этап: отчет {report_name} расположен в {out_dir}\n")
    stdout.write(f"   Оверлеи кадров: {overlays}\n")
    return out_dir


def run_video_search_session(
    config: MoidConfig,
    *,
    refs: str | None = None,
    video: str | None = None,
    out: str | None = None,
    stub: bool = False,
    no_refine: bool = False,
    sample_fps: float | None = None,
    stdin: TextIO,
    stdout: TextIO,
    interactive: bool = True,
) -> Path:
    """Build an object profile and search for it in every video in a folder."""
    vlm = build_vlm(config, stub=stub)
    llm = build_llm(config, stub=stub)
    embedder = build_embedder(config)
    visual_encoder = build_visual_encoder(config)

    default_refs = refs or config.paths.refs
    stdout.write("0 этап: укажите директорию референсных фото\n")
    refs_path = default_refs
    if interactive and refs is None:
        refs_path = _ask(f"[{default_refs}]: ", stdin, stdout, default_refs)
    stdout.write(f"Референсы: {refs_path}\n")

    reference_paths = filter_reference_paths(list_images(refs_path), config)
    stdout.write(f"1 этап: обработка референсных фото ({len(reference_paths)} файлов)\n")
    for path in reference_paths:
        stdout.write(f"  - {path.name}\n")
    stdout.write("2 этап: формирование профиля визуальным энкодером и VLM\n")
    captions, profile, reference_paths = caption_references_with_paths(
        refs_path, vlm, config
    )
    stdout.write(f"После фильтрации использовано референсов: {len(captions)}\n")
    stdout.write(f"Профиль: {profile.object} ({profile.domain})\n")
    stdout.write(f"{profile.as_line()}\n")

    clarification = ""
    if not no_refine and interactive:
        stdout.write("3 этап: желаете дополнить описание? yes/no\n")
        if _yes(_ask("> ", stdin, stdout, "no")):
            questions = propose_refine_questions(profile, llm)
            if questions:
                stdout.write("Вопросы для уточнения:\n")
                for index, question in enumerate(questions, 1):
                    stdout.write(f"  {index}. {question}\n")
            stdout.write("4 этап: введите уточнение\n")
            clarification = _ask("> ", stdin, stdout, "")
            if clarification:
                profile = refine_profile(profile, clarification, llm)
                captions = apply_profile_to_captions(captions, profile)
                stdout.write(f"Обновлённый профиль: {profile.as_line()}\n")
            stdout.write("5 этап: уточнение обработано\n")
        else:
            stdout.write("4–5 этапы: уточнение пропущено.\n")
    else:
        stdout.write("3–5 этапы: уточнение пропущено.\n")

    default_videos = video or config.paths.videos
    stdout.write("6 этап: укажите директорию с видео\n")
    videos_path = default_videos
    if interactive and video is None:
        videos_path = _ask(f"[{default_videos}]: ", stdin, stdout, default_videos)
    video_paths = list_videos(videos_path, config.video.extensions)
    stdout.write(f"Видео: {videos_path} ({len(video_paths)} файлов)\n")

    requested_fps = sample_fps if sample_fps is not None else config.video.sample_fps
    stdout.write("7 этап: укажите частоту проверки кадров (кадров/с)\n")
    if interactive and sample_fps is None:
        raw_fps = _ask(f"[{requested_fps:g}]: ", stdin, stdout, f"{requested_fps:g}")
        try:
            requested_fps = float(raw_fps)
        except ValueError as exc:
            raise ValueError("Частота проверки должна быть числом") from exc
    if requested_fps <= 0:
        raise ValueError("Частота проверки должна быть больше нуля")
    stdout.write(f"Частота проверки: {requested_fps:g} кадров/с\n")

    detector = fit_detector(captions, embedder, config)
    proposer = build_proposer(config, text_prompt=config.regions.text_prompt or profile.object)
    ocr = build_ocr(config)
    reference_visual_embeddings = None
    if visual_encoder is not None:
        from moid.adapters.vlm import load_image

        stdout.write("Вычисление визуальных эмбеддингов референсов...\n")
        reference_visual_embeddings = np.stack(
            [visual_encoder.encode(load_image(path)) for path in reference_paths],
            axis=0,
        )

    out_dir = new_report_dir(
        Path(out) if out else Path(config.paths.reports),
        profile.object.replace(" ", "-")[:24] or "video-search",
    )
    overlays_dir = out_dir / "overlays"
    best_dir = out_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    stdout.write("8 этап: поиск целевого объекта на видео\n")
    video_payloads: list[dict[str, Any]] = []
    total_frames = 0
    total_positive = 0
    overall_best: tuple[float, Any, Any, Path, int] | None = None
    performance = StageTimer(enabled=config.performance.enabled)
    image_sizes: dict[str, tuple[int, int]] = {}
    stability_by_video: dict[str, dict[str, float | int]] = {}

    for video_path in video_paths:
        safe_stem = "".join(
            char if char.isalnum() or char in "-_" else "-" for char in video_path.stem
        ) or "video"
        frame_rows: list[dict[str, Any]] = []
        positive_count = 0
        video_best: tuple[float, Any, Any, int] | None = None
        tracker = (
            Tracker(config.tracking.iou_threshold, config.tracking.max_misses)
            if config.tracking.enabled
            else None
        )
        motion = CameraMotionEstimator(config.tracking.camera_compensation)
        video_image_size: tuple[int, int] | None = None
        stdout.write(f"  - {video_path.name}\n")

        for frame_index, timestamp, image in extract_frames(video_path, requested_fps):
            frame_timer = StageTimer(enabled=config.performance.enabled)
            transform = motion.update(image)
            frame_result = detect_on_image(
                image,
                detector,
                vlm,
                config=config,
                proposer=proposer,
                ocr=ocr,
                profile=profile,
                reference_captions=captions,
                visual_encoder=visual_encoder,
                reference_visual_embeddings=reference_visual_embeddings,
                timer=frame_timer,
            )
            if tracker is not None:
                with frame_timer.measure("tracking"):
                    tracker.update(frame_result.detections, transform)
            performance.merge(frame_timer)
            video_image_size = image.size
            image_sizes[frame_stem(frame_index)] = image.size
            result_dict = frame_result.to_dict()
            overlay = None
            if frame_result.frame_positive:
                positive_count += 1
                overlay_file = overlays_dir / safe_stem / f"frame_{frame_index:08d}.png"
                save_overlay(image, frame_result, overlay_file)
                overlay = overlay_file.relative_to(out_dir).as_posix()
                score = min(
                    (box.distance for box in frame_result.detections if not box.failed),
                    default=float("inf"),
                )
                if video_best is None or score < video_best[0]:
                    video_best = (score, image.copy(), frame_result, frame_index)
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "frame_positive": frame_result.frame_positive,
                    "overlay": overlay,
                    "detections": result_dict["detections"],
                    "all_regions": result_dict["all_regions"],
                }
            )

        if tracker is not None and video_image_size is not None:
            stability_by_video[video_path.name] = tracker.stability(video_image_size)

        best_overlay = None
        if video_best is not None:
            score, best_image, best_result, best_index = video_best
            best_file = best_dir / f"{safe_stem}_frame_{best_index:08d}.png"
            save_overlay(best_image, best_result, best_file)
            best_overlay = best_file.relative_to(out_dir).as_posix()
            if overall_best is None or score < overall_best[0]:
                overall_best = (score, best_image, best_result, video_path, best_index)

        total_frames += len(frame_rows)
        total_positive += positive_count
        video_payloads.append(
            {
                "file": video_path.name,
                "path": str(video_path),
                "summary": {
                    "frames_checked": len(frame_rows),
                    "positive_frames": positive_count,
                },
                "best_frame_overlay": best_overlay,
                "frames": frame_rows,
            }
        )

    overall_best_overlay = None
    overall_best_source = None
    if overall_best is not None:
        _, image, result, source_video, frame_index = overall_best
        overall_file = best_dir / "overall_best.png"
        save_overlay(image, result, overall_file)
        overall_best_overlay = overall_file.relative_to(out_dir).as_posix()
        overall_best_source = {"video": source_video.name, "frame_index": frame_index}

    payload = {
        "schema_version": 1,
        "run_type": "video_search",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "references_path": str(refs_path),
        "videos_path": str(videos_path),
        "sampling": {"requested_fps": requested_fps},
        "profile": profile.fields,
        "domain": profile.domain,
        "clarification": clarification,
        "reference_captions": captions,
        "summary": {
            "videos_processed": len(video_paths),
            "frames_checked": total_frames,
            "positive_frames": total_positive,
        },
        "best_frame_overlay": overall_best_overlay,
        "best_frame_source": overall_best_source,
        "videos": video_payloads,
    }
    if config.performance.enabled:
        payload["performance"] = performance.summary()
    if stability_by_video:
        payload["temporal_stability"] = stability_by_video
    _write_region_distances(out_dir / "region_distances.csv", payload)
    if config.evaluation.gt_annotations:
        ground_truth, categories = load_ground_truth(
            config.evaluation.gt_annotations,
            image_sizes=image_sizes,
        )
        processed_stems = set(image_sizes)
        ground_truth = [item for item in ground_truth if item.frame in processed_stems]
        if not ground_truth:
            raise ValueError(
                "GT contains no annotations matching processed frame_XXXXXXXX stems"
            )
        target_terms = {
            profile.object.casefold(),
            *[term.casefold() for term in config.references.allowed_terms],
        }
        category_id = next(
            (
                identifier
                for identifier, name in categories.items()
                if any(term and term in name.casefold() for term in target_terms)
            ),
            next(iter(categories)),
        )
        ground_truth = [item for item in ground_truth if item.category_id == category_id]
        predictions = predictions_from_video_payload(payload, category_id=category_id)
        metric_result = compute_metrics(ground_truth, predictions)
        payload["evaluation"] = {
            "gt_annotations": config.evaluation.gt_annotations,
            "categories": categories,
        }
        payload["metrics"] = metric_result.to_dict()
        if config.evaluation.save_metrics:
            import json

            (out_dir / "metrics.json").write_text(
                json.dumps(metric_result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (out_dir / "pr_curve.json").write_text(
                json.dumps(metric_result.pr_curve, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    write_video_run_dir(
        out_dir,
        payload=payload,
        profile=profile,
        clarification=clarification,
        llm=llm,
    )
    stdout.write("9 этап: JSON, LLM-отчёт и кадры с рамками сохранены\n")
    stdout.write(f"Директория отчёта: {out_dir}\n")
    return out_dir


def _write_region_distances(path: Path, payload: dict[str, Any]) -> None:
    columns = [
        "video",
        "frame_index",
        "timestamp",
        "x1",
        "y1",
        "x2",
        "y2",
        "distance",
        "threshold",
        "confidence",
        "accepted",
        "target_match",
        "proposal_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for video in payload.get("videos", []):
            for frame in video.get("frames", []):
                for region in frame.get("all_regions", []):
                    box = region.get("bbox", [None] * 4)
                    writer.writerow(
                        {
                            "video": video.get("file"),
                            "frame_index": frame.get("frame_index"),
                            "timestamp": frame.get("timestamp"),
                            "x1": box[0],
                            "y1": box[1],
                            "x2": box[2],
                            "y2": box[3],
                            "distance": region.get("distance"),
                            "threshold": region.get("threshold"),
                            "confidence": region.get("confidence"),
                            "accepted": region.get("accepted"),
                            "target_match": region.get("target_match"),
                            "proposal_source": region.get("proposal_source"),
                        }
                    )
