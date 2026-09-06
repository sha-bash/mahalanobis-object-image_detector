from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from moid.config import load_config
from moid.factory import build_embedder, build_llm, build_ocr, build_vlm
from moid.pipeline.few_shot import run_few_shot
from moid.pipeline.zero_shot import run_zero_shot
from moid.session import run_search_session, run_video_search_session
from moid.video import process_video  # new import


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="moid", description="Few-shot / zero-shot object detector (VLM + mcd)")
    sub = parser.add_subparsers(dest="command", required=True)

    # few-shot
    fs = sub.add_parser("few-shot", help="Fit on reference images, localize object on a target image")
    fs.add_argument("--refs", required=True, help="Folder or single reference image")
    fs.add_argument("--target", required=True, help="Target image")
    fs.add_argument("--out", default="", help="Write JSON result to this path")
    fs.add_argument("--config", default="", help="YAML config path")
    fs.add_argument("--stub", action="store_true", help="Force StubVLM (sidecar .txt files)")

    # zero-shot
    zs = sub.add_parser("zero-shot", help="Normalize a text query and pick the closest image")
    zs.add_argument("--query", required=True, help="Free-text object description")
    zs.add_argument("--images", required=True, help="Folder of candidate images")
    zs.add_argument("--out", default="", help="Write JSON result to this path")
    zs.add_argument("--config", default="", help="YAML config path")
    zs.add_argument("--stub", action="store_true", help="Force StubVLM and StubLLM")
    zs.add_argument("--mode", choices=["pairwise", "one_shot"], default="", help="Override zero-shot mode")

    # interactive search
    search = sub.add_parser("search", help="Interactive few-shot search with optional description refine")
    search.add_argument("--refs", default="", help="Reference image folder (skip prompt)")
    search.add_argument("--search", default="", help="Search image folder (skip prompt)")
    search.add_argument("--out", default="", help="Report directory or parent")
    search.add_argument("--config", default="", help="YAML config path")
    search.add_argument("--stub", action="store_true", help="Force stub VLM/LLM")
    search.add_argument("--no-refine", action="store_true", help="Skip description clarification")
    search.add_argument("--non-interactive", action="store_true", help="Do not prompt; use flags/defaults")

    # video search (non-interactive)
    video = sub.add_parser("search-video", help="Search for reference object in a video file (non-interactive)")
    video.add_argument("--refs", required=True, help="Reference image folder or single image")
    video.add_argument("--video", required=True, help="Input video file")
    video.add_argument("--out", default="", help="Output JSON report path")
    video.add_argument("--config", default="", help="YAML config path")
    video.add_argument("--frame-step", type=int, default=10, help="Process every Nth frame (default: 10)")

    # video search (interactive)
    video_i = sub.add_parser("search-video-i", help="Interactive video search: ref → hint → video → detect")
    video_i.add_argument("--refs", default="", help="Reference image (skip prompt)")
    video_i.add_argument("--video", default="", help="Video file (skip prompt)")
    video_i.add_argument("--out", default="", help="Report directory")
    video_i.add_argument("--config", default="", help="YAML config path")
    video_i.add_argument("--frame-step", type=int, default=10, help="Process every Nth frame (default: 10)")
    video_i.add_argument("--stub", action="store_true", help="Force stub VLM/LLM")
    video_i.add_argument("--no-refine", action="store_true", help="Skip description clarification")
    video_i.add_argument("--non-interactive", action="store_true", help="Do not prompt; use flags/defaults")

    args = parser.parse_args(argv)
    cfg = load_config(args.config or None)

    if args.command == "zero-shot" and args.mode:
        cfg.zero_shot.mode = args.mode  # type: ignore[assignment]

    if args.command == "search":
        interactive = (not args.non_interactive) and sys.stdin.isatty()
        run_search_session(
            cfg,
            refs=args.refs or None,
            search=args.search or None,
            out=args.out or None,
            stub=args.stub,
            no_refine=args.no_refine or args.non_interactive,
            stdin=sys.stdin,
            stdout=sys.stdout,
            interactive=interactive and not args.non_interactive,
        )
        return 0

    if args.command == "search-video-i":
        interactive = (not args.non_interactive) and sys.stdin.isatty()
        out_path = run_video_search_session(
            cfg,
            refs=args.refs or None,
            video=args.video or None,
            out=args.out or None,
            stub=args.stub,
            no_refine=args.no_refine or args.non_interactive,
            frame_step=args.frame_step,
            stdin=sys.stdin,
            stdout=sys.stdout,
            interactive=interactive and not args.non_interactive,
        )
        if out_path:
            print(f"Отчёт: {out_path}")
        return 0

    if args.command == "few-shot":
        result = run_few_shot(
            refs=args.refs,
            target=args.target,
            vlm=build_vlm(cfg, stub=args.stub),
            embedder=build_embedder(cfg),
            config=cfg,
            ocr=build_ocr(cfg),
        )
        payload = result.to_dict()

    elif args.command == "zero-shot":
        result = run_zero_shot(
            query=args.query,
            images=args.images,
            vlm=build_vlm(cfg, stub=args.stub),
            llm=build_llm(cfg, stub=args.stub),
            embedder=build_embedder(cfg),
            config=cfg,
        )
        payload = result.to_dict()

    elif args.command == "search-video":
        # process_video handles all component creation and returns list of per-frame results
        results = process_video(
            video_path=args.video,
            refs=args.refs,
            config=cfg,
            frame_step=args.frame_step,
            output_json=args.out if args.out else None,
        )
        # If output_json was provided, process_video saved the report already
        if args.out:
            print(args.out)
        else:
            # Print to stdout if no output file given
            print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    else:
        parser.error("Unknown command")

    # For few-shot and zero-shot, save or print JSON
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(args.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())