import json
from io import StringIO
from pathlib import Path

from PIL import Image

from moid.config import MoidConfig
from moid.pipeline.few_shot import FewShotResult
from moid.regions.base import BBox
from moid.scoring import ScoredBox
from moid.session import run_search_session, run_video_search_session


def test_search_session_noninteractive(tmp_path: Path, monkeypatch):
    refs = tmp_path / "refs"
    search = tmp_path / "search"
    refs.mkdir()
    search.mkdir()
    Image.new("RGB", (16, 16), (255, 0, 0)).save(refs / "a.png")
    (refs / "a.txt").write_text(
        "object: red cube; target_match: yes; category: object; color: red; vehicle_class: n/a",
        encoding="utf-8",
    )
    Image.new("RGB", (16, 16), (255, 0, 0)).save(search / "b.png")
    (search / "b.txt").write_text(
        "object: red cube; target_match: yes; category: object; color: red; vehicle_class: n/a",
        encoding="utf-8",
    )

    cfg = MoidConfig()
    cfg.adapters.vlm = "stub"
    cfg.adapters.llm = "stub"
    cfg.detector.sbert_model = "unused"
    cfg.grid.rows = 1
    cfg.grid.cols = 1
    cfg.grid.extra_scales = []
    cfg.decision.target_match_gate = False
    out = tmp_path / "reports"

    from tests.embedder_utils import LexicalEmbedder
    import moid.session as session_mod

    monkeypatch.setattr(session_mod, "build_embedder", lambda config, embedder=None: LexicalEmbedder())

    stdout = StringIO()
    report_dir = run_search_session(
        cfg,
        refs=str(refs),
        search=str(search),
        out=str(out),
        stub=True,
        no_refine=True,
        stdin=StringIO(""),
        stdout=stdout,
        interactive=False,
    )
    text = stdout.getvalue()
    assert "0 этап" in text
    assert "9 этап" in text
    assert (report_dir / "report.md").is_file()
    assert (report_dir / "results.json").is_file()


def test_video_session_uses_defaults_and_writes_all_artifacts(tmp_path: Path, monkeypatch):
    refs = tmp_path / "refs"
    videos = tmp_path / "videos"
    refs.mkdir()
    videos.mkdir()
    Image.new("RGB", (16, 16), (255, 0, 0)).save(refs / "ref.png")
    caption = "object: red cube; target_match: yes; category: object; color: red"
    (refs / "ref.txt").write_text(caption, encoding="utf-8")
    (videos / "b.mp4").touch()
    (videos / "a.mov").touch()

    cfg = MoidConfig()
    cfg.paths.refs = str(refs)
    cfg.paths.videos = str(videos)
    cfg.video.sample_fps = 2.0
    cfg.adapters.vlm = "stub"
    cfg.adapters.llm = "stub"
    cfg.detector.sbert_model = "unused"
    out = tmp_path / "reports"

    from tests.embedder_utils import LexicalEmbedder
    import moid.session as session_mod

    monkeypatch.setattr(session_mod, "build_embedder", lambda config, embedder=None: LexicalEmbedder())
    monkeypatch.setattr(
        session_mod,
        "extract_frames",
        lambda _path, _fps: iter([(3, 1.5, Image.new("RGB", (16, 16), (255, 0, 0)))]),
    )
    scored = ScoredBox(
        BBox(1, 1, 14, 14),
        distance=0.2,
        threshold=1.0,
        accepted=True,
        caption=caption,
        target_match="yes",
    )
    fake_result = FewShotResult(
        detections=[scored],
        all_regions=[scored],
        reference_captions=[caption],
        failed_regions=0,
        frame_positive=True,
    )
    monkeypatch.setattr(session_mod, "detect_on_image", lambda *args, **kwargs: fake_result)

    stdout = StringIO()
    report_dir = run_video_search_session(
        cfg,
        out=str(out),
        stub=True,
        stdin=StringIO("\nno\n\n\n"),
        stdout=stdout,
        interactive=True,
    )

    payload = json.loads((report_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["sampling"]["requested_fps"] == 2.0
    assert payload["summary"] == {
        "videos_processed": 2,
        "frames_checked": 2,
        "positive_frames": 2,
    }
    assert [video["file"] for video in payload["videos"]] == ["a.mov", "b.mp4"]
    assert all(video["frames"][0]["detections"][0]["bbox"] == [1, 1, 14, 14] for video in payload["videos"])
    assert len(list((report_dir / "overlays").rglob("*.png"))) == 2
    assert (report_dir / "best" / "overall_best.png").is_file()
    assert (report_dir / "report.md").is_file()
    assert "9 этап" in stdout.getvalue()
