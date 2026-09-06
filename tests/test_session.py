from io import StringIO
from pathlib import Path

from PIL import Image

from moid.config import MoidConfig
from moid.session import run_search_session


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
