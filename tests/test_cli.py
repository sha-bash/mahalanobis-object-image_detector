import moid.cli as cli_mod
from moid.cli import main


def test_cli_help():
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0


def test_cli_search_help():
    try:
        main(["search", "--help"])
    except SystemExit as e:
        assert e.code == 0


def test_cli_search_video_dispatches_unified_session(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cli_mod,
        "run_video_search_session",
        lambda config, **kwargs: captured.update(kwargs),
    )

    assert main(
        [
            "search-video",
            "--refs",
            "refs",
            "--video",
            "videos",
            "--sample-fps",
            "2.5",
            "--non-interactive",
        ]
    ) == 0
    assert captured["refs"] == "refs"
    assert captured["video"] == "videos"
    assert captured["sample_fps"] == 2.5
    assert captured["interactive"] is False
