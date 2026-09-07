import json
from pathlib import Path

from moid.identity import build_identity_profile
from moid.reporting import write_video_run_dir


def test_write_video_run_dir_uses_video_fallback_template(tmp_path: Path):
    profile = build_identity_profile(["object: red truck; color: red; target_match: yes"])
    payload = {
        "profile": profile.fields,
        "summary": {"videos_processed": 1, "frames_checked": 3, "positive_frames": 1},
        "videos": [
            {
                "file": "sample.mp4",
                "summary": {"frames_checked": 3, "positive_frames": 1},
                "best_frame_overlay": "best/sample.png",
                "distance": float("inf"),
            }
        ],
    }

    report = write_video_run_dir(
        tmp_path,
        payload=payload,
        profile=profile,
        clarification="",
    )

    saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert saved["videos"][0]["distance"] is None
    text = report.read_text(encoding="utf-8")
    assert "sample.mp4" in text
    assert "1 из 3" in text
