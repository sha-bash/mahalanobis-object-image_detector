from pathlib import Path

import numpy as np

import moid.video as video_mod
from moid.video import extract_frames, list_videos


class FakeCapture:
    def __init__(self, frame_count: int = 5, fps: float = 4.0):
        self.frames = [np.zeros((2, 3, 3), dtype=np.uint8) for _ in range(frame_count)]
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, _key):
        return self.fps

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


def test_extract_frames_uses_requested_fps(monkeypatch):
    capture = FakeCapture()
    monkeypatch.setattr(video_mod.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(video_mod.cv2, "cvtColor", lambda frame, _mode: frame)

    frames = list(extract_frames("sample.mp4", sample_fps=2.0))

    assert [index for index, _, _ in frames] == [0, 2, 4]
    assert [timestamp for _, timestamp, _ in frames] == [0.0, 0.5, 1.0]
    assert capture.released is True


def test_extract_frames_caps_sampling_at_source_fps(monkeypatch):
    monkeypatch.setattr(video_mod.cv2, "VideoCapture", lambda _path: FakeCapture(3, 2.0))
    monkeypatch.setattr(video_mod.cv2, "cvtColor", lambda frame, _mode: frame)

    assert len(list(extract_frames("sample.mp4", sample_fps=20.0))) == 3


def test_list_videos_filters_and_sorts(tmp_path: Path):
    (tmp_path / "b.MP4").touch()
    (tmp_path / "A.mov").touch()
    (tmp_path / "notes.txt").touch()

    videos = list_videos(tmp_path, [".mp4", ".mov"])

    assert [path.name for path in videos] == ["A.mov", "b.MP4"]
