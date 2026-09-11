import subprocess
import sys

import cv2
import numpy as np
import pytest

from moid.instance import InstanceLocator, run


def test_cpu_entrypoint_does_not_load_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import moid.instance, sys; assert "torch" not in sys.modules',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def reference_image():
    rng = np.random.default_rng(42)
    ref = rng.integers(0, 255, (150, 180, 3), dtype=np.uint8)
    ref = cv2.GaussianBlur(ref, (3, 3), 0)
    cv2.putText(
        ref, "TARGET", (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    return ref


def test_localizes_transformed_reference_and_tracks_then_rejects_absence():
    ref = reference_image()
    locator = InstanceLocator(ref)
    transform = cv2.getRotationMatrix2D((90, 75), 12, 0.85)
    transform[:, 2] += [160, 120]
    frame = cv2.warpAffine(ref, transform, (640, 480))
    found = locator.update(frame)
    assert found is not None
    expected = locator.polygon(np.vstack([transform, [0, 0, 1]]))
    assert np.max(np.abs(np.array(found["polygon"]) - expected)) < 3
    transform[:, 2] += [3, 2]
    following = cv2.warpAffine(ref, transform, (640, 480))
    found = locator.update(following, allow_detection=False)
    assert found is not None
    assert found["source"] == "tracked"
    assert locator.update(np.zeros_like(frame)) is None


def test_unrelated_scene_does_not_match():
    locator = InstanceLocator(reference_image())
    other = np.random.default_rng(9).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    assert locator.update(other) is None


def test_partial_visibility_keeps_tracking():
    ref = reference_image()
    locator = InstanceLocator(ref)
    frame = np.zeros((300, 400, 3), np.uint8)
    frame[140:290, 100:280] = ref
    assert locator.update(frame) is not None
    # Smooth motion takes part of the reference past the bottom edge.
    for y in range(143, 181, 3):
        frame = np.zeros((300, 400, 3), np.uint8)
        visible = min(150, 300 - y)
        frame[y : y + visible, 100:280] = ref[:visible]
        assert locator.update(frame, allow_detection=False) is not None


def test_video_export_and_vlm_veto(tmp_path):
    ref = reference_image()
    reference = tmp_path / "ref.png"
    cv2.imwrite(str(reference), ref)
    video = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (640, 480))
    for offset in range(5):
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[100:250, 200 + offset : 380 + offset] = ref
        writer.write(frame)
    writer.release()
    result = run(reference, video, tmp_path / "positive", baseline=True)
    assert result["processed_frames"] == 5
    assert result["found_frames"] == 5
    assert result["vlm_model"] is None
    cap = cv2.VideoCapture(str(tmp_path / "positive" / "annotated.mp4"))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 5
    assert cap.get(cv2.CAP_PROP_FPS) == 5
    cap.release()

    class Veto:
        model = "test-only"

        def __init__(self, ref):
            pass

        def verify(self, candidate):
            return {"match": "no", "reason": "Different vehicle"}

    rejected = run(
        reference, video, tmp_path / "negative", verifier=Veto, baseline=True
    )
    assert rejected["found_frames"] == 0
    assert rejected["vlm_checks"][0]["match"] == "no"

    class Confirm(Veto):
        def verify(self, candidate):
            return {"match": "yes", "reason": "Matching details"}

    periodic = run(
        reference,
        video,
        tmp_path / "periodic",
        verifier=Confirm,
        verify_every=0.2,
        baseline=True,
    )
    assert periodic["found_frames"] == 5
    assert len(periodic["vlm_checks"]) == 5
    with pytest.raises(FileExistsError):
        run(reference, video, tmp_path / "positive", baseline=True)
