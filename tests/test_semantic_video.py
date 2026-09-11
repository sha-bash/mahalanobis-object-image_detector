import json

import cv2
import numpy as np
import pytest

from moid.instance import run
from moid.semantic import (
    DescriptionError,
    IndependentCaptioner,
    SemanticVerifier,
    canonical_caption,
)
from tests.test_instance import reference_image


class Captions:
    def __init__(self, texts):
        self.texts = iter(texts)

    def describe(self, image):
        return {"caption": next(self.texts), "cache_hit": False}


class Embeddings:
    def embed(self, texts):
        return np.array([[1.0, 0.0] if t == "van" else [0.0, 1.0] for t in texts])


def verifier(texts, **kwargs):
    return SemanticVerifier(
        reference_image(), captioner=Captions(texts), embedder=Embeddings(), **kwargs
    )


def test_real_mahalanobis_math_and_rejection():
    gate = verifier(["van", "van", "sedan"], threshold=6, variance=0.01)
    same = gate.verify(reference_image())
    other = gate.verify(reference_image())
    assert same["distance"] == pytest.approx(0)
    assert same["accepted"] is True
    assert other["distance"] == pytest.approx(np.sqrt(200))
    assert other["accepted"] is False


def test_external_covariance_is_used(tmp_path):
    path = tmp_path / "variance.npy"
    np.save(path, [0.25, 1.0])
    gate = verifier(["van", "sedan"], covariance_path=path, threshold=2)
    result = gate.verify(reference_image())
    assert result["distance"] == pytest.approx(np.sqrt(5))
    assert result["accepted"] is False


def clip(tmp_path):
    ref = reference_image()
    image = tmp_path / "ref.png"
    cv2.imwrite(str(image), ref)
    video = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (640, 480))
    for i in range(5):
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[100:250, 200 + i : 380 + i] = ref
        writer.write(frame)
    writer.release()
    return image, video


def test_geometry_cannot_override_semantic_rejection(tmp_path):
    image, video = clip(tmp_path)
    factory = lambda ref: verifier(["van", "sedan"])
    result = run(image, video, tmp_path / "rejected", verifier=factory)
    assert result["mandatory_mahalanobis"] is True
    assert result["found_frames"] == 0
    assert (
        result["semantic_checks"][0]["distance"]
        > result["semantic_checks"][0]["threshold"]
    )


def test_match_flag_cannot_override_actual_distance(tmp_path):
    image, video = clip(tmp_path)
    gate = verifier(["van"])
    gate.verify = lambda candidate: {
        "match": "yes",
        "accepted": True,
        "distance": 9.0,
        "threshold": 6.0,
    }
    result = run(image, video, tmp_path / "inconsistent", verifier=lambda ref: gate)
    assert result["found_frames"] == 0
    assert result["semantic_checks"][0]["accepted"] is False


def test_periodic_semantic_failure_removes_track(tmp_path):
    image, video = clip(tmp_path)
    result = run(
        image,
        video,
        tmp_path / "lost",
        verify_every=0.4,
        verifier=lambda ref: verifier(["van", "van", "sedan"]),
    )
    assert [row["found"] for row in result["frames"]] == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert result["frames"][0]["semantic_state"] == "confirmed"
    assert result["frames"][1]["semantic_state"] == "propagated"
    assert result["frames"][1]["semantic_source_frame"] == 0
    assert len(result["semantic_checks"]) == 2
    assert (tmp_path / "lost" / "semantic_distances.csv").is_file()


def test_missing_semantic_gate_cannot_silently_run_baseline(tmp_path):
    with pytest.raises(ValueError, match="required"):
        run("ref", "video", tmp_path / "out")


def test_unusable_candidate_is_rejected_without_fabricating_distance(tmp_path):
    gate = verifier(["van"])

    class Unknown:
        def describe(self, image):
            raise DescriptionError("Insufficient visible attributes")

    gate.captioner = Unknown()
    image, video = clip(tmp_path)
    result = run(image, video, tmp_path / "unknown", verifier=lambda ref: gate)
    assert result["found_frames"] == 0
    assert result["semantic_checks"][0]["distance"] is None
    assert result["semantic_checks"][0]["accepted"] is False
    assert result["processed_frames"] == 5


def test_invalid_observations_are_not_embedded():
    with pytest.raises(ValueError):
        canonical_caption(
            {
                k: "unknown"
                for k in ("object", "shape", "color", "parts", "distinctive_features")
            }
        )


def test_descriptions_are_independent_and_cached(tmp_path):
    requests = []
    fields = dict(
        object="van",
        shape="long body",
        color="white",
        parts="windows",
        distinctive_features="unknown",
    )

    def chat(**kwargs):
        requests.append(kwargs)
        return json.dumps(fields)

    captioner = IndependentCaptioner(
        "test-model", "http://localhost", tmp_path, chat_fn=chat
    )
    first = captioner.describe(reference_image())
    again = captioner.describe(reference_image())
    assert first["caption"] == again["caption"]
    assert again["cache_hit"] is True
    assert len(requests) == 1
    assert len(requests[0]["messages"][0]["images"]) == 1
    assert "target_match" not in first["caption"]
    assert "unknown" not in first["caption"]
