import json
from pathlib import Path

from PIL import Image
import pytest

import moid.cli as cli_mod
from moid.calibration import optimize_threshold_by_f1
from moid.cli import main
from moid.config import MoidConfig
from moid.metrics import (
    GroundTruth,
    Prediction,
    compute_metrics,
    load_ground_truth,
)
from moid.pipeline.few_shot import caption_references
from moid.regions.base import BBox, ProposedBox
from moid.regions.yolo import YoloProposalGenerator
from moid.scoring import ScoredBox, soft_nms
from moid.tracking import Tracker


def test_metrics_perfect_detection():
    gt = [GroundTruth("frame_00000000", (10, 10, 20, 20))]
    predictions = [Prediction("frame_00000000", (10, 10, 20, 20), 0.9)]

    result = compute_metrics(gt, predictions)

    assert result.precision == result.recall == result.f1 == 1.0
    assert result.map_50 == result.map_50_95 == 1.0


def test_metrics_duplicate_is_false_positive():
    gt = [GroundTruth("frame_00000000", (0, 0, 10, 10))]
    predictions = [
        Prediction("frame_00000000", (0, 0, 10, 10), 0.9),
        Prediction("frame_00000000", (0, 0, 10, 10), 0.8),
    ]

    result = compute_metrics(gt, predictions)

    assert (result.tp, result.fp, result.fn) == (1, 1, 0)
    assert result.precision == 0.5


def test_load_coco_uses_file_stem(tmp_path: Path):
    annotation = tmp_path / "instances.json"
    annotation.write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": "frames/frame_00000042.jpg"}],
                "annotations": [
                    {"id": 1, "image_id": 7, "category_id": 3, "bbox": [1, 2, 4, 5]}
                ],
                "categories": [{"id": 3, "name": "van"}],
            }
        ),
        encoding="utf-8",
    )

    gt, categories = load_ground_truth(annotation)

    assert gt[0].frame == "frame_00000042"
    assert gt[0].bbox == (1.0, 2.0, 5.0, 7.0)
    assert categories == {3: "van"}


def test_load_yolo_denormalizes_boxes(tmp_path: Path):
    (tmp_path / "frame_00000000.txt").write_text("0 0.5 0.5 0.2 0.4\n")

    gt, _ = load_ground_truth(
        tmp_path, image_sizes={"frame_00000000": (100, 50)}
    )

    assert gt[0].bbox == pytest.approx((40, 15, 60, 35))


def test_soft_nms_reduces_duplicate_confidence():
    boxes = [
        ScoredBox(BBox(0, 0, 10, 10), 1.0, 2.0, True, ""),
        ScoredBox(BBox(1, 1, 11, 11), 1.1, 2.0, True, ""),
    ]

    kept = soft_nms(boxes, iou_threshold=0.3)

    assert len(kept) == 2
    assert kept[1].confidence < kept[0].confidence


def test_tracker_preserves_identity():
    tracker = Tracker(iou_threshold=0.3)
    first = ScoredBox(BBox(0, 0, 10, 10), 1.0, 2.0, True, "")
    second = ScoredBox(BBox(1, 0, 11, 10), 1.0, 2.0, True, "")

    tracker.update([first])
    tracker.update([second])

    assert first.track_id == second.track_id == 1
    assert tracker.stability((100, 100))["matched_transitions"] == 1


def test_yolo_infer_function_does_not_require_ultralytics():
    expected = [ProposedBox(BBox(1, 2, 3, 4), 0.8, "yolo")]
    proposer = YoloProposalGenerator(infer_fn=lambda _image: expected)

    assert proposer.propose(Image.new("RGB", (10, 10))) == expected


def test_threshold_calibration_maximizes_f1():
    result = optimize_threshold_by_f1([0.1, 0.2, 0.8, 0.9], [True, True, False, False])

    assert result.threshold == 0.2
    assert result.f1 == 1.0


class _CaptionVLM:
    def describe(self, image):
        return "object: sedan" if "sedan" in str(image) else "object: van"


def test_reference_filter_excludes_sedan_filename(tmp_path: Path):
    Image.new("RGB", (4, 4)).save(tmp_path / "van.png")
    Image.new("RGB", (4, 4)).save(tmp_path / "sedan.png")
    cfg = MoidConfig()
    cfg.references.allowed_terms = ["van", "gazelle"]

    captions, _ = caption_references(tmp_path, _CaptionVLM(), cfg)

    assert captions == ["object: van"]


def test_cli_quality_flags_override_config(monkeypatch):
    captured = {}

    def fake_session(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)

    monkeypatch.setattr(cli_mod, "run_video_search_session", fake_session)

    assert (
        main(
            [
                "search-video",
                "--refs",
                "refs",
                "--video",
                "video.mp4",
                "--detector",
                "yolo",
                "--nms-method",
                "soft",
                "--track",
                "--tau",
                "1.5",
                "--gt-annotations",
                "gt.json",
                "--save-metrics",
                "--non-interactive",
            ]
        )
        == 0
    )
    cfg = captured["config"]
    assert cfg.regions.backend == "yolo"
    assert cfg.nms.method == "soft"
    assert cfg.tracking.enabled is True
    assert cfg.detector.manual_threshold == 1.5
    assert cfg.evaluation.gt_annotations == "gt.json"
    assert cfg.evaluation.save_metrics is True
