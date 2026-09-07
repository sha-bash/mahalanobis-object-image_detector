from pathlib import Path

import numpy as np
from PIL import Image

from moid.adapters.vlm import load_image
from moid.config import MoidConfig
from moid.identity import build_identity_profile
from moid.pipeline.few_shot import detect_on_image, fit_detector, run_few_shot
from moid.regions.dino import DinoProposer, HybridProposer
from moid.regions.grid import GridProposer
from moid.scoring import ScoredBox, iou, nms
from moid.regions.base import BBox
from tests.embedder_utils import LexicalEmbedder

TRUCK = (
    "object: red truck; color: red; shape: rectangular; material: metal; "
    "parts: cabin; markings: none; size_relative: medium"
)
TERRAIN = (
    "object: terrain; color: green; shape: irregular; material: vegetation; "
    "parts: none; markings: none; size_relative: large"
)


class ColorVLM:
    def describe(self, image) -> str:
        arr = np.asarray(load_image(image).convert("RGB"))
        mean = arr.reshape(-1, 3).mean(axis=0)
        return TRUCK if mean[0] >= mean[1] else TERRAIN


def test_grid_proposer_count():
    img = Image.new("RGB", (100, 50), (0, 0, 0))
    boxes = GridProposer(rows=1, cols=2, overlap=0.0).propose(img)
    assert len(boxes) == 2


def test_dino_infer_fn_and_hybrid_fallback():
    img = Image.new("RGB", (40, 40), (0, 0, 0))
    dino = DinoProposer(infer_fn=lambda _im: [BBox(1, 1, 20, 20)], min_area_ratio=0.0)
    boxes = dino.propose(img)
    assert boxes and boxes[0].x1 == 1

    empty = DinoProposer(infer_fn=lambda _im: [])
    hybrid = HybridProposer(empty, fallback=GridProposer(rows=1, cols=1, overlap=0.0))
    fb = hybrid.propose(img)
    assert len(fb) == 1


def test_nms_drops_overlap():
    a = ScoredBox(BBox(0, 0, 10, 10), distance=1.0, threshold=5.0, accepted=True, caption="")
    b = ScoredBox(BBox(1, 1, 11, 11), distance=2.0, threshold=5.0, accepted=True, caption="")
    kept = nms([a, b], iou_threshold=0.3)
    assert len(kept) == 1
    assert kept[0].distance == 1.0
    assert iou(a.box, b.box) > 0.3


def test_few_shot_prefers_red_region(tmp_path: Path):
    refs = tmp_path / "refs"
    refs.mkdir()
    Image.new("RGB", (40, 40), (255, 0, 0)).save(refs / "ref.png")
    target = tmp_path / "target.png"
    img = Image.new("RGB", (80, 40), (0, 200, 0))
    for x in range(40):
        for y in range(40):
            img.putpixel((x, y), (255, 0, 0))
    img.save(target)

    cfg = MoidConfig()
    cfg.grid.rows = 1
    cfg.grid.cols = 2
    cfg.grid.overlap = 0.0
    cfg.grid.include_best_if_none_accepted = True
    cfg.detector.threshold_margin = 5.0
    cfg.decision.target_match_gate = False

    result = run_few_shot(
        refs=refs,
        target=target,
        vlm=ColorVLM(),
        embedder=LexicalEmbedder(),
        config=cfg,
        proposer=GridProposer(rows=1, cols=2, overlap=0.0),
    )
    assert result.detections
    best = result.detections[0]
    assert best.box.x1 < 40
    object_regions = [s for s in result.all_regions if not s.failed]
    left = min(object_regions, key=lambda s: s.box.x1)
    right = max(object_regions, key=lambda s: s.box.x1)
    assert left.distance < right.distance


def test_min_positive_regions_and_visualization_only(tmp_path: Path):
    refs = tmp_path / "refs"
    refs.mkdir()
    Image.new("RGB", (20, 20), (255, 0, 0)).save(refs / "ref.png")
    target = tmp_path / "target.png"
    Image.new("RGB", (40, 20), (0, 200, 0)).save(target)
    cfg = MoidConfig()
    cfg.grid.rows = 1
    cfg.grid.cols = 1
    cfg.grid.overlap = 0.0
    cfg.grid.include_best_if_none_accepted = True
    cfg.decision.min_positive_regions = 2
    cfg.decision.target_match_gate = False
    cfg.detector.threshold_margin = 0.0
    result = run_few_shot(
        refs=refs,
        target=target,
        vlm=ColorVLM(),
        embedder=LexicalEmbedder(),
        config=cfg,
        proposer=GridProposer(rows=1, cols=1, overlap=0.0),
    )
    assert result.frame_positive is False
    assert result.detections
    assert result.detections[0].visualization_only is True


def test_detect_on_image_wires_visual_prefilter():
    class CountingVLM:
        def __init__(self):
            self.calls = 0

        def describe(self, _image):
            self.calls += 1
            return TRUCK

    class ColorEncoder:
        def encode(self, image):
            mean = np.asarray(image).reshape(-1, 3).mean(axis=0)
            return np.array([mean[0], mean[1]], dtype=float)

    image = Image.new("RGB", (40, 20), (0, 255, 0))
    for x in range(20):
        for y in range(20):
            image.putpixel((x, y), (255, 0, 0))
    cfg = MoidConfig()
    cfg.visual.top_k_before_vlm = 1
    cfg.decision.target_match_gate = False
    vlm = CountingVLM()
    profile = build_identity_profile([TRUCK])
    detector = fit_detector([TRUCK], LexicalEmbedder(), cfg)

    result = detect_on_image(
        image,
        detector,
        vlm,
        config=cfg,
        proposer=GridProposer(rows=1, cols=2, overlap=0.0),
        profile=profile,
        reference_captions=[TRUCK],
        visual_encoder=ColorEncoder(),
        reference_visual_embeddings=np.array([[1.0, 0.0]]),
    )

    assert vlm.calls == 1
    assert len([region for region in result.all_regions if not region.failed]) == 1
