from pathlib import Path

from PIL import Image

from moid.adapters.ocr import StubOCR, adjust_distance, markings_match, normalize_marking
from moid.captions import parse_caption
from moid.config import MoidConfig
from moid.identity import build_identity_profile
from moid.pipeline.few_shot import _ocr_if_needed


def test_normalize_and_match():
    assert normalize_marking("е661сх73") == "E661CX73"
    assert markings_match("Е661СХ73", "E661CX73")


def test_adjust_scales():
    assert adjust_distance(2.0, "E661CX73", "E661CX73") == 1.0
    assert adjust_distance(2.0, "X000XX00", "E661CX73") == 3.0
    assert adjust_distance(2.0, None, "E661CX73") == 2.0


def test_stub_ocr_sidecar(tmp_path: Path):
    img = tmp_path / "car.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(img)
    (tmp_path / "car.plate.txt").write_text("E661CX73", encoding="utf-8")
    read = StubOCR().read_text(img)
    assert read is not None
    assert read.text == "E661CX73"


def test_ocr_skipped_for_generic_domain():
    profile = build_identity_profile(
        ["object: tree; target_match: yes; category: plant; vehicle_class: n/a; parts: leaves"]
    )
    cfg = MoidConfig()
    cfg.ocr.backend = "stub"
    fields = parse_caption("object: tree; parts: license plate")
    assert _ocr_if_needed(fields, None, StubOCR(), profile, cfg) is None
