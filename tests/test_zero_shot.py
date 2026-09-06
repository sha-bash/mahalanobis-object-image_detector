from pathlib import Path

from PIL import Image

from moid.adapters.llm import StubLLM
from moid.config import MoidConfig
from moid.pipeline.zero_shot import run_zero_shot
from tests.embedder_utils import LexicalEmbedder

TRUCK = "object: red truck; color: red; shape: rectangular; material: metal; parts: cabin; markings: none; size_relative: medium"
TREE = "object: tree; color: green; shape: irregular; material: vegetation; parts: none; markings: none; size_relative: large"


class SidecarOrNameVLM:
    def describe(self, image) -> str:
        name = Path(image).stem.lower()
        if "truck" in name:
            return TRUCK
        return TREE


def _write_blank(path: Path) -> None:
    Image.new("RGB", (8, 8), (128, 128, 128)).save(path)


def test_zero_shot_pairwise_argmin(tmp_path: Path):
    truck = tmp_path / "truck.png"
    tree = tmp_path / "tree.png"
    _write_blank(truck)
    _write_blank(tree)
    cfg = MoidConfig()
    cfg.zero_shot.mode = "pairwise"
    cfg.zero_shot.reject_threshold = None
    result = run_zero_shot(
        query="red truck",
        images=tmp_path,
        vlm=SidecarOrNameVLM(),
        llm=StubLLM(),
        embedder=LexicalEmbedder(),
        config=cfg,
    )
    assert result.rejected is False
    assert result.best is not None
    assert "truck" in result.best.path
    dists = {Path(s.path).stem: s.distance for s in result.scores if not s.failed}
    assert dists["truck"] < dists["tree"]


def test_zero_shot_one_shot_and_reject(tmp_path: Path):
    _write_blank(tmp_path / "tree.png")
    cfg = MoidConfig()
    cfg.zero_shot.mode = "one_shot"
    cfg.zero_shot.reject_threshold = 0.01
    result = run_zero_shot(
        query="red truck",
        images=tmp_path,
        vlm=SidecarOrNameVLM(),
        llm=StubLLM(),
        embedder=LexicalEmbedder(),
        config=cfg,
    )
    assert result.rejected is True
    assert result.best is None
