from pathlib import Path

import numpy as np

from mcd.embedding.projector import ProjectedEmbedder
from tests.embedder_utils import LexicalEmbedder


def test_linear_projector_roundtrip(tmp_path: Path):
    weight = np.eye(4, 8, dtype=np.float64)
    bias = np.zeros(4, dtype=np.float64)
    path = tmp_path / "proj.npz"
    np.savez(path, weight=weight, bias=bias)
    wrapped = ProjectedEmbedder(LexicalEmbedder(), path)
    out = wrapped.embed(["red truck"])
    assert out.shape == (1, 4)
