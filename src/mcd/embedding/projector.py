from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mcd.embedding.base import Embedder


class ProjectedEmbedder:
    """Wrap an embedder with a linear (or 2-layer) projection loaded from .npz / .pt."""

    def __init__(self, base: Embedder, path: str | Path) -> None:
        self.base = base
        self.path = Path(path)
        self.weight_1, self.bias_1, self.weight_2, self.bias_2 = _load_projection(self.path)

    def embed(self, texts):
        x = np.asarray(self.base.embed(texts), dtype=np.float64)
        h = x @ self.weight_1.T + self.bias_1
        if self.weight_2 is None:
            return h
        h = np.maximum(h, 0.0)
        return h @ self.weight_2.T + self.bias_2


def _load_projection(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"projector_path not found: {path}")
    if path.suffix == ".npz":
        data = np.load(path)
        w1 = np.asarray(data["weight"], dtype=np.float64)
        b1 = np.asarray(data["bias"], dtype=np.float64)
        w2 = np.asarray(data["weight2"], dtype=np.float64) if "weight2" in data.files else None
        b2 = np.asarray(data["bias2"], dtype=np.float64) if "bias2" in data.files else None
        return w1, b1, w2, b2
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Loading .pt projectors requires torch") from exc
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "weight" in payload:
        w1 = _as_numpy(payload["weight"])
        b1 = _as_numpy(payload.get("bias", np.zeros(w1.shape[0])))
        w2 = _as_numpy(payload["weight2"]) if "weight2" in payload else None
        b2 = _as_numpy(payload["bias2"]) if "bias2" in payload else None
        return w1, b1, w2, b2
    raise ValueError(f"Unsupported projector file: {path}")


def _as_numpy(value) -> NDArray[np.float64]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)
