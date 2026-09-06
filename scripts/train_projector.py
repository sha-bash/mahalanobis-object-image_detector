#!/usr/bin/env python
"""Fit a linear / 2-layer projection from a caption corpus (PCA init, NumPy only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mcd.embedding.sbert import SBERT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a caption embedding projector")
    parser.add_argument("--captions", required=True, help="Text file, one caption per line")
    parser.add_argument("--out", required=True, help="Output .npz path")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args(argv)

    lines = [ln.strip() for ln in Path(args.captions).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        raise SystemExit("Need at least 2 captions")
    embedder = SBERT(model_name=args.model)
    x = np.asarray(embedder.embed(lines), dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    dim = min(args.dim, vt.shape[0])
    weight = vt[:dim]
    bias = np.zeros(dim, dtype=np.float64)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, weight=weight, bias=bias)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
