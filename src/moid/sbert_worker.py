"""Short-lived CPU SBERT worker; exits before the next VLM request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument(
        "--prepare", action="store_true", help="Download and verify SBERT weights once"
    )
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    if not args.prepare and (not args.input or not args.output):
        parser.error("--input and --output are required unless --prepare is used")
    import numpy as np
    import torch
    from mcd.embedding.sbert import SBERT

    torch.set_num_threads(4)
    texts = (
        ["Local embedding readiness check."]
        if args.prepare
        else json.loads(Path(args.input).read_text(encoding="utf-8"))
    )
    if (
        not isinstance(texts, list)
        or not texts
        or not all(isinstance(t, str) and t.strip() for t in texts)
    ):
        raise ValueError("Expected a non-empty list of captions")
    # Normal processing never needs an internet connection after preparation.
    embedder = SBERT(args.model, local_files_only=not args.prepare)
    embedder.model.to("cpu")
    vectors = np.asarray(embedder.embed(texts), dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms <= 0):
        raise ValueError("SBERT produced invalid vectors")
    if args.prepare:
        print(f"SBERT ready: {vectors.shape[1]} dimensions")
    else:
        np.save(args.output, vectors / norms, allow_pickle=False)


if __name__ == "__main__":
    main()
