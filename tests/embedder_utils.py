from __future__ import annotations

import numpy as np


class LexicalEmbedder:
    """Deterministic bag-of-keyword embedder for tests (no SBERT download)."""

    dim = 8

    def embed(self, texts):
        rows = []
        for text in texts:
            low = text.lower()
            v = np.zeros(self.dim, dtype=np.float64)
            for i, token in enumerate(
                ("truck", "red", "trailer", "terrain", "green", "tree", "white", "cabin")
            ):
                if token in low:
                    v[i] = 1.0
            rows.append(v)
        return np.vstack(rows)
