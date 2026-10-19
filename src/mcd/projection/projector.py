"""Trainable MLP projector from high-dimensional embeddings to a compact space."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn


class DeepMahalanobisProjector(nn.Module):
    """Multi-layer feed-forward projector with BatchNorm, GELU, and dropout.

    Maps input embeddings (e.g. SBERT) to a lower-dimensional representation
    suitable for Mahalanobis distance in reduced space.

    Args:
        input_dim: Dimension of input embedding vectors.
        hidden_dims: Sizes of hidden linear layers before the final projection.
        output_dim: Dimension of the output (projected) vectors.
        dropout: Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))

    def architecture_dict(self) -> dict[str, Any]:
        """Return JSON-serializable hyperparameters for save/load."""
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        if not linears:
            raise ValueError("projector has no Linear layers")
        dropouts = [m for m in self.net if isinstance(m, nn.Dropout)]
        dropout = float(dropouts[0].p) if dropouts else 0.0
        return {
            "input_dim": linears[0].in_features,
            "hidden_dims": [lin.out_features for lin in linears[:-1]],
            "output_dim": linears[-1].out_features,
            "dropout": dropout,
        }

    @classmethod
    def from_architecture_dict(cls, cfg: dict[str, Any]) -> DeepMahalanobisProjector:
        """Build a projector from a dict produced by :meth:`architecture_dict` or training JSON."""
        hidden = cfg.get("hidden_dims")
        if hidden is None:
            hidden = [256, 128]
        return cls(
            input_dim=int(cfg["input_dim"]),
            hidden_dims=list(hidden),
            output_dim=int(cfg["output_dim"]),
            dropout=float(cfg.get("dropout", 0.1)),
        )
