from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def _to_numpy_matrix(features) -> np.ndarray:
    if hasattr(features, "image_embeds"):
        features = features.image_embeds
    elif hasattr(features, "pooler_output") and features.pooler_output is not None:
        features = features.pooler_output
    elif hasattr(features, "last_hidden_state"):
        features = features.last_hidden_state[:, 0]
    tensor = torch.as_tensor(features)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)
    return tensor.detach().cpu().numpy().astype(np.float64, copy=False)


def _to_numpy_vector(features) -> np.ndarray:
    """Accept a tensor or Hugging Face vision output and return a 1-D numpy vector."""
    return _to_numpy_matrix(features)[0]


class VisualEncoder:
    """Encodes images into fixed-size vectors using a pretrained vision model."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device

    def encode(self, image: Image.Image) -> np.ndarray:
        return self.encode_batch([image], batch_size=1)[0]

    def encode_batch(self, images: list[Image.Image], batch_size: int = 4) -> np.ndarray:
        if not images:
            return np.zeros((0, 0), dtype=np.float64)
        rows: list[np.ndarray] = []
        step = max(1, int(batch_size))
        for start in range(0, len(images), step):
            chunk = images[start : start + step]
            inputs = self.processor(images=chunk, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.no_grad():
                if hasattr(self.model, "get_image_features"):
                    features = self.model.get_image_features(pixel_values=pixel_values)
                else:
                    features = self.model.vision_model(pixel_values=pixel_values)
            rows.append(_to_numpy_matrix(features))
        return np.concatenate(rows, axis=0)
