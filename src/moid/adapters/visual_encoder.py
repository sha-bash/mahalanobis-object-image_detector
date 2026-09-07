from __future__ import annotations

import numpy as np
from PIL import Image
import torch

class VisualEncoder:
    """Encodes images into fixed-size vectors using a pretrained vision model."""

    def __init__(self, model_name: str, device: str = "cpu"):
        # Example with CLIP from transformers; can be replaced by DINOv2 etc.
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device

    def encode(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        return features.cpu().numpy().flatten()
