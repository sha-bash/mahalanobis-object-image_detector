from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from PIL import Image
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class VisualEncoder:
    """Encodes images into fixed-size vectors using a pretrained vision model.
    
    Supports CLIP (via transformers) and DINOv2 backends.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._backend: Literal["clip", "dinov2"] = "clip"
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            logger.info("Loading CLIP model: %s on %s", self.model_name, self.device)
            self._model = CLIPModel.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._backend = "clip"
        except Exception:
            logger.exception("Failed to load CLIP model, trying DINOv2")
            try:
                from transformers import AutoModel, AutoImageProcessor
                logger.info("Loading DINOv2 model: %s on %s", self.model_name, self.device)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model = self._model.to(self.device)
                self._processor = AutoImageProcessor.from_pretrained(self.model_name)
                self._backend = "dinov2"
            except Exception:
                raise RuntimeError(
                    f"Failed to load visual encoder '{self.model_name}'. "
                    "Supported: CLIP (e.g. 'openai/clip-vit-base-patch32') or DINOv2 "
                    "(e.g. 'facebook/dinov2-small'). Install transformers and torch."
                )

    def encode(self, image: Image.Image) -> NDArray[np.float64]:
        """Encode a single PIL Image into a normalized embedding vector."""
        pil = image.convert("RGB")
        inputs = self._processor(images=pil, return_tensors="pt").to(self._model.device)
        with self._model.no_grad():
            if self._backend == "clip":
                features = self._model.get_image_features(**inputs)
            else:
                outputs = self._model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)  # [CLS] pooling
        return features.cpu().numpy().flatten().astype(np.float64)
