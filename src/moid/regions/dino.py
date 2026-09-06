from __future__ import annotations

from PIL import Image

from moid.regions.base import BBox
from moid.regions.grid import GridProposer


class DinoProposer:
    """Zero-shot boxes via Hugging Face Grounding DINO (optional extra ``moid[dino]``)."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        text_prompt: str = "object",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        max_boxes: int = 20,
        min_area_ratio: float = 0.002,
        infer_fn=None,
    ) -> None:
        self.model_id = model_id
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_boxes = max_boxes
        self.min_area_ratio = min_area_ratio
        self._infer_fn = infer_fn
        self._processor = None
        self._model = None

    def propose(self, image: Image.Image) -> list[BBox]:
        if self._infer_fn is not None:
            raw = self._infer_fn(image)
            return self._filter(image, raw)
        boxes = self._hf_infer(image)
        return self._filter(image, boxes)

    def _filter(self, image: Image.Image, boxes: list[BBox]) -> list[BBox]:
        area = max(image.width * image.height, 1)
        kept: list[BBox] = []
        seen: set[tuple[int, int, int, int]] = set()
        for box in boxes:
            clipped = box.clip(image.width, image.height)
            if clipped.area() < self.min_area_ratio * area:
                continue
            key = clipped.as_tuple()
            if key in seen:
                continue
            seen.add(key)
            kept.append(clipped)
            if len(kept) >= self.max_boxes:
                break
        return kept

    def _hf_infer(self, image: Image.Image) -> list[BBox]:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "DinoProposer requires transformers and torch. Install with: pip install moid[dino]"
            ) from exc
        if self._processor is None or self._model is None:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
            self._model.eval()
        prompt = self.text_prompt.strip() or "object"
        if not prompt.endswith("."):
            prompt = prompt + "."
        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )[0]
        boxes: list[BBox] = []
        for xyxy in results.get("boxes", []):
            x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
            boxes.append(BBox(x1, y1, x2, y2))
        return boxes


class HybridProposer:
    """Use a primary proposer; fall back to the grid if it returns nothing."""

    def __init__(self, primary, fallback: GridProposer | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or GridProposer()

    def propose(self, image: Image.Image) -> list[BBox]:
        try:
            boxes = self.primary.propose(image)
        except Exception:
            boxes = []
        if boxes:
            return boxes
        return self.fallback.propose(image)
