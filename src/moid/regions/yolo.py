from __future__ import annotations

from collections.abc import Callable

from PIL import Image

from moid.regions.base import BBox, ProposedBox


class YoloProposalGenerator:
    """Ultralytics YOLO proposal generator with lazy model loading."""

    _VEHICLE_CLASS_IDS = {2, 3, 5, 7}

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        *,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        max_boxes: int = 20,
        vehicle_only: bool = True,
        infer_fn: Callable[[Image.Image], list[ProposedBox]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.max_boxes = max_boxes
        self.vehicle_only = vehicle_only
        self.infer_fn = infer_fn
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "YOLO proposals require the optional dependency: "
                    'pip install -e ".[yolo]"'
                ) from exc
            self._model = YOLO(self.model_name)
        return self._model

    def propose(self, image: Image.Image) -> list[ProposedBox]:
        if self.infer_fn is not None:
            return self.infer_fn(image)
        classes = sorted(self._VEHICLE_CLASS_IDS) if self.vehicle_only else None
        result = self._load_model().predict(
            source=image,
            conf=self.confidence,
            iou=self.iou_threshold,
            max_det=self.max_boxes,
            classes=classes,
            verbose=False,
        )[0]
        proposals: list[ProposedBox] = []
        if result.boxes is None:
            return proposals
        coordinates = result.boxes.xyxy.detach().cpu().tolist()
        scores = result.boxes.conf.detach().cpu().tolist()
        for raw_box, score in zip(coordinates, scores):
            x1, y1, x2, y2 = (int(round(value)) for value in raw_box)
            box = BBox(x1, y1, x2, y2).clip(image.width, image.height)
            if box.area() > 0:
                proposals.append(ProposedBox(box, float(score), "yolo"))
        return proposals
