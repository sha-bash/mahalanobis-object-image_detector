from __future__ import annotations

from PIL import Image

from moid.regions.base import BBox


class GridProposer:
    """Overlapping grid of windows, optionally at several row/col scales."""

    def __init__(
        self,
        rows: int = 3,
        cols: int = 3,
        overlap: float = 0.2,
        extra_scales: tuple[tuple[int, int], ...] = (),
        fine_scales: tuple[tuple[int, int], ...] = (),
        fine_overlap: float = 0.1,
    ) -> None:
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols must be >= 1")
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")
        self.rows = rows
        self.cols = cols
        self.overlap = overlap
        self.extra_scales = extra_scales
        self.fine_scales = fine_scales
        self.fine_overlap = fine_overlap

    def propose(self, image: Image.Image) -> list[BBox]:
        boxes: list[BBox] = []
        seen: set[tuple[int, int, int, int]] = set()
        for rows, cols in ((self.rows, self.cols), *self.extra_scales):
            for box in _grid_boxes(image.width, image.height, rows, cols, self.overlap):
                key = box.as_tuple()
                if key not in seen and box.area() > 0:
                    seen.add(key)
                    boxes.append(box)
        for rows, cols in self.fine_scales:
            for box in _grid_boxes(image.width, image.height, rows, cols, self.fine_overlap):
                key = box.as_tuple()
                if key not in seen and box.area() > 0:
                    seen.add(key)
                    boxes.append(box)
        return boxes


def _grid_boxes(width: int, height: int, rows: int, cols: int, overlap: float) -> list[BBox]:
    cell_w = width / cols
    cell_h = height / rows
    win_w = cell_w * (1.0 + overlap)
    win_h = cell_h * (1.0 + overlap)
    boxes: list[BBox] = []
    for r in range(rows):
        for c in range(cols):
            cx = (c + 0.5) * cell_w
            cy = (r + 0.5) * cell_h
            x1 = int(round(cx - win_w / 2))
            y1 = int(round(cy - win_h / 2))
            x2 = int(round(cx + win_w / 2))
            y2 = int(round(cy + win_h / 2))
            box = BBox(x1, y1, x2, y2).clip(width, height)
            if box.area() > 0:
                boxes.append(box)
    return boxes
