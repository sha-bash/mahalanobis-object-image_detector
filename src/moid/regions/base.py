from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def clip(self, width: int, height: int) -> BBox:
        x1 = max(0, min(self.x1, width))
        y1 = max(0, min(self.y1, height))
        x2 = max(0, min(self.x2, width))
        y2 = max(0, min(self.y2, height))
        return BBox(x1, y1, x2, y2)

    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class ProposedBox:
    box: BBox
    confidence: float = 1.0
    source: str = "unknown"


class RegionProposer(Protocol):
    def propose(self, image: Image.Image) -> list[BBox | ProposedBox]:
        """Return candidate boxes in pixel coordinates (inclusive-exclusive)."""


def crop_box(image: Image.Image, box: BBox) -> Image.Image:
    clipped = box.clip(image.width, image.height)
    if clipped.area() == 0:
        raise ValueError(f"Empty crop for box {box}")
    return image.crop(clipped.as_tuple())
