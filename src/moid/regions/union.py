from __future__ import annotations

import logging

from PIL import Image

from moid.regions.base import BBox, ProposedBox

logger = logging.getLogger(__name__)


def as_proposed(items, default_source: str = "unknown") -> list[ProposedBox]:
    proposed: list[ProposedBox] = []
    for item in items:
        if isinstance(item, ProposedBox):
            proposed.append(item)
        elif isinstance(item, BBox):
            proposed.append(ProposedBox(item, 1.0, default_source))
        else:
            raise TypeError(f"Unsupported proposal type: {type(item)!r}")
    return proposed


def dedupe_proposed(boxes: list[ProposedBox]) -> list[ProposedBox]:
    seen: set[tuple[int, int, int, int, str]] = set()
    unique: list[ProposedBox] = []
    for item in boxes:
        key = (*item.box.as_tuple(), item.source)
        if key in seen or item.box.area() <= 0:
            continue
        seen.add(key)
        unique.append(item)
    return unique


class UnionProposer:
    """Concatenate proposals from several backends (YOLO + grid) and keep both."""

    def __init__(self, *proposers, max_yolo_area_ratio: float = 0.12) -> None:
        self.proposers = proposers
        self.max_yolo_area_ratio = max_yolo_area_ratio

    def propose(self, image: Image.Image) -> list[ProposedBox]:
        collected: list[ProposedBox] = []
        image_area = max(image.width * image.height, 1)
        for proposer in self.proposers:
            source = getattr(proposer, "__class__", type(proposer)).__name__.lower()
            default_source = "yolo" if "yolo" in source else "grid" if "grid" in source else "union"
            try:
                raw = proposer.propose(image)
            except Exception:
                logger.exception("Union proposer failed for %s", type(proposer).__name__)
                continue
            for item in as_proposed(raw, default_source):
                if item.source == "yolo" and item.box.area() / image_area > self.max_yolo_area_ratio:
                    collected.append(
                        ProposedBox(item.box, item.confidence, "yolo_large")
                    )
                else:
                    collected.append(item)
        return dedupe_proposed(collected)
