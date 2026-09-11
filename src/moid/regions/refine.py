from __future__ import annotations

from moid.regions.base import BBox, ProposedBox
from moid.regions.union import as_proposed, dedupe_proposed


def expand_around_seeds(
    seeds,
    *,
    width: int,
    height: int,
    scales: tuple[float, ...] = (0.7, 1.0, 1.3),
    shifts: tuple[float, ...] = (-0.15, 0.0, 0.15),
    source: str = "refine",
) -> list[ProposedBox]:
    """Coarse-to-fine windows: shift and scale around seed boxes."""
    expanded: list[ProposedBox] = []
    for seed in as_proposed(seeds, source):
        box = seed.box
        bw = max(1, box.x2 - box.x1)
        bh = max(1, box.y2 - box.y1)
        cx = (box.x1 + box.x2) / 2.0
        cy = (box.y1 + box.y2) / 2.0
        for scale in scales:
            win_w = bw * scale
            win_h = bh * scale
            for dx in shifts:
                for dy in shifts:
                    ncx = cx + dx * bw
                    ncy = cy + dy * bh
                    x1 = int(round(ncx - win_w / 2.0))
                    y1 = int(round(ncy - win_h / 2.0))
                    x2 = int(round(ncx + win_w / 2.0))
                    y2 = int(round(ncy + win_h / 2.0))
                    refined = BBox(x1, y1, x2, y2).clip(width, height)
                    if refined.area() > 0:
                        expanded.append(ProposedBox(refined, seed.confidence, source))
    return dedupe_proposed(expanded)
