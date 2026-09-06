from __future__ import annotations

from PIL import Image, ImageDraw

from moid.scoring import ScoredBox


def draw_overlays(image: Image.Image, boxes: list[ScoredBox], path) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for scored in boxes:
        x1, y1, x2, y2 = scored.box.as_tuple()
        color = (0, 200, 0) if scored.accepted else (220, 160, 0) if scored.visualization_only else (200, 40, 40)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{scored.distance:.2f}"
        draw.text((x1 + 4, y1 + 4), label, fill=color)
    canvas.save(path)
