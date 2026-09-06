from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Union

from PIL import Image

from moid.adapters.base import ImageLike
from moid.adapters.vlm import load_image

CYR_TO_LAT = str.maketrans(
    {
        "А": "A",
        "В": "B",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "С": "C",
        "Т": "T",
        "У": "Y",
        "Х": "X",
    }
)


@dataclass
class TextRead:
    text: str
    confidence: float = 1.0


class OCRClient(Protocol):
    def read_text(self, image: ImageLike) -> TextRead | None:
        """Read prominent text (e.g. a license plate). None if nothing usable."""


def normalize_marking(text: str) -> str:
    raw = (text or "").strip().upper().translate(CYR_TO_LAT)
    return "".join(ch for ch in raw if ch.isalnum())


def markings_match(a: str, b: str) -> bool:
    na, nb = normalize_marking(a), normalize_marking(b)
    return bool(na) and na == nb


class NullOCR:
    def read_text(self, image: ImageLike) -> TextRead | None:
        return None


class StubOCR:
    """Read a sidecar ``*.plate.txt`` next to a file path."""

    def read_text(self, image: ImageLike) -> TextRead | None:
        if isinstance(image, Image.Image):
            return None
        path = Path(image)
        sidecar = path.with_suffix(".plate.txt")
        if not sidecar.is_file():
            return None
        text = sidecar.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return TextRead(text=text, confidence=1.0)


class EasyOCRClient:
    def __init__(self, langs: tuple[str, ...] = ("en", "ru"), min_confidence: float = 0.3) -> None:
        self.langs = list(langs)
        self.min_confidence = min_confidence
        self._reader = None

    def _reader_or_raise(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError("EasyOCR is not installed. pip install moid[ocr]") from exc
            self._reader = easyocr.Reader(self.langs, verbose=False)
        return self._reader

    def read_text(self, image: ImageLike) -> TextRead | None:
        pil = load_image(image)
        import numpy as np

        results = self._reader_or_raise().readtext(np.asarray(pil))
        best: TextRead | None = None
        best_score = -1.0
        for _box, text, conf in results:
            score = float(conf)
            if score < self.min_confidence:
                continue
            if score > best_score:
                best_score = score
                best = TextRead(text=str(text), confidence=score)
        return best


def adjust_distance(
    distance: float,
    observed: str | None,
    expected: str | None,
    *,
    match_scale: float = 0.5,
    mismatch_scale: float = 1.5,
) -> float:
    if not observed or not expected:
        return distance
    if normalize_marking(expected) in {"", "UNKNOWN", "NONE", "NA"}:
        return distance
    if markings_match(observed, expected):
        return float(distance) * float(match_scale)
    return float(distance) * float(mismatch_scale)
