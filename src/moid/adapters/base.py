from __future__ import annotations

from pathlib import Path
from typing import Protocol, Union

from PIL import Image

ImageLike = Union[str, Path, Image.Image]


class VLMClient(Protocol):
    def describe(self, image: ImageLike) -> str:
        """Return a standardized attribute caption, or empty string on failure."""


class LLMClient(Protocol):
    def normalize_query(self, text: str) -> str:
        """Map a free-text query to the same attribute schema as image captions."""

    def complete(self, prompt: str) -> str:
        """Free-form completion for refine questions and reports."""


class DescriptionError(Exception):
    """Raised when a client cannot produce a usable description."""
