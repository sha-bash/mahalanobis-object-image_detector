from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from moid.adapters.base import ImageLike
from moid.adapters.gigachat_client import build_gigachat
from moid.prompts import SCHEMA_LINE, VLM_SYSTEM_PROMPT, compose_vlm_text

logger = logging.getLogger(__name__)


def load_image(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    return Image.open(image).convert("RGB")


def sidecar_caption_path(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


class StubVLM:
    """Read a .txt sidecar next to the image, or return a fallback caption."""

    def __init__(self, fallback: str | None = None) -> None:
        self.fallback = fallback or SCHEMA_LINE.replace("<value or unknown>", "unknown")
        self.prompt = VLM_SYSTEM_PROMPT

    def describe(self, image: ImageLike) -> str:
        if isinstance(image, (str, Path)):
            path = Path(image)
            sidecar = sidecar_caption_path(path)
            if sidecar.is_file():
                text = sidecar.read_text(encoding="utf-8").strip()
                if text:
                    return text
            logger.warning("StubVLM: no sidecar for %s, using fallback", path)
        return self.fallback


def _image_to_jpeg_bytes(image: ImageLike) -> bytes:
    pil = load_image(image)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _jpeg_filename(image: ImageLike) -> str:
    if isinstance(image, (str, Path)):
        return Path(image).with_suffix(".jpg").name
    return "image.jpg"


def _vision_human_message(file_id: str, text: str | None = None) -> Any:
    from langchain_core.messages import HumanMessage

    return HumanMessage(
        content_blocks=[
            {"type": "text", "text": text if text is not None else VLM_SYSTEM_PROMPT},
            {"type": "image", "file_id": file_id},
        ]
    )


class ContextualVLM:
    """Wrap a VLM so every describe() call includes a fixed prompt context."""

    def __init__(
        self,
        inner: Any,
        context: str,
        system_prompt: str | None = None,
    ) -> None:
        self._inner = inner
        self.context = context
        self.system_prompt = system_prompt

    def describe(self, image: ImageLike) -> str:
        try:
            return self._inner.describe(
                image, context=self.context, system_prompt=self.system_prompt
            )
        except TypeError:
            return self._inner.describe(image)


class GigaChatVLM:
    """Caption images with GigaChat Vision via GigaChain (`langchain-gigachat`)."""

    def __init__(
        self,
        credentials: str | None = None,
        model: str | None = None,
        verify_ssl_certs: bool = False,
        client: Any | None = None,
    ) -> None:
        self.credentials = credentials
        self.model = model
        self.verify_ssl_certs = verify_ssl_certs
        self._client = client

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client
        self._client = build_gigachat(
            credentials=self.credentials,
            model=self.model,
            verify_ssl_certs=self.verify_ssl_certs,
            max_tokens=512,
        )
        return self._client

    def describe(
        self,
        image: ImageLike,
        context: str = "",
        system_prompt: str | None = None,
    ) -> str:
        file_id: str | None = None
        client: Any | None = None
        try:
            client = self._client_or_raise()
            uploaded = client.upload_file((_jpeg_filename(image), _image_to_jpeg_bytes(image)))
            file_id = getattr(uploaded, "id_", None) or getattr(uploaded, "id", None)
            if not file_id:
                logger.error("GigaChatVLM: upload returned no file id")
                return ""
            reply = client.invoke(
                [
                    _vision_human_message(
                        file_id,
                        compose_vlm_text(
                            context,
                            system_prompt=system_prompt,
                            few_shot=system_prompt is None,
                        ),
                    )
                ]
            )
            return str(getattr(reply, "content", "") or "").strip()
        except Exception:
            logger.exception("GigaChatVLM request failed")
            return ""
        finally:
            if client is not None and file_id:
                try:
                    client.delete_file(file_id)
                except Exception:
                    logger.warning("GigaChatVLM: failed to delete uploaded file %s", file_id)
