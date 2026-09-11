from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from moid.adapters.base import ImageLike
from moid.adapters.gigachat_client import build_gigachat
from moid.adapters.ollama_client import ollama_chat, resolve_host, resolve_vlm_model
from moid.prompts import (
    SCHEMA_LINE,
    VLM_SYSTEM_PROMPT,
    attribute_json_schema,
    caption_from_json_payload,
    compose_vlm_text,
)

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


def _image_to_jpeg_bytes(
    image: ImageLike,
    *,
    quality: int = 90,
    max_side: int | None = None,
) -> bytes:
    pil = load_image(image)
    if max_side and max(pil.size) > max_side:
        pil = pil.copy()
        pil.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
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
        max_image_side: int | None = None,
    ) -> None:
        self._inner = inner
        self.context = context
        self.system_prompt = system_prompt
        self.max_image_side = max_image_side

    def describe(self, image: ImageLike) -> str:
        kwargs: dict[str, Any] = {
            "context": self.context,
            "system_prompt": self.system_prompt,
        }
        if self.max_image_side is not None:
            kwargs["max_image_side"] = self.max_image_side
        try:
            return self._inner.describe(image, **kwargs)
        except TypeError:
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


class OllamaVLM:
    """Caption images with a local Ollama vision model via `/api/chat`."""

    def __init__(
        self,
        *,
        host: str | None = None,
        model: str | None = None,
        timeout_sec: float = 180.0,
        keep_alive: str = "5m",
        num_ctx: int = 2048,
        num_predict: int = 512,
        max_image_side: int = 768,
        jpeg_quality: int = 85,
        cache: bool = True,
        cache_dir: str | Path | None = None,
        chat_fn=None,
    ) -> None:
        self.host = resolve_host(host)
        self.model = resolve_vlm_model(model)
        self.timeout_sec = timeout_sec
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.max_image_side = max_image_side
        self.jpeg_quality = jpeg_quality
        self.cache_enabled = cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._chat = chat_fn or ollama_chat
        self._memory_cache: dict[str, str] = {}

    def describe(
        self,
        image: ImageLike,
        context: str = "",
        system_prompt: str | None = None,
        max_image_side: int | None = None,
    ) -> str:
        prompt = compose_vlm_text(
            context,
            system_prompt=system_prompt,
            few_shot=system_prompt is None,
        )
        side = max_image_side if max_image_side is not None else self.max_image_side
        try:
            jpeg = _image_to_jpeg_bytes(image, quality=self.jpeg_quality, max_side=side)
            cache_key = self._cache_key(prompt, jpeg)
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached
            caption = self._describe_jpeg(jpeg, prompt)
            if caption:
                self._cache_put(cache_key, caption)
            return caption
        except Exception:
            logger.exception("OllamaVLM request failed")
            return ""

    def _describe_jpeg(self, jpeg: bytes, prompt: str) -> str:
        import base64

        encoded = base64.b64encode(jpeg).decode("ascii")
        messages = [{"role": "user", "content": prompt, "images": [encoded]}]
        schema = attribute_json_schema()
        raw = self._invoke(messages, schema)
        try:
            return caption_from_json_payload(raw)
        except (ValueError, json.JSONDecodeError):
            retry_messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": "Return only valid JSON matching the required schema. Fill every field.",
                },
            ]
            raw = self._invoke(retry_messages, schema)
            try:
                return caption_from_json_payload(raw)
            except (ValueError, json.JSONDecodeError):
                logger.error("OllamaVLM returned a caption that does not match the schema")
                return ""

    def _invoke(self, messages: list[dict], schema: dict) -> str:
        return self._chat(
            host=self.host,
            model=self.model,
            messages=messages,
            timeout_sec=self.timeout_sec,
            temperature=0.0,
            num_predict=self.num_predict,
            num_ctx=self.num_ctx,
            keep_alive=self.keep_alive,
            response_format=schema,
        )

    def _cache_key(self, prompt: str, jpeg: bytes) -> str:
        import hashlib

        digest = hashlib.sha256()
        digest.update(self.model.encode("utf-8"))
        digest.update(b"\0")
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\0")
        digest.update(jpeg)
        return digest.hexdigest()

    def _cache_get(self, key: str) -> str | None:
        if not self.cache_enabled:
            return None
        if key in self._memory_cache:
            return self._memory_cache[key]
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.txt"
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                self._memory_cache[key] = text
                return text
        return None

    def _cache_put(self, key: str, caption: str) -> None:
        if not self.cache_enabled:
            return
        self._memory_cache[key] = caption
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.txt").write_text(caption, encoding="utf-8")
