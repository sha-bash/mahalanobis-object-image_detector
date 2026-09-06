from __future__ import annotations

import logging
from typing import Any

from moid.adapters.gigachat_client import build_gigachat
from moid.prompts import ATTRIBUTE_KEYS, LLM_NORMALIZE_PROMPT, SCHEMA_LINE

logger = logging.getLogger(__name__)


class StubLLM:
    """Return a canned attribute line, or wrap the query if it already looks like the schema."""

    def normalize_query(self, text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return SCHEMA_LINE.replace("<value or unknown>", "unknown")
        if "object:" in raw.lower() and ";" in raw:
            return raw
        values = {
            key: (raw if key == "object" else "unknown")
            for key in ATTRIBUTE_KEYS
        }
        return "; ".join(f"{key}: {value}" for key, value in values.items())

    def complete(self, prompt: str) -> str:
        low = (prompt or "").lower()
        if "ask up to" in low or "questions" in low:
            return "1. What unique markings or text are on the object?\n2. What color and size class should we lock in?"
        if "markdown" in low or "experiment report" in low:
            return "# Отчёт\n\nStub LLM: отчёт построен по результатам поиска.\n"
        return self.normalize_query(prompt)


class GigaChatLLM:
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
        )
        return self._client

    def normalize_query(self, text: str) -> str:
        return self.complete(LLM_NORMALIZE_PROMPT + text.strip())

    def complete(self, prompt: str) -> str:
        try:
            client = self._client_or_raise()
            reply = client.invoke(prompt)
            return str(getattr(reply, "content", "") or "").strip()
        except Exception:
            logger.exception("GigaChat LLM request failed")
            return ""
