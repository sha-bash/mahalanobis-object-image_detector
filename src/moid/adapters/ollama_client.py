from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5vl:3b"


def resolve_host(host: str | None = None) -> str:
    value = host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
    return value.rstrip("/")


def resolve_vlm_model(model: str | None = None) -> str:
    return model or os.environ.get("OLLAMA_VLM_MODEL") or DEFAULT_OLLAMA_MODEL


def resolve_llm_model(model: str | None = None) -> str:
    return model or os.environ.get("OLLAMA_LLM_MODEL") or DEFAULT_OLLAMA_MODEL


def ollama_chat(
    *,
    host: str,
    model: str,
    messages: list[dict[str, Any]],
    timeout_sec: float = 180.0,
    temperature: float = 0.0,
    num_predict: int = 512,
    num_ctx: int | None = 2048,
    keep_alive: str = "5m",
    response_format: str | dict[str, Any] | None = None,
    num_thread: int | None = None,
    num_gpu: int | None = None,
    think: bool | None = None,
    structured_response_fallback: bool = False,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    if num_ctx is not None:
        payload["options"]["num_ctx"] = int(num_ctx)
    if num_thread is not None:
        payload["options"]["num_thread"] = int(num_thread)
    if num_gpu is not None:
        payload["options"]["num_gpu"] = int(num_gpu)
    if response_format is not None:
        payload["format"] = response_format
    if think is not None:
        payload["think"] = think
    request = Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {host}. Start `ollama serve` and pull `{model}`."
        ) from exc
    message = body.get("message") or {}
    content = str(message.get("content") or "").strip()
    # Some Ollama/model-template combinations route an entire JSON answer into
    # `thinking` even with think=False. Accept only a complete JSON object in
    # this explicit compatibility mode; never use free-form reasoning as output.
    if not content and structured_response_fallback and think is False and response_format:
        candidate = str(message.get("thinking") or "").strip()
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)
    return content
