from __future__ import annotations

import os
from typing import Any

DEFAULT_GIGACHAT_MODEL = "GigaChat-2-Pro"


def resolve_credentials(credentials: str | None = None) -> str:
    return credentials if credentials is not None else os.environ.get("GIGACHAT_CREDENTIALS", "")


def resolve_model(model: str | None = None) -> str:
    return model or os.environ.get("GIGACHAT_MODEL") or DEFAULT_GIGACHAT_MODEL


def build_gigachat(
    credentials: str | None = None,
    model: str | None = None,
    verify_ssl_certs: bool = False,
    max_tokens: int | None = None,
) -> Any:
    creds = resolve_credentials(credentials)
    if not creds:
        raise RuntimeError(
            "GigaChat credentials missing. Set GIGACHAT_CREDENTIALS or use stub adapters."
        )
    try:
        from langchain_gigachat import GigaChat
    except ImportError as e:
        raise RuntimeError(
            "Package langchain-gigachat is not installed. pip install moid[gigachat]"
        ) from e
    kwargs: dict[str, Any] = {
        "credentials": creds,
        "model": resolve_model(model),
        "verify_ssl_certs": verify_ssl_certs,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return GigaChat(**kwargs)
