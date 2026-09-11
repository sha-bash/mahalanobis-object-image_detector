from __future__ import annotations

import re
from typing import Literal

from moid.prompts import ATTRIBUTE_KEYS

MatchValue = Literal["yes", "no", "uncertain", "unknown"]

_PAIR = re.compile(r"([a-z_]+)\s*:\s*([^;]+)", re.IGNORECASE)


def parse_caption(text: str) -> dict[str, str]:
    raw = (text or "").strip()
    fields = {key: "unknown" for key in ATTRIBUTE_KEYS}
    if not raw:
        return fields
    for match in _PAIR.finditer(raw):
        key = match.group(1).strip().lower()
        value = match.group(2).strip() or "unknown"
        if key in fields:
            fields[key] = value
    return fields


def target_match_value(text_or_fields: str | dict[str, str]) -> MatchValue:
    if isinstance(text_or_fields, dict):
        raw = str(text_or_fields.get("target_match", "unknown"))
    else:
        raw = parse_caption(text_or_fields).get("target_match", "unknown")
    token = raw.strip().lower()
    if token in {"yes", "y", "true"}:
        return "yes"
    if token in {"no", "n", "false"}:
        return "no"
    if token in {"uncertain", "unsure", "maybe"}:
        return "uncertain"
    return "unknown"


def crop_coverage_value(text_or_fields: str | dict[str, str]) -> str:
    if isinstance(text_or_fields, dict):
        raw = str(text_or_fields.get("crop_coverage", "unknown"))
    else:
        raw = parse_caption(text_or_fields).get("crop_coverage", "unknown")
    token = raw.strip().lower()
    if token in {"full", "complete", "entire"}:
        return "full"
    if token in {"partial", "cropped", "cut", "clipped"}:
        return "partial"
    if token in {"none", "no", "empty", "background"}:
        return "none"
    return "unknown"


def looks_like_vehicle(fields: dict[str, str]) -> bool:
    vehicle_class = fields.get("vehicle_class", "").lower()
    if vehicle_class in {"n/a", "na", "non-vehicle", "none", "unknown", ""}:
        pass
    elif vehicle_class:
        return True
    blob = " ".join(
        fields.get(k, "") for k in ("object", "category", "vehicle_class", "body_style")
    ).lower()
    tokens = ("car", "sedan", "vehicle", "truck", "suv", "van", "hatchback", "wagon", "motorcycle")
    return any(tok in blob for tok in tokens)
