from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from moid.captions import looks_like_vehicle, parse_caption
from moid.prompts import (
    ATTRIBUTE_KEYS,
    REFINE_PROFILE_PROMPT,
    REFINE_QUESTIONS_PROMPT,
    schema_line_from_fields,
)

Domain = Literal["generic", "vehicle"]


@dataclass
class IdentityProfile:
    fields: dict[str, str] = field(default_factory=dict)
    captions: list[str] = field(default_factory=list)
    domain: Domain = "generic"
    user_clarification: str = ""

    @property
    def object(self) -> str:
        return self.fields.get("object", "unknown")

    @property
    def markings(self) -> str:
        return self.fields.get("markings", "unknown")

    def as_line(self) -> str:
        return schema_line_from_fields(self.fields)

    def unknown_fields(self) -> list[str]:
        skip = {"n/a", "na", "unknown", "none", ""}
        keys = ATTRIBUTE_KEYS if self.domain == "vehicle" else ATTRIBUTE_KEYS[:11]
        return [k for k in keys if self.fields.get(k, "unknown").strip().lower() in skip]


def _majority(values: list[str]) -> str:
    cleaned = [v.strip() for v in values if v and v.strip().lower() not in {"unknown", "n/a", "na", ""}]
    if not cleaned:
        return "unknown"
    return Counter(cleaned).most_common(1)[0][0]


def build_identity_profile(
    captions: list[str],
    hints_text: str = "",
    known_traits: list[str] | None = None,
) -> IdentityProfile:
    parsed = [parse_caption(c) for c in captions if (c or "").strip()]
    fields = {key: "unknown" for key in ATTRIBUTE_KEYS}
    if parsed:
        for key in ATTRIBUTE_KEYS:
            fields[key] = _majority([row.get(key, "unknown") for row in parsed])
    if hints_text.strip() and fields.get("object", "unknown") == "unknown":
        fields["object"] = hints_text.strip()
    traits = [t.strip() for t in (known_traits or []) if t.strip()]
    if traits and fields.get("distinctive_features", "unknown") in {"unknown", "n/a"}:
        fields["distinctive_features"] = "; ".join(traits)
    fields["target_match"] = "yes"
    domain: Domain = "vehicle" if looks_like_vehicle(fields) else "generic"
    return IdentityProfile(fields=fields, captions=list(captions), domain=domain)


def propose_refine_questions(profile: IdentityProfile, llm: Any) -> list[str]:
    prompt = REFINE_QUESTIONS_PROMPT.format(profile=profile.as_line())
    raw = (llm.complete(prompt) if hasattr(llm, "complete") else llm.normalize_query(prompt)) or ""
    questions = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = stripped.lstrip("0123456789.-) ").strip()
        if stripped:
            questions.append(stripped)
    return questions[:5]


def refine_profile(profile: IdentityProfile, user_text: str, llm: Any) -> IdentityProfile:
    text = (user_text or "").strip()
    if not text:
        return profile
    prompt = REFINE_PROFILE_PROMPT.format(
        profile=profile.as_line(),
        captions="\n".join(profile.captions) or "(none)",
        user_text=text,
    )
    line = (llm.complete(prompt) if hasattr(llm, "complete") else llm.normalize_query(prompt)) or ""
    merged = parse_caption(line if line.strip() else profile.as_line())
    for key, value in profile.fields.items():
        if merged.get(key, "unknown").lower() in {"unknown", ""}:
            merged[key] = value
    merged["target_match"] = "yes"
    captions = [schema_line_from_fields(merged)] + list(profile.captions)
    domain: Domain = "vehicle" if looks_like_vehicle(merged) else "generic"
    return IdentityProfile(
        fields=merged,
        captions=captions,
        domain=domain,
        user_clarification=text,
    )


def apply_profile_to_captions(captions: list[str], profile: IdentityProfile) -> list[str]:
    """Keep original captions; prepend a canonical profile line used in clustering."""
    line = profile.as_line()
    if not captions:
        return [line]
    if captions[0] == line:
        return list(captions)
    return [line, *captions]
