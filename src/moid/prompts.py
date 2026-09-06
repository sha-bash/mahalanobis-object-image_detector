"""Shared attribute schema for VLM captions and query normalization."""

from __future__ import annotations

from typing import Any

GENERIC_KEYS = (
    "object",
    "target_match",
    "category",
    "color",
    "size_class",
    "material",
    "shape",
    "parts",
    "distinctive_features",
    "markings",
    "text_on_object",
)

VEHICLE_KEYS = (
    "vehicle_class",
    "brand",
    "model",
    "body_style",
    "wheel_size",
    "wheel_to_body_ratio",
    "wheel_design",
    "tire_type",
    "grille",
    "headlights",
)

ATTRIBUTE_KEYS = GENERIC_KEYS + VEHICLE_KEYS

SCHEMA_LINE = "; ".join(f"{k}: <value or unknown>" for k in ATTRIBUTE_KEYS)

FEW_SHOT_CAPTION_EXAMPLES = """Examples (format only; do not copy values unless they match the current image):
object: cardboard shipping carton; target_match: yes; category: container; color: brown; size_class: medium; material: cardboard; shape: rectangular prism; parts: flaps, tape seam; distinctive_features: crushed corner on the lid; markings: barcode sticker; text_on_object: FRAGILE; vehicle_class: n/a; brand: unknown; model: n/a; body_style: n/a; wheel_size: n/a; wheel_to_body_ratio: n/a; wheel_design: n/a; tire_type: n/a; grille: n/a; headlights: n/a
object: compact hatchback; target_match: no; category: vehicle; color: blue; size_class: compact; material: painted metal; shape: hatchback silhouette; parts: license plate, side mirrors; distinctive_features: aftermarket roof spoiler; markings: unknown plate; text_on_object: unknown; vehicle_class: passenger car; brand: unknown; model: unknown; body_style: hatchback; wheel_size: normal; wheel_to_body_ratio: normal; wheel_design: five-spoke alloy; tire_type: road; grille: honeycomb; headlights: projector; 
object: empty asphalt patch; target_match: no; category: background; color: gray; size_class: unknown; material: asphalt; shape: irregular; parts: none; distinctive_features: none; markings: none; text_on_object: none; vehicle_class: non-vehicle; brand: unknown; model: n/a; body_style: n/a; wheel_size: n/a; wheel_to_body_ratio: n/a; wheel_design: n/a; tire_type: n/a; grille: n/a; headlights: n/a
"""

VLM_SYSTEM_PROMPT = f"""You describe the dominant physical object in an image as a compact English attribute list.

Output exactly one line in this format:
{SCHEMA_LINE}

Rules:
- Ignore background, terrain, sky, lighting, shadows, weather, UI chrome, and whether the image is a 3D render, a photograph, or a screenshot.
- Do NOT mention viewpoint, camera angle, or phrases like "from the side", "from above", "top-down", or "3/4 view".
- target_match is yes, no, or uncertain. Use yes only when the target identity is supported by visible evidence in the current image.
- Fill every generic field from what is visible. If an attribute is not visible, write unknown.
- category is a coarse class (vehicle, plant, animal, container, device, person, background, other).
- If the object is not a vehicle, set vehicle_class: n/a (or non-vehicle) and the remaining vehicle fields to n/a. Do not invent a make.
- If the object is a vehicle, set vehicle_class to passenger car, SUV, truck, off-road buggy, special vehicle, motorcycle, or unknown. Fill brand and model only when a badge, grille, or silhouette supports them; otherwise unknown.
- distinctive_features and markings must be specific to THIS image. Never copy identity text from the prompt unless it is visible here.
- No extra sentences, markdown, or commentary.
"""

REFERENCE_EXTRACT_PROMPT = """This image is a REFERENCE of the object we will search for later.
Set target_match: yes.
Extract the maximum amount of visible identity detail. Do not invent brand, model, text, or parts that are not on this photo.
If optional user hints are provided, use them only when they do not contradict the image.
"""

LLM_NORMALIZE_PROMPT = f"""Convert the user's object query into the same English attribute line used for image captions.

Output exactly one line:
{SCHEMA_LINE}

Rules:
- Input may be Russian or English; output MUST be English.
- Fill only attributes stated or clearly implied. Everything else is unknown or n/a.
- Fill target_match: yes only when the query explicitly describes the target; otherwise uncertain.
- Do not add scene, viewpoint, lighting, or background.
- No extra sentences, markdown, or commentary.

User query:
"""

REFINE_PROFILE_PROMPT = f"""Merge the current object profile with the user's clarification.

Return exactly one English attribute line:
{SCHEMA_LINE}

Rules:
- Keep facts that still agree with the reference captions.
- Apply user clarifications (any language) as English field values.
- Do not invent unseen attributes. Unmentioned fields stay as in the current profile.
- target_match must be yes for the reference identity.
- No extra sentences.

Current profile:
{{profile}}

Reference captions:
{{captions}}

User clarification:
{{user_text}}
"""

REFINE_QUESTIONS_PROMPT = """The following identity profile was extracted from reference photos. Some fields are unknown.

Ask up to 5 short clarifying questions in the user's language (default Russian) that would most improve later search.
Return a numbered list only, no preamble.

Profile:
{profile}
"""

REPORT_PROMPT = """Write a concise experiment report in Markdown (Russian).

Include:
- What object was sought (from the identity profile)
- Whether the user added clarifications
- Per-image: filename, found or not (frame_positive), best distance, short caption
- Likely false positives / misses if obvious from the numbers
- No raw JSON dump

Profile:
{profile}

User clarification:
{clarification}

Results JSON:
{results}
"""


def schema_line_from_fields(values: dict[str, Any]) -> str:
    parts = []
    for key in ATTRIBUTE_KEYS:
        parts.append(f"{key}: {values.get(key, 'unknown')}")
    return "; ".join(parts)


def compose_vlm_text(context: str = "", system_prompt: str | None = None, *, few_shot: bool = True) -> str:
    base = VLM_SYSTEM_PROMPT if system_prompt is None else system_prompt
    chunks = [base.rstrip()]
    if few_shot:
        chunks.append(FEW_SHOT_CAPTION_EXAMPLES.strip())
    extra = (context or "").strip()
    if extra:
        chunks.append(extra)
    return "\n\n".join(chunks)


def build_reference_context(hints_text: str = "", known_traits: list[str] | None = None) -> str:
    traits = ", ".join(t for t in (known_traits or []) if t.strip())
    hint = (hints_text or "").strip()
    extra = ""
    if hint or traits:
        extra = f"Optional user hints (do not override visible evidence): {hint} {traits}".strip()
    return f"{REFERENCE_EXTRACT_PROMPT.strip()}\n{extra}".strip()


def build_search_context(profile_line: str, hints_text: str = "") -> str:
    hint = (hints_text or "").strip()
    hint_line = f"Optional extra hints: {hint}" if hint else ""
    return f"""The target identity was extracted from reference photos:
{profile_line}

First classify the visible object independently, then compare it with that profile.
Set target_match: yes only with strong evidence in THIS crop (unique markings/text, or several distinctive features together).
Set target_match: no when the class is incompatible or it is clearly a different instance.
If evidence is incomplete, set target_match: uncertain. Do not upgrade uncertain to yes.
Do NOT copy brand, model, markings, or text_on_object from the target profile unless they are visible on this crop.
If there is no relevant object, set target_match: no; category: background.
{hint_line}
""".strip()
