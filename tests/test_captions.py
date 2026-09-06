from moid.captions import looks_like_vehicle, parse_caption, target_match_value
from moid.identity import build_identity_profile
from moid.prompts import ATTRIBUTE_KEYS, VLM_SYSTEM_PROMPT, build_search_context
from moid.scoring import ScoredBox, apply_target_match_gate
from moid.regions.base import BBox


def test_parse_caption_keys():
    line = "object: green tree; target_match: no; category: plant; color: green; markings: none"
    fields = parse_caption(line)
    assert fields["object"] == "green tree"
    assert target_match_value(fields) == "no"
    assert not looks_like_vehicle(fields)
    for key in ATTRIBUTE_KEYS:
        assert key in fields


def test_profile_from_tree_captions():
    caps = [
        "object: green deciduous tree; target_match: yes; category: plant; color: green; "
        "distinctive_features: dense crown; markings: none; vehicle_class: n/a"
    ]
    profile = build_identity_profile(caps)
    assert profile.object == "green deciduous tree"
    assert profile.domain == "generic"
    ctx = build_search_context(profile.as_line())
    assert "Lada" not in ctx
    assert "green deciduous tree" in ctx


def test_prompt_forbids_viewpoint():
    low = VLM_SYSTEM_PROMPT.lower()
    assert "viewpoint" in low
    for key in ATTRIBUTE_KEYS:
        assert key in VLM_SYSTEM_PROMPT


def test_gate_rejects_target_match_no():
    box = ScoredBox(
        BBox(0, 0, 10, 10),
        distance=0.1,
        threshold=5.0,
        accepted=True,
        caption="object: sedan; target_match: no",
        target_match="no",
    )
    gated = apply_target_match_gate([box], gated=True)
    assert gated[0].accepted is False
