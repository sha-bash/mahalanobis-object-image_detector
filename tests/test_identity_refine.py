from moid.adapters.llm import StubLLM
from moid.identity import build_identity_profile, propose_refine_questions, refine_profile


class ScriptedLLM:
    def __init__(self, line: str) -> None:
        self.line = line

    def normalize_query(self, text: str) -> str:
        return self.line

    def complete(self, prompt: str) -> str:
        return self.line


def test_refine_updates_markings():
    caps = [
        "object: passenger car; target_match: yes; category: vehicle; color: gray; "
        "markings: unknown; vehicle_class: passenger car; brand: unknown; model: unknown"
    ]
    profile = build_identity_profile(caps)
    assert profile.domain == "vehicle"
    llm = ScriptedLLM(
        "object: Lada Vesta; target_match: yes; category: vehicle; color: gray; "
        "markings: E661CX73; vehicle_class: passenger car; brand: Lada; model: Vesta"
    )
    refined = refine_profile(profile, "это Веста, номер E661CX73", llm)
    assert "E661CX73" in refined.markings
    assert "Vesta" in refined.fields.get("model", "")


def test_stub_questions_nonempty():
    profile = build_identity_profile(["object: box; target_match: yes; category: container"])
    qs = propose_refine_questions(profile, StubLLM())
    assert qs
