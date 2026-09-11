from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from moid.adapters.gigachat_client import build_gigachat
from moid.adapters.llm import GigaChatLLM, StubLLM
from moid.adapters.vlm import GigaChatVLM, StubVLM
from moid.config import MoidConfig, load_config
from moid.factory import build_llm, build_vlm
from moid.prompts import ATTRIBUTE_KEYS, VLM_SYSTEM_PROMPT


def test_stub_llm_wraps_free_text():
    out = StubLLM().normalize_query("красный грузовик")
    assert "object:" in out.lower()
    assert "грузовик" in out or "unknown" in out


def test_compose_vlm_text_uses_custom_system_prompt():
    from moid.prompts import compose_vlm_text

    custom = "CUSTOM SYSTEM"
    out = compose_vlm_text("контекст цели", system_prompt=custom, few_shot=False)
    assert "контекст цели" in out
    assert "CUSTOM SYSTEM" in out
    assert VLM_SYSTEM_PROMPT not in out


def test_search_context_does_not_hardcode_vesta():
    from moid.prompts import build_search_context

    ctx = build_search_context("object: green tree; markings: none")
    assert "Vesta" not in ctx
    assert "green tree" in ctx


def test_vlm_prompt_forbids_viewpoint():
    low = VLM_SYSTEM_PROMPT.lower()
    assert "viewpoint" in low or "viewed from" in low
    for key in ATTRIBUTE_KEYS:
        assert key in VLM_SYSTEM_PROMPT


def test_stub_vlm_sidecar(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(img)
    (tmp_path / "a.txt").write_text("object: crate", encoding="utf-8")
    assert StubVLM().describe(img) == "object: crate"


def test_build_gigachat_requires_credentials(monkeypatch):
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    with pytest.raises(RuntimeError, match="GIGACHAT_CREDENTIALS"):
        build_gigachat()


def test_gigachat_vlm_upload_invoke_delete(monkeypatch):
    monkeypatch.setattr(
        "moid.adapters.vlm._vision_human_message",
        lambda file_id, text=None: SimpleNamespace(
            content_blocks=[
                {"type": "text", "text": text or VLM_SYSTEM_PROMPT},
                {"type": "image", "file_id": file_id},
            ]
        ),
    )
    client = MagicMock()
    client.upload_file.return_value = SimpleNamespace(id_="file-1")
    client.invoke.return_value = SimpleNamespace(
        content="object: crate; brand: unknown; model: unknown; color: unknown; body_style: unknown; parts: unknown; markings: unknown"
    )
    img = Image.new("RGB", (8, 8), (0, 0, 0))
    out = GigaChatVLM(client=client).describe(img)
    assert "crate" in out
    name, data = client.upload_file.call_args[0][0]
    assert name.endswith(".jpg")
    assert isinstance(data, bytes) and data
    message = client.invoke.call_args[0][0][0]
    blocks = message.content_blocks
    assert blocks[0]["type"] == "text"
    assert VLM_SYSTEM_PROMPT in blocks[0]["text"]
    assert blocks[1] == {"type": "image", "file_id": "file-1"}
    client.delete_file.assert_called_once_with("file-1")


def test_gigachat_vlm_passes_context(monkeypatch):
    monkeypatch.setattr(
        "moid.adapters.vlm._vision_human_message",
        lambda file_id, text=None: SimpleNamespace(
            content_blocks=[
                {"type": "text", "text": text or ""},
                {"type": "image", "file_id": file_id},
            ]
        ),
    )
    client = MagicMock()
    client.upload_file.return_value = SimpleNamespace(id_="file-3")
    client.invoke.return_value = SimpleNamespace(content="object: car; brand: Lada; model: Vesta")
    img = Image.new("RGB", (4, 4), (2, 2, 2))
    GigaChatVLM(client=client).describe(img, context="TARGET: Lada Vesta E661CX73")
    text = client.invoke.call_args[0][0][0].content_blocks[0]["text"]
    assert VLM_SYSTEM_PROMPT.strip() in text
    assert "E661CX73" in text
    assert "Lada Vesta" in text


def test_gigachat_vlm_passes_custom_system_prompt(monkeypatch):
    custom = "CUSTOM SYSTEM PROMPT"

    monkeypatch.setattr(
        "moid.adapters.vlm._vision_human_message",
        lambda file_id, text=None: SimpleNamespace(
            content_blocks=[
                {"type": "text", "text": text or ""},
                {"type": "image", "file_id": file_id},
            ]
        ),
    )
    client = MagicMock()
    client.upload_file.return_value = SimpleNamespace(id_="file-4")
    client.invoke.return_value = SimpleNamespace(content="object: car")
    img = Image.new("RGB", (4, 4), (3, 3, 3))
    GigaChatVLM(client=client).describe(
        img, context="цель: объект", system_prompt=custom
    )
    text = client.invoke.call_args[0][0][0].content_blocks[0]["text"]
    assert "CUSTOM SYSTEM PROMPT" in text
    assert "цель: объект" in text
    assert VLM_SYSTEM_PROMPT not in text


def test_gigachat_vlm_deletes_file_if_invoke_fails(monkeypatch):
    monkeypatch.setattr(
        "moid.adapters.vlm._vision_human_message",
        lambda file_id, text=None: SimpleNamespace(content_blocks=[]),
    )
    client = MagicMock()
    client.upload_file.return_value = SimpleNamespace(id_="file-2")
    client.invoke.side_effect = RuntimeError("boom")
    img = Image.new("RGB", (4, 4), (1, 1, 1))
    assert GigaChatVLM(client=client).describe(img) == ""
    client.delete_file.assert_called_once_with("file-2")


def test_gigachat_llm_invoke():
    client = MagicMock()
    client.invoke.return_value = SimpleNamespace(
        content="object: truck; brand: unknown; model: unknown; color: red; body_style: unknown; parts: unknown; markings: unknown"
    )
    out = GigaChatLLM(client=client).normalize_query("red truck")
    assert "truck" in out
    prompt = client.invoke.call_args[0][0]
    assert "red truck" in prompt


def test_factory_builds_gigachat_adapters():
    cfg = MoidConfig()
    cfg.adapters.vlm = "gigachat"
    cfg.adapters.llm = "gigachat"
    assert isinstance(build_vlm(cfg), GigaChatVLM)
    assert isinstance(build_llm(cfg), GigaChatLLM)
    assert isinstance(build_vlm(cfg, stub=True), StubVLM)
    assert isinstance(build_llm(cfg, stub=True), StubLLM)


def test_default_yaml_adapters():
    cfg = load_config("configs/default.yaml")
    assert cfg.adapters.vlm == "gigachat"
    assert cfg.adapters.llm == "gigachat"
    assert cfg.adapters.gigachat_model == "GigaChat-3-Ultra"
    assert cfg.detector.threshold_margin == 0.8
    assert cfg.grid.extra_scales == [[2, 2]]
    assert cfg.grid.include_best_if_none_accepted is False
    assert cfg.regions.backend == "grid"
    assert cfg.decision.target_match_gate is True
    assert cfg.ocr.backend == "none"
    assert cfg.paths.refs == "data/refs"
    assert cfg.detector.projector_path is None
