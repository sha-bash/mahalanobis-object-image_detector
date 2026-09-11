import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock
import urllib.error
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "moid_bootstrap", Path(__file__).parents[1] / "run.py"
)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def test_archive_checksum_and_path_validation(tmp_path):
    archive = tmp_path / "ollama.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("ollama.exe", b"test fixture")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    bootstrap.extract_verified_archive(archive, tmp_path / "valid", digest)
    assert (tmp_path / "valid/ollama.exe").read_bytes() == b"test fixture"
    with pytest.raises(ValueError, match="checksum"):
        bootstrap.extract_verified_archive(archive, tmp_path / "invalid", "0" * 64)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.exe", b"test fixture")
    with pytest.raises(ValueError, match="Unsafe"):
        bootstrap.extract_verified_archive(
            archive,
            tmp_path / "unsafe",
            hashlib.sha256(archive.read_bytes()).hexdigest(),
        )
    assert not (tmp_path / "outside.exe").exists()


def test_existing_server_is_reused_without_starting_or_stopping(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bootstrap, "request_json", lambda url: {"models": [{"name": "vision"}]}
    )
    find = Mock(side_effect=AssertionError("Existing server needs no installation"))
    monkeypatch.setattr(bootstrap, "find_ollama", find)
    with bootstrap.local_server("http://127.0.0.1:11435", "vision", root=tmp_path):
        pass
    find.assert_not_called()


def test_owned_server_is_stopped_on_failure(monkeypatch, tmp_path):
    query = Mock(
        side_effect=[urllib.error.URLError("offline"), {"models": [{"name": "vision"}]}]
    )
    monkeypatch.setattr(bootstrap, "request_json", query)
    monkeypatch.setattr(bootstrap, "find_ollama", lambda *args: Path("ollama.exe"))
    process = Mock()
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr(bootstrap.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="processing failed"):
        with bootstrap.local_server("http://127.0.0.1:11435", "vision", root=tmp_path):
            raise RuntimeError("processing failed")
    process.terminate.assert_called_once()
    process.wait.assert_called_once()
    assert popen.call_args.kwargs["env"]["OLLAMA_NUM_PARALLEL"] == "1"


def test_missing_inputs_fail_before_installing(monkeypatch):
    install = Mock(side_effect=AssertionError("Must validate inputs first"))
    monkeypatch.setattr(bootstrap, "prepare_python", install)
    with pytest.raises(ValueError, match="does not exist"):
        bootstrap.main(
            ["--reference", "missing-reference.jpg", "--video", "missing-video.mp4"]
        )
    install.assert_not_called()


def test_nonlocal_server_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="local HTTP"):
        with bootstrap.local_server(
            "http://example.com:11434", "vision", root=tmp_path
        ):
            pass


def test_download_failure_does_not_announce_ready(monkeypatch, tmp_path):
    import io

    monkeypatch.setattr(bootstrap, "request_json", lambda url: {"models": []})
    monkeypatch.setattr(
        bootstrap.urllib.request,
        "urlopen",
        lambda *a, **kw: io.BytesIO(json.dumps({"error": "download failed"}).encode()),
    )
    with pytest.raises(RuntimeError, match="download failed"):
        with bootstrap.local_server("http://127.0.0.1:11435", "vision", root=tmp_path):
            pytest.fail("Failed download must not permit processing")
