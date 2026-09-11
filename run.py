"""One-command local setup and video processing (Python 3.11+, standard library only)."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
OLLAMA_VERSION = "0.34.0"
OLLAMA_SHA256 = "a7dd1b174f39d3d1b8a25d4cbc86045d0e190b17187bfdcbe2f2ee3b5a11470e"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def command(args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def prepare_python(root=ROOT):
    python = (
        root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    stamp = python.parent.parent / ".moid-dependencies"
    fingerprint = hashlib.sha256(
        (root / "pyproject.toml").read_bytes()
        + (root / "requirements/cpu.txt").read_bytes()
        + platform.python_version().encode()
    ).hexdigest()
    if not python.is_file():
        print("Creating isolated Python environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(python.parent.parent)
    if not stamp.is_file() or stamp.read_text() != fingerprint:
        print(
            "Installing CPU dependencies (first setup may take several minutes)...",
            flush=True,
        )
        command(
            [python, "-m", "pip", "install", "torch==2.14.0", "--index-url", CPU_INDEX]
        )
        command(
            [python, "-m", "pip", "install", "-c", root / "requirements/cpu.txt", root]
        )
        command([python, "-m", "pip", "check"])
        stamp.write_text(fingerprint)
    return python


def request_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "moid-local-setup"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def extract_verified_archive(archive, destination, expected_digest):
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != expected_digest.removeprefix("sha256:"):
        raise ValueError("Ollama archive checksum mismatch")
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Unsafe path in Ollama archive")
        bundle.extractall(destination)


def install_ollama(root=ROOT):
    if os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError(
            "Install Ollama from https://ollama.com/download and retry. "
            "Automatic portable setup supports Windows x64."
        )
    destination = root / "models/ollama"
    url = f"https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/ollama-windows-amd64.zip"
    downloads = root / ".cache/downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / f"ollama-{OLLAMA_VERSION}.zip"
    if not archive.is_file():
        temporary = archive.with_suffix(".part")
        print("Downloading portable Ollama from its official release...", flush=True)
        with urllib.request.urlopen(url, timeout=60) as source:
            with temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        temporary.replace(archive)
    extract_verified_archive(archive, destination, OLLAMA_SHA256)
    return destination / "ollama.exe"


def find_ollama(explicit=None, root=ROOT):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Ollama executable not found: {path}")
        return path
    candidates = [root / "models/ollama/ollama.exe", shutil.which("ollama")]
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(
            Path(os.environ["LOCALAPPDATA"]) / "Programs/Ollama/ollama.exe"
        )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return install_ollama(root)


@contextlib.contextmanager
def local_server(host, model, executable=None, root=ROOT):
    parsed = urlparse(host)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1"}
        or not parsed.port
    ):
        raise ValueError("--host must be a local HTTP address with an explicit port")
    process = None
    root.joinpath("reports").mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        try:
            try:
                tags = request_json(f"{host}/api/tags")
            except (urllib.error.URLError, TimeoutError):
                ollama = find_ollama(executable, root)
                env = os.environ.copy()
                env.update(
                    OLLAMA_HOST=parsed.netloc,
                    OLLAMA_MODELS=str(root / "models/ollama-models"),
                    OLLAMA_NUM_PARALLEL="1",
                    OLLAMA_MAX_LOADED_MODELS="1",
                )
                log = stack.enter_context(
                    (root / "reports/ollama.log").open("a", encoding="utf-8")
                )
                process = subprocess.Popen(
                    [str(ollama), "serve"],
                    env=env,
                    stdout=log,
                    stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                for _ in range(60):
                    if process.poll() is not None:
                        raise RuntimeError("Ollama exited; see reports/ollama.log")
                    try:
                        tags = request_json(f"{host}/api/tags")
                        break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(1)
                else:
                    raise RuntimeError(
                        "Ollama startup timed out; see reports/ollama.log"
                    )
            if model not in {entry["name"] for entry in tags["models"]}:
                print(f"Downloading local vision model {model}...", flush=True)
                payload = json.dumps({"model": model, "stream": True}).encode()
                request = urllib.request.Request(
                    f"{host}/api/pull",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                success = False
                with urllib.request.urlopen(request, timeout=600) as response:
                    for line in response:
                        status = json.loads(line)
                        if status.get("error"):
                            raise RuntimeError(status["error"])
                        success = status.get("status") == "success"
                if not success:
                    raise RuntimeError("Model download did not complete")
            yield
        finally:
            # Only stop a service created by this launch; never terminate an existing one.
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def input_file(value, label):
    if not value:
        if not sys.stdin.isatty():
            raise ValueError(f"Provide --{label.lower()} PATH for non-interactive use")
        value = input(f"{label} file path: ").strip().strip('"')
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} file does not exist: {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference")
    parser.add_argument("--video")
    parser.add_argument("--ref-box", type=int, nargs=4)
    parser.add_argument("--scene-context", default="")
    parser.add_argument("--model", default="qwen3.5:0.8b")
    parser.add_argument("--host", default="http://127.0.0.1:11435")
    parser.add_argument("--ollama", help="Optional existing Ollama executable")
    parser.add_argument("--tau", type=float, default=6.0)
    parser.add_argument("--verify-every", type=float, default=10.0)
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        parser.error("Python 3.11 or newer is required")
    if not args.setup_only:
        reference = input_file(args.reference, "Reference")
        video = input_file(args.video, "Video")
    python = prepare_python()
    # Run source directly so subsequent edits are used without reinstalling dependencies.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    with local_server(args.host.rstrip("/"), args.model, args.ollama):
        ready = ROOT / ".venv/.moid-sbert-ready"
        if args.setup_only or not ready.is_file():
            print("Preparing local SBERT weights...", flush=True)
            command(
                [
                    python,
                    "-m",
                    "moid.sbert_worker",
                    "--prepare",
                    "--model",
                    "all-MiniLM-L6-v2",
                ],
                env=env,
                cwd=ROOT,
            )
            ready.write_text("all-MiniLM-L6-v2")
        if args.setup_only:
            print("Setup complete. The local models and Python environment are ready.")
            return
        output = ROOT / "reports" / datetime.now().strftime("instance_%Y%m%d_%H%M%S_%f")
        cmd = [
            python,
            "-m",
            "moid.instance",
            "--reference",
            reference,
            "--video",
            video,
            "--out",
            output,
            "--host",
            args.host.rstrip("/"),
            "--model",
            args.model,
            "--tau",
            args.tau,
            "--verify-every",
            args.verify_every,
            "--scene-context",
            args.scene_context,
        ]
        if args.ref_box:
            cmd += ["--ref-box", *args.ref_box]
        command(cmd, env=env, cwd=ROOT)
        print(f"Video: {output / 'annotated.mp4'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
