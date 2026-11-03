"""Where and with what code a run happened."""
from __future__ import annotations

import getpass
import importlib.metadata
import platform
import socket
import subprocess
import sys
from pathlib import Path

_PACKAGES = ("torch", "transformers", "smolagents", "qwen-agent", "datasets", "huggingface-hub", "numpy", "pydantic")


def git_state(path: Path) -> dict:
    """The harness's commit and whether the working tree had local changes."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:  # noqa: BLE001 - provenance is best-effort
            return None

    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def package_versions() -> dict:
    out = {}
    for name in _PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def environment() -> dict:
    return {
        "git": git_state(Path(__file__).resolve().parent),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "argv": list(sys.argv),
        "packages": package_versions(),
    }
