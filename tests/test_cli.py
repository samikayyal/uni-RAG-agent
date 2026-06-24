from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import uni_rag_agent

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "uni_rag_agent", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_package_importable() -> None:
    assert uni_rag_agent.__version__


def test_help_exits_successfully() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    assert "uv run -m uni_rag_agent config check" in result.stdout
    assert "uv run -m uni_rag_agent inventory run" in result.stdout


def test_unknown_command_returns_nonzero_with_message() -> None:
    result = run_cli("unknown")

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert "unknown" in result.stderr


def test_registered_stub_command_fails_clearly() -> None:
    result = run_cli("inventory", "run")

    assert result.returncode == 1
    assert "not implemented yet" in result.stderr
    assert "Feature Spec 03" in result.stderr


def test_env_example_exists_and_env_is_ignored() -> None:
    assert (REPO_ROOT / ".env.example").is_file()

    ignored_patterns = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert ".env" in ignored_patterns
    assert ".env.example" not in ignored_patterns
