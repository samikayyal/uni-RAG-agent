from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import uni_rag_agent

REPO_ROOT = Path(__file__).resolve().parents[1]
UNI_RAG_ENV_PREFIX = "UNI_RAG_"


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
    result = run_cli("extract", "run")

    assert result.returncode == 1
    assert "not implemented yet" in result.stderr
    assert "Feature Spec 04" in result.stderr


def test_inventory_run_cli_fills_temp_storage(tmp_path: Path) -> None:
    courses_root = tmp_path / "Courses"
    data_dir = tmp_path / "data"
    course_dir = courses_root / "Information Retrieval"
    course_dir.mkdir(parents=True)
    (course_dir / "syllabus.txt").write_text("BM25", encoding="utf-8")
    (course_dir / "lecture.mp4").write_bytes(b"mp4")
    env = _subprocess_env(
        {
            "UNI_RAG_COURSES_ROOT": str(courses_root),
            "UNI_RAG_DATA_DIR": str(data_dir),
            "UNI_RAG_SQLITE_PATH": str(data_dir / "uni_rag.sqlite"),
            "UNI_RAG_CHROMA_DIR": str(data_dir / "indexes" / "vector"),
            "UNI_RAG_RUNS_DIR": str(data_dir / "runs"),
            "UNI_RAG_USE_FAKE_LLM": "true",
            "UNI_RAG_USE_FAKE_EMBEDDINGS": "true",
        }
    )

    run_result = subprocess.run(
        [sys.executable, "-m", "uni_rag_agent", "inventory", "run"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    summary_result = subprocess.run(
        [sys.executable, "-m", "uni_rag_agent", "inventory", "summary"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert run_result.returncode == 0, run_result.stderr
    assert "Inventory run completed" in run_result.stdout
    assert "courses_seen: 1" in run_result.stdout
    assert "files_seen: 2" in run_result.stdout
    assert "files_pending: 1" in run_result.stdout
    assert "files_metadata_only: 1" in run_result.stdout
    assert (data_dir / "uni_rag.sqlite").is_file()

    assert summary_result.returncode == 0, summary_result.stderr
    assert "Inventory summary" in summary_result.stdout
    assert "files_total: 2" in summary_result.stdout
    assert "Information Retrieval: files=2" in summary_result.stdout


def test_env_example_exists_and_env_is_ignored() -> None:
    assert (REPO_ROOT / ".env.example").is_file()

    ignored_patterns = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert ".env" in ignored_patterns
    assert ".env.example" not in ignored_patterns


def _subprocess_env(overrides: dict[str, str]) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(UNI_RAG_ENV_PREFIX)
    }
    env.update(overrides)
    return env
