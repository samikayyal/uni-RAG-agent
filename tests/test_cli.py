from __future__ import annotations

import json
import os
import sqlite3
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
    result = run_cli("index", "keyword")

    assert result.returncode == 1
    assert "not implemented yet" in result.stderr
    assert "Feature Spec 06" in result.stderr


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
    assert run_result.stdout.strip()
    assert (data_dir / "uni_rag.sqlite").is_file()
    assert _sqlite_count(data_dir, "courses") == 1
    assert _sqlite_count(data_dir, "files") == 2
    assert _sqlite_count(data_dir, "files", "index_status = 'pending'") == 1
    assert _sqlite_count(data_dir, "files", "index_status = 'metadata_only'") == 1
    inventory_events = _run_log_events(data_dir, "inventory-run")
    assert [event["event"] for event in inventory_events] == [
        "inventory_started",
        "inventory_completed",
    ]
    assert inventory_events[-1]["run_id"] == 1
    assert inventory_events[-1]["count"] == 2

    assert summary_result.returncode == 0, summary_result.stderr
    assert summary_result.stdout.strip()


def test_inventory_cli_validates_paths_before_creating_run_log(tmp_path: Path) -> None:
    courses_root = tmp_path / "Courses"
    data_dir = tmp_path / "data"
    courses_root.mkdir()
    env = _subprocess_env(
        {
            "UNI_RAG_COURSES_ROOT": str(courses_root),
            "UNI_RAG_DATA_DIR": str(data_dir),
            "UNI_RAG_SQLITE_PATH": str(data_dir / "uni_rag.sqlite"),
            "UNI_RAG_CHROMA_DIR": str(data_dir / "indexes" / "vector"),
            "UNI_RAG_RUNS_DIR": str(courses_root / "runs"),
            "UNI_RAG_USE_FAKE_LLM": "true",
            "UNI_RAG_USE_FAKE_EMBEDDINGS": "true",
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "uni_rag_agent", "inventory", "run"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "must not point inside the Courses root" in result.stderr
    assert not (courses_root / "runs").exists()


def test_extract_run_cli_writes_chunks_and_status(tmp_path: Path) -> None:
    courses_root = tmp_path / "Courses"
    data_dir = tmp_path / "data"
    course_dir = courses_root / "Information Retrieval"
    course_dir.mkdir(parents=True)
    (course_dir / "syllabus.txt").write_text("BM25 keyword search", encoding="utf-8")
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

    inventory_result = subprocess.run(
        [sys.executable, "-m", "uni_rag_agent", "inventory", "run"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    extract_result = subprocess.run(
        [sys.executable, "-m", "uni_rag_agent", "extract", "run"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    status_result = subprocess.run(
        [sys.executable, "-m", "uni_rag_agent", "extract", "status"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert inventory_result.returncode == 0, inventory_result.stderr
    assert extract_result.returncode == 0, extract_result.stderr
    assert extract_result.stdout.strip()
    assert _sqlite_count(data_dir, "files", "index_status = 'indexed'") == 1
    assert _sqlite_count(data_dir, "chunks") == 1
    extraction_events = _run_log_events(data_dir, "extract-run")
    assert [event["event"] for event in extraction_events] == [
        "extraction_started",
        "extraction_completed",
    ]
    assert extraction_events[-1]["count"] == 1

    assert status_result.returncode == 0, status_result.stderr
    assert status_result.stdout.strip()


def test_env_example_exists_and_env_is_ignored() -> None:
    assert (REPO_ROOT / ".env.example").is_file()

    ignored_patterns = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert ".env" in ignored_patterns
    assert ".env.example" not in ignored_patterns


def _sqlite_count(data_dir: Path, table: str, where: str | None = None) -> int:
    query = f"SELECT COUNT(*) FROM {table}"
    if where is not None:
        query += f" WHERE {where}"
    with sqlite3.connect(data_dir / "uni_rag.sqlite") as connection:
        row = connection.execute(query).fetchone()
    return int(row[0])


def _run_log_events(data_dir: Path, slug: str) -> list[dict[str, object]]:
    log_files = sorted((data_dir / "runs").glob(f"*-{slug}.jsonl"))
    assert log_files
    return [
        json.loads(line)
        for line in log_files[-1].read_text(encoding="utf-8").splitlines()
    ]


def _subprocess_env(overrides: dict[str, str]) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(UNI_RAG_ENV_PREFIX)
    }
    env.update(overrides)
    return env
