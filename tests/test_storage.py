from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from uni_rag_agent.config import Config, load_config
from uni_rag_agent.storage import (
    REQUIRED_TABLES,
    check_storage,
    connect_sqlite,
    ensure_data_dirs,
    initialize_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
UNI_RAG_ENV_PREFIX = "UNI_RAG_"


def make_config(tmp_path: Path) -> Config:
    (tmp_path / "Courses").mkdir()
    return load_config(repo_root=tmp_path, env_file=tmp_path / "missing.env")


def test_storage_init_creates_expected_directories_and_schema(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    ensure_data_dirs(config)
    with connect_sqlite(config) as connection:
        initialize_schema(connection)

    assert config.data_dir.is_dir()
    assert config.extracted_dir.is_dir()
    assert config.chroma_dir.is_dir()
    assert config.runs_dir.is_dir()
    assert config.sqlite_path.is_file()

    result = check_storage(config)
    assert result.ok
    assert result.missing_tables == ()
    assert result.required_tables_present == REQUIRED_TABLES


def test_storage_initialization_is_idempotent(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    for _ in range(2):
        ensure_data_dirs(config)
        with connect_sqlite(config) as connection:
            initialize_schema(connection)

    result = check_storage(config)
    assert result.ok
    assert result.required_tables_present == REQUIRED_TABLES


def test_check_storage_reports_missing_database_without_creating_it(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    result = check_storage(config)

    assert not result.ok
    assert not config.sqlite_path.exists()
    assert "SQLite database does not exist" in "\n".join(result.diagnostics)
    assert set(result.missing_tables) == set(REQUIRED_TABLES)


def test_chunk_fts_table_uses_fts5(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    ensure_data_dirs(config)
    with connect_sqlite(config) as connection:
        initialize_schema(connection)
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'chunk_fts'"
        ).fetchone()

    assert row is not None
    assert "USING fts5" in row[0]
    assert "tokenize='unicode61'" in row[0]


def test_storage_init_cli_uses_temp_config_and_prints_health(tmp_path: Path) -> None:
    courses_root = tmp_path / "Courses"
    data_dir = tmp_path / "data"
    courses_root.mkdir()
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

    result = subprocess.run(
        [sys.executable, "-m", "uni_rag_agent", "storage", "init"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Storage initialized" in result.stdout
    assert "fts5_available: True" in result.stdout
    assert "missing_tables: none" in result.stdout
    assert (data_dir / "uni_rag.sqlite").is_file()
    assert (data_dir / "indexes" / "vector").is_dir()


def test_storage_check_cli_fails_clearly_before_init(tmp_path: Path) -> None:
    courses_root = tmp_path / "Courses"
    courses_root.mkdir()
    env = _subprocess_env(
        {
            "UNI_RAG_COURSES_ROOT": str(courses_root),
            "UNI_RAG_DATA_DIR": str(tmp_path / "data"),
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "uni_rag_agent", "storage", "check"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 3
    assert "Storage check failed" in result.stdout
    assert "SQLite database does not exist" in result.stdout


def _subprocess_env(overrides: dict[str, str]) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(UNI_RAG_ENV_PREFIX)
    }
    env.update(overrides)
    return env
