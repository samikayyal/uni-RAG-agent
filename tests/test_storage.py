from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from uni_rag_agent.config import Config, load_config
from uni_rag_agent.storage import (
    REQUIRED_TABLES,
    StorageError,
    check_storage,
    connect_sqlite,
    ensure_data_dirs,
    initialize_schema,
)
from uni_rag_agent.storage import core as storage_core

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


def test_search_results_chunk_reference_nulls_when_chunk_is_deleted(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    ensure_data_dirs(config)
    with connect_sqlite(config) as connection:
        initialize_schema(connection)
        rows = connection.execute("PRAGMA foreign_key_list(search_results)").fetchall()

    chunk_reference = next(
        row
        for row in rows
        if row["table"] == "chunks" and row["from"] == "chunk_id"
    )
    assert chunk_reference["on_delete"] == "SET NULL"


def test_initialize_schema_migrates_existing_search_results_chunk_reference(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    ensure_data_dirs(config)
    with connect_sqlite(config) as connection:
        initialize_schema(connection)
        connection.execute("DROP TABLE search_results")
        connection.execute(
            """
            CREATE TABLE search_results (
                id INTEGER PRIMARY KEY,
                search_run_id INTEGER NOT NULL REFERENCES search_runs(id),
                chunk_id INTEGER REFERENCES chunks(id),
                file_id INTEGER REFERENCES files(id),
                retrieval_method TEXT NOT NULL,
                rank INTEGER NOT NULL,
                score REAL,
                selected_for_evidence INTEGER NOT NULL DEFAULT 0,
                result_json TEXT
            )
            """
        )
        file_id = connection.execute(
            """
            INSERT INTO files (
                path,
                relative_path,
                filename,
                extension,
                size_bytes,
                category,
                index_status,
                discovered_at,
                last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(config.courses_root / "notes.md"),
                "notes.md",
                "notes.md",
                ".md",
                1,
                "document",
                "indexed",
                "2026-06-27T00:00:00+00:00",
                "2026-06-27T00:00:00+00:00",
            ),
        ).lastrowid
        extraction_run_id = connection.execute(
            """
            INSERT INTO extraction_runs (started_at, status, config_json)
            VALUES (?, ?, ?)
            """,
            ("2026-06-27T00:00:00+00:00", "completed", "{}"),
        ).lastrowid
        extracted_document_id = connection.execute(
            """
            INSERT INTO extracted_documents (
                file_id,
                extraction_run_id,
                extractor_name,
                status,
                text_length,
                chunk_count,
                extracted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                extraction_run_id,
                "markdown-text",
                "indexed",
                4,
                1,
                "2026-06-27T00:00:00+00:00",
            ),
        ).lastrowid
        chunk_id = connection.execute(
            """
            INSERT INTO chunks (
                file_id,
                extracted_document_id,
                chunk_uid,
                source_type,
                chunk_index,
                text,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                extracted_document_id,
                "file-1-chunk-0",
                "document",
                0,
                "BM25",
                "2026-06-27T00:00:00+00:00",
            ),
        ).lastrowid
        search_run_id = connection.execute(
            """
            INSERT INTO search_runs (
                query,
                router_output_json,
                searched_courses_json,
                searched_indexes_json,
                keyword_terms_json,
                semantic_queries_json,
                started_at,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "bm25",
                "{}",
                "[]",
                "[]",
                '["bm25"]',
                "[]",
                "2026-06-27T00:00:00+00:00",
                "completed",
            ),
        ).lastrowid
        search_result_id = connection.execute(
            """
            INSERT INTO search_results (
                search_run_id,
                chunk_id,
                file_id,
                retrieval_method,
                rank
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (search_run_id, chunk_id, file_id, "keyword", 1),
        ).lastrowid
        connection.commit()

        initialize_schema(connection)
        rows = connection.execute("PRAGMA foreign_key_list(search_results)").fetchall()
        chunk_reference = next(
            row
            for row in rows
            if row["table"] == "chunks" and row["from"] == "chunk_id"
        )
        connection.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
        historical_result = connection.execute(
            "SELECT chunk_id, file_id FROM search_results WHERE id = ?",
            (search_result_id,),
        ).fetchone()

    assert chunk_reference["on_delete"] == "SET NULL"
    assert historical_result["chunk_id"] is None
    assert historical_result["file_id"] == file_id


def test_initialize_schema_reports_clear_fts5_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path)
    ensure_data_dirs(config)
    monkeypatch.setattr(
        storage_core,
        "check_fts5_available",
        lambda _connection: (False, "no such module: fts5"),
    )

    with connect_sqlite(config) as connection:
        with pytest.raises(StorageError, match="SQLite FTS5 is not available"):
            initialize_schema(connection)


def test_check_storage_reports_invalid_sqlite_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    ensure_data_dirs(config)
    config.sqlite_path.write_text("not a sqlite database", encoding="utf-8")

    result = check_storage(config)

    assert not result.ok
    assert result.sqlite_exists
    assert any(
        "SQLite database cannot be inspected" in diagnostic
        for diagnostic in result.diagnostics
    )


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
