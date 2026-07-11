from __future__ import annotations

import os
import sqlite3
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
    connect_sqlite_read_only,
    ensure_data_dirs,
    initialize_schema,
)
from uni_rag_agent.storage import core as storage_core
from tests.sqlite_helpers import (
    insert_embedding_row,
    insert_minimal_chunk,
    insert_search_result,
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


def test_check_storage_reports_missing_database_without_creating_it(
    tmp_path: Path,
) -> None:
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


def test_read_only_sqlite_connection_rejects_writes(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    ensure_data_dirs(config)
    with connect_sqlite(config) as connection:
        initialize_schema(connection)

    with connect_sqlite_read_only(config) as connection:
        row = connection.execute("SELECT COUNT(*) FROM courses").fetchone()
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute(
                """
                INSERT INTO courses (name, path, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                ("Read Only", "Read Only", "2026-06-30", "2026-06-30"),
            )

    assert row[0] == 0


def test_search_results_chunk_reference_nulls_when_chunk_is_deleted(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    ensure_data_dirs(config)
    with connect_sqlite(config) as connection:
        initialize_schema(connection)
        rows = connection.execute("PRAGMA foreign_key_list(search_results)").fetchall()

    chunk_reference = next(
        row for row in rows if row["table"] == "chunks" and row["from"] == "chunk_id"
    )
    assert chunk_reference["on_delete"] == "SET NULL"


def test_initialize_schema_migrates_existing_search_results_chunk_reference(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    ensure_data_dirs(config)
    with connect_sqlite(config) as connection:
        initialize_schema(connection)
        _replace_search_results_with_legacy_chunk_fk(connection)
        stored_chunk = insert_minimal_chunk(connection, config)
        search_result = insert_search_result(
            connection,
            chunk_id=stored_chunk.chunk_id,
            file_id=stored_chunk.file_id,
        )
        connection.commit()

        initialize_schema(connection)
        rows = connection.execute("PRAGMA foreign_key_list(search_results)").fetchall()
        chunk_reference = next(
            row
            for row in rows
            if row["table"] == "chunks" and row["from"] == "chunk_id"
        )
        connection.execute("DELETE FROM chunks WHERE id = ?", (stored_chunk.chunk_id,))
        historical_result = connection.execute(
            "SELECT chunk_id, file_id FROM search_results WHERE id = ?",
            (search_result.search_result_id,),
        ).fetchone()

    assert chunk_reference["on_delete"] == "SET NULL"
    assert historical_result["chunk_id"] is None
    assert historical_result["file_id"] == stored_chunk.file_id


def test_embeddings_chunk_reference_cascades_on_fresh_schema(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    ensure_data_dirs(config)
    with connect_sqlite(config) as connection:
        initialize_schema(connection)
        rows = connection.execute("PRAGMA foreign_key_list(embeddings)").fetchall()

    chunk_reference = next(
        row for row in rows if row["table"] == "chunks" and row["from"] == "chunk_id"
    )
    assert chunk_reference["on_delete"] == "CASCADE"


def test_initialize_schema_migrates_embeddings_chunk_cascade(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    ensure_data_dirs(config)
    with connect_sqlite(config) as connection:
        initialize_schema(connection)
        _replace_embeddings_with_legacy_chunk_fk(connection)
        stored_chunk = insert_minimal_chunk(connection, config)
        embedding_id = insert_embedding_row(connection, chunk_id=stored_chunk.chunk_id)
        connection.commit()

        initialize_schema(connection)
        rows = connection.execute("PRAGMA foreign_key_list(embeddings)").fetchall()
        chunk_reference = next(
            row
            for row in rows
            if row["table"] == "chunks" and row["from"] == "chunk_id"
        )
        unique_indexes = {
            tuple(
                detail["name"]
                for detail in connection.execute(
                    f"PRAGMA index_info('{index['name']}')"
                ).fetchall()
            )
            for index in connection.execute("PRAGMA index_list(embeddings)").fetchall()
            if index["unique"]
        }
        chunk_id_index_present = any(
            index["name"] == "idx_embeddings_chunk_id"
            for index in connection.execute("PRAGMA index_list(embeddings)").fetchall()
        )
        preserved_before = connection.execute(
            "SELECT id, chunk_id FROM embeddings WHERE id = ?",
            (embedding_id,),
        ).fetchone()
        connection.execute("DELETE FROM chunks WHERE id = ?", (stored_chunk.chunk_id,))
        remaining = connection.execute(
            "SELECT COUNT(*) FROM embeddings WHERE id = ?",
            (embedding_id,),
        ).fetchone()[0]

    assert chunk_reference["on_delete"] == "CASCADE"
    assert preserved_before["chunk_id"] == stored_chunk.chunk_id
    assert chunk_id_index_present
    assert ("vector_backend", "vector_collection", "vector_id") in unique_indexes
    assert ("chunk_id", "vector_backend", "vector_collection") in unique_indexes
    assert remaining == 0


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


def _replace_embeddings_with_legacy_chunk_fk(
    connection: sqlite3.Connection,
) -> None:
    connection.execute("DROP TABLE embeddings")
    connection.execute(
        """
        CREATE TABLE embeddings (
            id INTEGER PRIMARY KEY,
            chunk_id INTEGER NOT NULL REFERENCES chunks(id),
            vector_backend TEXT NOT NULL,
            vector_collection TEXT NOT NULL,
            vector_id TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL,
            embedded_at TEXT NOT NULL,
            UNIQUE(vector_backend, vector_collection, vector_id),
            UNIQUE(chunk_id, embedding_model)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id)"
    )


def _replace_search_results_with_legacy_chunk_fk(
    connection: sqlite3.Connection,
) -> None:
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


def _subprocess_env(overrides: dict[str, str]) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(UNI_RAG_ENV_PREFIX)
    }
    env.update(overrides)
    return env
