"""SQLite persistence helpers for extraction runs and chunks."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from uni_rag_agent.config import Config

from .constants import (
    DEFAULT_MAX_CHUNK_TOKENS,
    ERROR_CHAR_LIMIT,
    TEXT_EXTRACTION_CATEGORIES,
)
from .models import (
    ExtractedDocument,
    ExtractionError,
    ExtractionFailureSummary,
    PendingFileRecord,
)


def _record_failed_extraction(
    connection: sqlite3.Connection,
    *,
    run_id: int,
    file_record: PendingFileRecord,
    extractor_name: str,
    extractor_version: str | None,
    metadata_json: str | None,
    error: str,
) -> None:
    with connection:
        _persist_failed_extraction(
            connection,
            run_id=run_id,
            file_record=file_record,
            extractor_name=extractor_name,
            extractor_version=extractor_version,
            metadata_json=metadata_json,
            error=error,
            extracted_at=_utc_now(),
        )


def _persist_successful_extraction(
    connection: sqlite3.Connection,
    *,
    run_id: int,
    extracted: ExtractedDocument,
    extracted_at: str,
) -> None:
    _delete_existing_chunks(connection, extracted.file_id)
    extracted_document_id = _upsert_extracted_document(
        connection,
        run_id=run_id,
        file_id=extracted.file_id,
        extractor_name=extracted.extractor_name,
        extractor_version=extracted.extractor_version,
        status=extracted.status,
        text_length=extracted.text_length,
        chunk_count=extracted.chunk_count,
        metadata_json=extracted.metadata_json,
        error=None,
        extracted_at=extracted_at,
    )
    for chunk in extracted.chunks:
        connection.execute(
            """
            INSERT INTO chunks (
                file_id,
                extracted_document_id,
                chunk_uid,
                source_type,
                chunk_index,
                title,
                text,
                token_count,
                location_type,
                location_value,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.file_id,
                extracted_document_id,
                chunk.chunk_uid,
                chunk.source_type,
                chunk.chunk_index,
                chunk.title,
                chunk.text,
                chunk.token_count,
                chunk.location_type,
                chunk.location_value,
                chunk.metadata_json,
                extracted_at,
            ),
        )
    connection.execute(
        """
        UPDATE files
        SET index_status = 'indexed',
            reason_not_indexed = NULL
        WHERE id = ?
        """,
        (extracted.file_id,),
    )


def _persist_failed_extraction(
    connection: sqlite3.Connection,
    *,
    run_id: int,
    file_record: PendingFileRecord,
    extractor_name: str,
    extractor_version: str | None,
    metadata_json: str | None,
    error: str,
    extracted_at: str,
) -> None:
    _delete_existing_chunks(connection, file_record.id)
    _upsert_extracted_document(
        connection,
        run_id=run_id,
        file_id=file_record.id,
        extractor_name=extractor_name,
        extractor_version=extractor_version,
        status="failed",
        text_length=0,
        chunk_count=0,
        metadata_json=metadata_json
        or _json_dumps({"extension": file_record.extension}),
        error=error,
        extracted_at=extracted_at,
    )
    connection.execute(
        """
        UPDATE files
        SET index_status = 'failed',
            reason_not_indexed = ?
        WHERE id = ?
        """,
        (_truncate(error, ERROR_CHAR_LIMIT), file_record.id),
    )


def _delete_existing_chunks(connection: sqlite3.Connection, file_id: int) -> None:
    connection.execute(
        """
        DELETE FROM embeddings
        WHERE chunk_id IN (
            SELECT id
            FROM chunks
            WHERE file_id = ?
        )
        """,
        (file_id,),
    )
    connection.execute(
        """
        DELETE FROM chunk_fts
        WHERE chunk_id IN (
            SELECT id
            FROM chunks
            WHERE file_id = ?
        )
        """,
        (file_id,),
    )
    connection.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))


def _upsert_extracted_document(
    connection: sqlite3.Connection,
    *,
    run_id: int,
    file_id: int,
    extractor_name: str,
    extractor_version: str | None,
    status: str,
    text_length: int,
    chunk_count: int,
    metadata_json: str,
    error: str | None,
    extracted_at: str,
) -> int:
    connection.execute(
        """
        INSERT INTO extracted_documents (
            file_id,
            extraction_run_id,
            extractor_name,
            extractor_version,
            status,
            text_length,
            chunk_count,
            metadata_json,
            error,
            extracted_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_id, extractor_name) DO UPDATE SET
            extraction_run_id = excluded.extraction_run_id,
            extractor_version = excluded.extractor_version,
            status = excluded.status,
            text_length = excluded.text_length,
            chunk_count = excluded.chunk_count,
            metadata_json = excluded.metadata_json,
            error = excluded.error,
            extracted_at = excluded.extracted_at
        """,
        (
            file_id,
            run_id,
            extractor_name,
            extractor_version,
            status,
            text_length,
            chunk_count,
            metadata_json,
            error,
            extracted_at,
        ),
    )
    row = connection.execute(
        """
        SELECT id
        FROM extracted_documents
        WHERE file_id = ? AND extractor_name = ?
        """,
        (file_id, extractor_name),
    ).fetchone()
    if row is None:
        raise ExtractionError(f"Failed to upsert extracted document for file {file_id}")
    return int(row["id"])


def _load_pending_files(
    connection: sqlite3.Connection,
    category: str | None,
) -> tuple[PendingFileRecord, ...]:
    categories = (
        [category] if category is not None else sorted(TEXT_EXTRACTION_CATEGORIES)
    )
    placeholders = ",".join("?" for _ in categories)
    rows = connection.execute(
        f"""
        SELECT id, path, relative_path, filename, extension, category, content_hash
        FROM files
        WHERE index_status = 'pending'
          AND category IN ({placeholders})
        ORDER BY relative_path COLLATE NOCASE
        """,
        categories,
    ).fetchall()
    return tuple(
        PendingFileRecord(
            id=int(row["id"]),
            path=Path(str(row["path"])),
            relative_path=str(row["relative_path"]),
            filename=str(row["filename"]),
            extension=str(row["extension"]),
            category=str(row["category"]),
            content_hash=row["content_hash"],
        )
        for row in rows
    )


def _start_extraction_run(
    connection: sqlite3.Connection,
    config: Config,
    category: str | None,
    started_at: str,
) -> int:
    config_json = _json_dumps(
        {
            "run_type": "extraction",
            "category": category,
            "courses_root": str(config.courses_root),
            "data_dir": str(config.data_dir),
            "sqlite_path": str(config.sqlite_path),
            "ocr_enabled": config.ocr_enabled,
            "max_chunk_tokens": DEFAULT_MAX_CHUNK_TOKENS,
            "handled_categories": sorted(TEXT_EXTRACTION_CATEGORIES),
        }
    )
    cursor = connection.execute(
        """
        INSERT INTO extraction_runs (
            started_at,
            status,
            config_json,
            files_seen,
            files_indexed,
            files_metadata_only,
            files_failed
        )
        VALUES (?, 'running', ?, 0, 0, 0, 0)
        """,
        (started_at, config_json),
    )
    return int(cursor.lastrowid)


def _finish_extraction_run(
    connection: sqlite3.Connection,
    *,
    run_id: int,
    finished_at: str,
    status: str,
    files_seen: int,
    files_indexed: int,
    files_failed: int,
    error: str | None,
) -> None:
    connection.execute(
        """
        UPDATE extraction_runs
        SET finished_at = ?,
            status = ?,
            files_seen = ?,
            files_indexed = ?,
            files_metadata_only = 0,
            files_failed = ?,
            error = ?
        WHERE id = ?
        """,
        (
            finished_at,
            status,
            files_seen,
            files_indexed,
            files_failed,
            error,
            run_id,
        ),
    )


def _latest_extraction_run(
    connection: sqlite3.Connection,
) -> tuple[int | None, str | None]:
    rows = connection.execute("""
        SELECT id, started_at, config_json
        FROM extraction_runs
        ORDER BY id DESC
        """).fetchall()
    for row in rows:
        try:
            payload = json.loads(row["config_json"])
        except json.JSONDecodeError:
            continue
        if payload.get("run_type") == "extraction":
            return int(row["id"]), str(row["started_at"])
    return None, None


def _count_pending_by_category(connection: sqlite3.Connection) -> dict[str, int]:
    placeholders = ",".join("?" for _ in TEXT_EXTRACTION_CATEGORIES)
    rows = connection.execute(
        f"""
        SELECT category, COUNT(*) AS count
        FROM files
        WHERE index_status = 'pending'
          AND category IN ({placeholders})
        GROUP BY category
        ORDER BY category
        """,
        sorted(TEXT_EXTRACTION_CATEGORIES),
    ).fetchall()
    return {str(row["category"]): int(row["count"]) for row in rows}


def _count_chunks_by_source_type(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute("""
        SELECT source_type, COUNT(*) AS count
        FROM chunks
        GROUP BY source_type
        ORDER BY source_type
        """).fetchall()
    return {str(row["source_type"]): int(row["count"]) for row in rows}


def _count_files_by_status(connection: sqlite3.Connection, status: str) -> int:
    placeholders = ",".join("?" for _ in TEXT_EXTRACTION_CATEGORIES)
    row = connection.execute(
        f"""
        SELECT COUNT(*) AS count
        FROM files
        WHERE index_status = ?
          AND category IN ({placeholders})
        """,
        (status, *sorted(TEXT_EXTRACTION_CATEGORIES)),
    ).fetchone()
    return int(row["count"] if row else 0)


def _count_table_rows(connection: sqlite3.Connection, table: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return int(row["count"] if row else 0)


def _load_recent_failures(
    connection: sqlite3.Connection,
) -> tuple[ExtractionFailureSummary, ...]:
    rows = connection.execute("""
        SELECT files.id AS file_id, files.path AS path, extracted_documents.error AS error
        FROM extracted_documents
        JOIN files ON files.id = extracted_documents.file_id
        WHERE extracted_documents.status = 'failed'
        ORDER BY extracted_documents.extracted_at DESC
        LIMIT 10
        """).fetchall()
    return tuple(
        ExtractionFailureSummary(
            file_id=int(row["file_id"]),
            path=str(row["path"]),
            error=str(row["error"]),
        )
        for row in rows
    )


def _json_dumps(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
