"""SQLite persistence for data summary extraction."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from uni_rag_agent.config import Config

from .._textutils import _json_dumps, _utc_now
from ..constants import DEFAULT_MAX_CHUNK_TOKENS
from ..models import ChunkRecord, DataSummary, ExtractedDocument, PendingFileRecord
from ..persistence import (
    _delete_existing_chunks,
    _finish_extraction_run,
    _persist_failed_extraction,
    _upsert_extracted_document,
)
from .builders import (
    DATA_SCHEMA_CATEGORY,
    DATA_SCHEMA_EXTENSIONS,
    DATA_SCHEMA_EXTRACTOR_NAME,
    DATA_SCHEMA_EXTRACTOR_VERSION,
    SAMPLE_ROW_LIMIT,
)

__all__ = [
    "_file_filter_diagnostics",
    "_finish_extraction_run",
    "_load_pending_data_files",
    "_persist_failed_data_summary",
    "_persist_successful_data_summary",
    "_start_data_summary_run",
]


def _persist_successful_data_summary(
    connection: sqlite3.Connection,
    *,
    run_id: int,
    summary: DataSummary,
    extracted: ExtractedDocument,
    chunks: tuple[ChunkRecord, ...],
    created_at: str,
) -> None:
    _delete_existing_chunks(connection, summary.file_id)
    _delete_existing_data_summary(connection, summary.file_id)
    extracted_document_id = _upsert_extracted_document(
        connection,
        run_id=run_id,
        file_id=summary.file_id,
        extractor_name=extracted.extractor_name,
        extractor_version=extracted.extractor_version,
        status=extracted.status,
        text_length=extracted.text_length,
        chunk_count=extracted.chunk_count,
        metadata_json=extracted.metadata_json,
        error=None,
        extracted_at=created_at,
    )
    connection.execute(
        """
        INSERT INTO data_summaries (
            file_id,
            format,
            row_count,
            column_count,
            table_count,
            sheet_count,
            schema_json,
            sample_json,
            summary_text,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_id) DO UPDATE SET
            format = excluded.format,
            row_count = excluded.row_count,
            column_count = excluded.column_count,
            table_count = excluded.table_count,
            sheet_count = excluded.sheet_count,
            schema_json = excluded.schema_json,
            sample_json = excluded.sample_json,
            summary_text = excluded.summary_text,
            created_at = excluded.created_at
        """,
        (
            summary.file_id,
            summary.format,
            summary.row_count,
            summary.column_count,
            summary.table_count,
            summary.sheet_count,
            summary.schema_json,
            summary.sample_json,
            summary.summary_text,
            created_at,
        ),
    )
    for chunk in chunks:
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
                created_at,
            ),
        )
    connection.execute(
        """
        UPDATE files
        SET index_status = 'indexed',
            reason_not_indexed = NULL
        WHERE id = ?
        """,
        (summary.file_id,),
    )


def _persist_failed_data_summary(
    connection: sqlite3.Connection,
    *,
    run_id: int,
    file_record: PendingFileRecord,
    error: str,
    metadata_json: str | None,
) -> None:
    _delete_existing_chunks(connection, file_record.id)
    _delete_existing_data_summary(connection, file_record.id)
    _persist_failed_extraction(
        connection,
        run_id=run_id,
        file_record=file_record,
        extractor_name=DATA_SCHEMA_EXTRACTOR_NAME,
        extractor_version=DATA_SCHEMA_EXTRACTOR_VERSION,
        metadata_json=metadata_json,
        error=error,
        extracted_at=_utc_now(),
    )


def _delete_existing_data_summary(
    connection: sqlite3.Connection,
    file_id: int,
) -> None:
    connection.execute("DELETE FROM data_summaries WHERE file_id = ?", (file_id,))


def _load_pending_data_files(
    connection: sqlite3.Connection,
    file_id: int | None,
) -> tuple[PendingFileRecord, ...]:
    parameters: list[object] = [DATA_SCHEMA_CATEGORY]
    file_filter = ""
    if file_id is not None:
        file_filter = "AND id = ?"
        parameters.append(file_id)
    rows = connection.execute(
        f"""
        SELECT id, path, relative_path, filename, extension, category, content_hash
        FROM files
        WHERE index_status = 'pending'
          AND category = ?
          {file_filter}
        ORDER BY relative_path COLLATE NOCASE
        """,
        parameters,
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


def _file_filter_diagnostics(
    connection: sqlite3.Connection,
    file_id: int | None,
    pending_files: tuple[PendingFileRecord, ...],
) -> list[str]:
    if file_id is None or pending_files:
        return []
    row = connection.execute(
        """
        SELECT path, category, index_status
        FROM files
        WHERE id = ?
        """,
        (file_id,),
    ).fetchone()
    if row is None:
        return [f"file_id {file_id} was not found"]
    if row["category"] != DATA_SCHEMA_CATEGORY:
        return [
            f"file_id {file_id} is category {row['category']}, not {DATA_SCHEMA_CATEGORY}"
        ]
    return [f"file_id {file_id} is {row['index_status']}, not pending: {row['path']}"]


def _start_data_summary_run(
    connection: sqlite3.Connection,
    config: Config,
    file_id: int | None,
    started_at: str,
) -> int:
    config_json = _json_dumps(
        {
            "run_type": "data_summary",
            "category": DATA_SCHEMA_CATEGORY,
            "file_id": file_id,
            "courses_root": str(config.courses_root),
            "data_dir": str(config.data_dir),
            "sqlite_path": str(config.sqlite_path),
            "sample_row_limit": SAMPLE_ROW_LIMIT,
            "handled_extensions": sorted(DATA_SCHEMA_EXTENSIONS),
            "max_chunk_tokens": DEFAULT_MAX_CHUNK_TOKENS,
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
