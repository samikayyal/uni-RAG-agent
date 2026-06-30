"""Text extraction orchestration."""

from __future__ import annotations

from collections import Counter
from contextlib import closing

from uni_rag_agent.config import Config
from uni_rag_agent.storage import connect_sqlite, ensure_data_dirs, initialize_schema

from ._textutils import (
    _failure_from_exception,
    _format_exception,
    _json_dumps,
    _utc_now,
)
from .chunking import finalize_chunks
from .constants import (
    DEFAULT_MAX_CHUNK_TOKENS,
    LEGACY_EXTENSIONS,
    LEGACY_FORMAT_REASON,
    NO_TEXT_REASON,
    TEXT_EXTRACTION_CATEGORIES,
)
from .extractors import (
    extract_raw_chunks,
    extractor_name_for_extension,
    extractor_version_for_extension,
)
from .models import (
    ExtractedDocument,
    ExtractionError,
    ExtractionFailure,
    ExtractionFailureSummary,
    ExtractionRunResult,
    ExtractionStatus,
    PendingFileRecord,
)
from .persistence import (
    _count_chunks_by_source_type,
    _count_files_by_status,
    _count_pending_by_category,
    _count_table_rows,
    _finish_extraction_run,
    _latest_extraction_run,
    _load_pending_files,
    _load_recent_failures,
    _persist_successful_extraction,
    _record_failed_extraction,
    _start_extraction_run,
)


def extract_pending_files(
    config: Config,
    category: str | None = None,
) -> ExtractionRunResult:
    """Extract pending text-like files and persist chunks per file."""
    _validate_category(category)
    ensure_data_dirs(config)

    with closing(connect_sqlite(config)) as connection:
        initialize_schema(connection)
        started_at = _utc_now()
        run_id = _start_extraction_run(connection, config, category, started_at)
        connection.commit()

        pending_files = _load_pending_files(connection, category)
        files_indexed = 0
        files_failed = 0
        chunks_created = 0
        by_source_type: Counter[str] = Counter()
        failures: list[ExtractionFailureSummary] = []
        diagnostics: list[str] = []

        try:
            for file_record in pending_files:
                try:
                    extracted = extract_file(file_record, config)
                except ExtractionFailure as exc:
                    files_failed += 1
                    failure = _failure_from_exception(file_record, exc)
                    failures.append(failure)
                    _record_failed_extraction(
                        connection,
                        run_id=run_id,
                        file_record=file_record,
                        extractor_name=exc.extractor_name,
                        extractor_version=exc.extractor_version,
                        metadata_json=exc.metadata_json,
                        error=failure.error,
                    )
                    continue
                except Exception as exc:
                    files_failed += 1
                    error = _format_exception(exc)
                    failures.append(
                        ExtractionFailureSummary(
                            file_id=file_record.id,
                            path=str(file_record.path),
                            error=error,
                        )
                    )
                    _record_failed_extraction(
                        connection,
                        run_id=run_id,
                        file_record=file_record,
                        extractor_name=extractor_name_for_extension(
                            file_record.extension
                        ),
                        extractor_version=None,
                        metadata_json=_json_dumps({"extension": file_record.extension}),
                        error=error,
                    )
                    continue

                files_indexed += 1
                chunks_created += extracted.chunk_count
                by_source_type.update(chunk.source_type for chunk in extracted.chunks)
                with connection:
                    _persist_successful_extraction(
                        connection,
                        run_id=run_id,
                        extracted=extracted,
                        extracted_at=_utc_now(),
                    )

            finished_at = _utc_now()
            status = "completed"
            with connection:
                _finish_extraction_run(
                    connection,
                    run_id=run_id,
                    finished_at=finished_at,
                    status=status,
                    files_seen=len(pending_files),
                    files_indexed=files_indexed,
                    files_failed=files_failed,
                    error=None,
                )
        except Exception as exc:
            with connection:
                _finish_extraction_run(
                    connection,
                    run_id=run_id,
                    finished_at=_utc_now(),
                    status="failed",
                    files_seen=len(pending_files),
                    files_indexed=files_indexed,
                    files_failed=files_failed,
                    error=str(exc),
                )
            raise

    return ExtractionRunResult(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        category=category,
        files_seen=len(pending_files),
        files_indexed=files_indexed,
        files_failed=files_failed,
        chunks_created=chunks_created,
        by_source_type=dict(sorted(by_source_type.items())),
        failures=tuple(failures),
        diagnostics=tuple(diagnostics),
    )


def load_extraction_status(config: Config) -> ExtractionStatus:
    """Read extraction/chunk coverage without traversing Courses."""
    if not config.sqlite_path.is_file():
        raise ExtractionError(f"SQLite database does not exist: {config.sqlite_path}")

    with closing(connect_sqlite(config)) as connection:
        latest_run_id, latest_started_at = _latest_extraction_run(connection)
        pending_by_category = _count_pending_by_category(connection)
        chunks_by_source_type = _count_chunks_by_source_type(connection)
        pending_text_files = sum(pending_by_category.values())
        indexed_text_files = _count_files_by_status(connection, "indexed")
        failed_text_files = _count_files_by_status(connection, "failed")
        extracted_documents = _count_table_rows(connection, "extracted_documents")
        chunks_total = _count_table_rows(connection, "chunks")
        recent_failures = _load_recent_failures(connection)

    return ExtractionStatus(
        latest_extraction_run_id=latest_run_id,
        latest_extraction_started_at=latest_started_at,
        pending_text_files=pending_text_files,
        indexed_text_files=indexed_text_files,
        failed_text_files=failed_text_files,
        extracted_documents=extracted_documents,
        chunks_total=chunks_total,
        pending_by_category=pending_by_category,
        chunks_by_source_type=chunks_by_source_type,
        recent_failures=recent_failures,
    )


def extract_file(file_record: PendingFileRecord, config: Config) -> ExtractedDocument:
    """Extract one pending file into final chunk records."""
    extension = file_record.extension
    extractor_name = extractor_name_for_extension(extension)
    extractor_version = extractor_version_for_extension(extension)

    if extension in LEGACY_EXTENSIONS:
        raise ExtractionFailure(
            LEGACY_FORMAT_REASON,
            extractor_name=extractor_name,
            extractor_version=extractor_version,
            metadata_json=_json_dumps({"extension": extension}),
        )

    raw_chunks = extract_raw_chunks(file_record, config)
    chunks = finalize_chunks(
        file_record=file_record,
        raw_chunks=raw_chunks,
        max_tokens=DEFAULT_MAX_CHUNK_TOKENS,
    )

    if not chunks:
        raise ExtractionFailure(
            NO_TEXT_REASON,
            extractor_name=extractor_name,
            extractor_version=extractor_version,
            metadata_json=_json_dumps({"extension": extension}),
        )

    metadata_json = _json_dumps(
        {
            "extension": extension,
            "relative_path": file_record.relative_path,
            "content_hash": file_record.content_hash,
            "max_chunk_tokens": DEFAULT_MAX_CHUNK_TOKENS,
        }
    )
    return ExtractedDocument(
        file_id=file_record.id,
        extractor_name=extractor_name,
        extractor_version=extractor_version,
        status="indexed",
        text_length=sum(len(chunk.text) for chunk in chunks),
        chunk_count=len(chunks),
        metadata_json=metadata_json,
        error=None,
        chunks=tuple(chunks),
    )


def _validate_category(category: str | None) -> None:
    if category is None:
        return
    if category not in TEXT_EXTRACTION_CATEGORIES:
        allowed = ", ".join(sorted(TEXT_EXTRACTION_CATEGORIES))
        raise ExtractionError(
            f"Text extraction category must be one of: {allowed}. "
            "Data schema summaries are handled by Feature 05."
        )
