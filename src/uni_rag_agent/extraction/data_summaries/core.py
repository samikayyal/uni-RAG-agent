"""Run orchestration for data schema summary extraction."""

from __future__ import annotations

from collections import Counter
from contextlib import closing

from uni_rag_agent.config import Config
from uni_rag_agent.storage import connect_sqlite, ensure_data_dirs, initialize_schema

from ..chunking import finalize_chunks
from ..constants import DEFAULT_MAX_CHUNK_TOKENS
from ..models import (
    ChunkRecord,
    DataSummary,
    DataSummaryRunResult,
    ExtractedDocument,
    ExtractionFailure,
    ExtractionFailureSummary,
    PendingFileRecord,
    RawChunk,
)
from .._textutils import (
    _failure_from_exception,
    _format_exception,
    _json_dumps,
    _utc_now,
)
from .builders import (
    DATA_SCHEMA_EXTRACTOR_NAME,
    DATA_SCHEMA_EXTRACTOR_VERSION,
    DATA_SCHEMA_SOURCE_TYPE,
)
from .formats import (
    summarize_csv,
    summarize_json,
    summarize_jsonl,
    summarize_sqlite,
    summarize_xlsx,
)
from .persistence import (
    _file_filter_diagnostics,
    _finish_extraction_run,
    _load_pending_data_files,
    _persist_failed_data_summary,
    _persist_successful_data_summary,
    _start_data_summary_run,
)


def summarize_data_files(
    config: Config,
    file_id: int | None = None,
) -> DataSummaryRunResult:
    """Summarize pending data-schema files and persist retrieval chunks."""
    ensure_data_dirs(config)

    with closing(connect_sqlite(config)) as connection:
        initialize_schema(connection)
        started_at = _utc_now()
        run_id = _start_data_summary_run(connection, config, file_id, started_at)
        connection.commit()

        pending_files = _load_pending_data_files(connection, file_id)
        diagnostics = _file_filter_diagnostics(connection, file_id, pending_files)
        files_indexed = 0
        files_failed = 0
        chunks_created = 0
        by_format: Counter[str] = Counter()
        failures: list[ExtractionFailureSummary] = []

        try:
            for file_record in pending_files:
                try:
                    summary = summarize_data_file(file_record)
                    chunks = data_summary_to_chunks(file_record, summary)
                    extracted = _extracted_document_from_summary(
                        file_record=file_record,
                        summary=summary,
                        chunks=chunks,
                    )
                except ExtractionFailure as exc:
                    files_failed += 1
                    failure = _failure_from_exception(file_record, exc)
                    failures.append(failure)
                    with connection:
                        _persist_failed_data_summary(
                            connection,
                            run_id=run_id,
                            file_record=file_record,
                            error=failure.error,
                            metadata_json=exc.metadata_json,
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
                    with connection:
                        _persist_failed_data_summary(
                            connection,
                            run_id=run_id,
                            file_record=file_record,
                            error=error,
                            metadata_json=_json_dumps(
                                {"extension": file_record.extension}
                            ),
                        )
                    continue

                files_indexed += 1
                chunks_created += len(chunks)
                by_format.update([summary.format])
                with connection:
                    _persist_successful_data_summary(
                        connection,
                        run_id=run_id,
                        summary=summary,
                        extracted=extracted,
                        chunks=chunks,
                        created_at=_utc_now(),
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

    return DataSummaryRunResult(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        file_id=file_id,
        files_seen=len(pending_files),
        files_indexed=files_indexed,
        files_failed=files_failed,
        summaries_created=files_indexed,
        chunks_created=chunks_created,
        by_format=dict(sorted(by_format.items())),
        failures=tuple(failures),
        diagnostics=tuple(diagnostics),
    )


def summarize_data_file(file_record: PendingFileRecord) -> DataSummary:
    """Summarize one pending data-schema file without loading unsafe artifacts."""
    extension = file_record.extension
    if extension == ".csv":
        return summarize_csv(file_record.path, file_id=file_record.id)
    if extension == ".xlsx":
        return summarize_xlsx(file_record.path, file_id=file_record.id)
    if extension == ".json":
        return summarize_json(file_record.path, file_id=file_record.id)
    if extension == ".jsonl":
        return summarize_jsonl(file_record.path, file_id=file_record.id)
    if extension in {".sqlite", ".db"}:
        return summarize_sqlite(file_record.path, file_id=file_record.id)
    raise ExtractionFailure(
        f"unsupported data schema extension: {extension or '<none>'}",
        extractor_name=DATA_SCHEMA_EXTRACTOR_NAME,
        extractor_version=DATA_SCHEMA_EXTRACTOR_VERSION,
        metadata_json=_json_dumps({"extension": extension}),
    )


def data_summary_to_chunks(
    file_record: PendingFileRecord,
    summary: DataSummary,
) -> tuple[ChunkRecord, ...]:
    raw_chunks = tuple(
        RawChunk(
            source_type=DATA_SCHEMA_SOURCE_TYPE,
            title=f"Data schema: {file_record.filename} ({section.name})",
            text=section.summary_text,
            location_type="schema",
            location_value=section.location_value,
            metadata={
                "format": summary.format,
                "section_kind": section.kind,
                "section_name": section.name,
                "row_count": section.row_count,
                "column_count": section.column_count,
            },
        )
        for section in summary.sections
    )
    return finalize_chunks(
        file_record=file_record,
        raw_chunks=raw_chunks,
        max_tokens=DEFAULT_MAX_CHUNK_TOKENS,
    )


def _extracted_document_from_summary(
    *,
    file_record: PendingFileRecord,
    summary: DataSummary,
    chunks: tuple[ChunkRecord, ...],
) -> ExtractedDocument:
    return ExtractedDocument(
        file_id=file_record.id,
        extractor_name=DATA_SCHEMA_EXTRACTOR_NAME,
        extractor_version=DATA_SCHEMA_EXTRACTOR_VERSION,
        status="indexed",
        text_length=sum(len(chunk.text) for chunk in chunks),
        chunk_count=len(chunks),
        metadata_json=_json_dumps(
            {
                "extension": file_record.extension,
                "relative_path": file_record.relative_path,
                "content_hash": file_record.content_hash,
                "format": summary.format,
                "max_chunk_tokens": DEFAULT_MAX_CHUNK_TOKENS,
            }
        ),
        error=None,
        chunks=tuple(chunks),
    )
