"""Text extraction data contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class ExtractionError(RuntimeError):
    """Raised when an extraction run cannot be completed."""


class ExtractionFailure(RuntimeError):
    """Raised for a single file that cannot be extracted."""

    def __init__(
        self,
        message: str,
        *,
        extractor_name: str,
        extractor_version: str | None = None,
        metadata_json: str | None = None,
    ) -> None:
        super().__init__(message)
        self.extractor_name = extractor_name
        self.extractor_version = extractor_version
        self.metadata_json = metadata_json


@dataclass(frozen=True)
class PendingFileRecord:
    id: int
    path: Path
    relative_path: str
    filename: str
    extension: str
    category: str
    content_hash: str | None


@dataclass(frozen=True)
class RawChunk:
    source_type: str
    title: str | None
    text: str
    location_type: str
    location_value: str
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class ChunkRecord:
    file_id: int
    chunk_uid: str
    source_type: str
    chunk_index: int
    title: str | None
    text: str
    token_count: int
    location_type: str
    location_value: str
    metadata_json: str


@dataclass(frozen=True)
class ExtractedDocument:
    file_id: int
    extractor_name: str
    extractor_version: str | None
    status: str
    text_length: int
    chunk_count: int
    metadata_json: str
    error: str | None
    chunks: tuple[ChunkRecord, ...]


@dataclass(frozen=True)
class ExtractionFailureSummary:
    file_id: int
    path: str
    error: str


@dataclass(frozen=True)
class ExtractionRunResult:
    run_id: int
    started_at: str
    finished_at: str
    status: str
    category: str | None
    files_seen: int
    files_indexed: int
    files_failed: int
    chunks_created: int
    by_source_type: Mapping[str, int]
    failures: tuple[ExtractionFailureSummary, ...]
    diagnostics: tuple[str, ...]

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "category": self.category,
            "files_seen": self.files_seen,
            "files_indexed": self.files_indexed,
            "files_failed": self.files_failed,
            "chunks_created": self.chunks_created,
            "by_source_type": dict(self.by_source_type),
            "failures": [failure.__dict__ for failure in self.failures],
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class ExtractionStatus:
    latest_extraction_run_id: int | None
    latest_extraction_started_at: str | None
    pending_text_files: int
    indexed_text_files: int
    failed_text_files: int
    extracted_documents: int
    chunks_total: int
    pending_by_category: Mapping[str, int]
    chunks_by_source_type: Mapping[str, int]
    recent_failures: tuple[ExtractionFailureSummary, ...]

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "latest_extraction_run_id": self.latest_extraction_run_id,
            "latest_extraction_started_at": self.latest_extraction_started_at,
            "pending_text_files": self.pending_text_files,
            "indexed_text_files": self.indexed_text_files,
            "failed_text_files": self.failed_text_files,
            "extracted_documents": self.extracted_documents,
            "chunks_total": self.chunks_total,
            "pending_by_category": dict(self.pending_by_category),
            "chunks_by_source_type": dict(self.chunks_by_source_type),
            "recent_failures": [failure.__dict__ for failure in self.recent_failures],
        }


@dataclass(frozen=True)
class DataSummarySection:
    name: str
    kind: str
    location_value: str
    row_count: int | None
    column_count: int | None
    columns: tuple[Mapping[str, object], ...]
    sample_rows: tuple[Mapping[str, object], ...]
    summary_text: str


@dataclass(frozen=True)
class DataSummary:
    file_id: int
    format: str
    row_count: int | None
    column_count: int | None
    table_count: int | None
    sheet_count: int | None
    schema_json: str
    sample_json: str | None
    summary_text: str
    sections: tuple[DataSummarySection, ...]


@dataclass(frozen=True)
class DataSummaryRunResult:
    run_id: int
    started_at: str
    finished_at: str
    status: str
    file_id: int | None
    files_seen: int
    files_indexed: int
    files_failed: int
    summaries_created: int
    chunks_created: int
    by_format: Mapping[str, int]
    failures: tuple[ExtractionFailureSummary, ...]
    diagnostics: tuple[str, ...]

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "file_id": self.file_id,
            "files_seen": self.files_seen,
            "files_indexed": self.files_indexed,
            "files_failed": self.files_failed,
            "summaries_created": self.summaries_created,
            "chunks_created": self.chunks_created,
            "by_format": dict(self.by_format),
            "failures": [failure.__dict__ for failure in self.failures],
            "diagnostics": list(self.diagnostics),
        }
