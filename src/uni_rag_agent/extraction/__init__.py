"""Text extraction and chunking public API."""

from .constants import (
    DEFAULT_MAX_CHUNK_TOKENS,
    LEGACY_FORMAT_REASON,
    SCANNED_PDF_OCR_REASON,
    SUPPORTED_TEXT_EXTENSIONS,
    TEXT_EXTRACTION_CATEGORIES,
)
from .core import extract_file, extract_pending_files, load_extraction_status
from .models import (
    ChunkRecord,
    ExtractedDocument,
    ExtractionError,
    ExtractionFailure,
    ExtractionFailureSummary,
    ExtractionRunResult,
    ExtractionStatus,
    PendingFileRecord,
    RawChunk,
)

__all__ = [
    "DEFAULT_MAX_CHUNK_TOKENS",
    "LEGACY_FORMAT_REASON",
    "SCANNED_PDF_OCR_REASON",
    "SUPPORTED_TEXT_EXTENSIONS",
    "TEXT_EXTRACTION_CATEGORIES",
    "ChunkRecord",
    "ExtractedDocument",
    "ExtractionError",
    "ExtractionFailure",
    "ExtractionFailureSummary",
    "ExtractionRunResult",
    "ExtractionStatus",
    "PendingFileRecord",
    "RawChunk",
    "extract_file",
    "extract_pending_files",
    "load_extraction_status",
]
