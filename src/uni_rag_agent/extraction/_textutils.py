"""Shared text, serialization, timing, and failure helpers for extraction.

These helpers were previously copy-pasted across the extraction package. They
live here as the single source of truth so a change to truncation, JSON
serialization, or failure formatting is made in exactly one place.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .constants import ERROR_CHAR_LIMIT, TEXT_ENCODINGS
from .models import ExtractionFailure, ExtractionFailureSummary, PendingFileRecord


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, sort_keys=True)


def _count_tokens(text: str) -> int:
    return len(text.split())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_text_file(path: Path) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return path.read_text()


def _title_from_text(text: str, max_length: int = 80) -> str | None:
    for line in text.splitlines():
        normalized = " ".join(line.split())
        if normalized:
            return _truncate(normalized, max_length)
    return None


def _package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _format_exception(exc: Exception) -> str:
    message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return _truncate(message, ERROR_CHAR_LIMIT)


def _failure_from_exception(
    file_record: PendingFileRecord,
    exc: ExtractionFailure,
) -> ExtractionFailureSummary:
    return ExtractionFailureSummary(
        file_id=file_record.id,
        path=str(file_record.path),
        error=_truncate(str(exc), ERROR_CHAR_LIMIT),
    )
