"""Data schema summary extraction public API."""

from __future__ import annotations

from .builders import (
    DATA_SCHEMA_CATEGORY,
    DATA_SCHEMA_EXTENSIONS,
    DATA_SCHEMA_EXTRACTOR_NAME,
    DATA_SCHEMA_EXTRACTOR_VERSION,
    DATA_SCHEMA_SOURCE_TYPE,
    SAMPLE_ROW_LIMIT,
)
from .core import data_summary_to_chunks, summarize_data_file, summarize_data_files
from .formats import (
    summarize_csv,
    summarize_json,
    summarize_jsonl,
    summarize_sqlite,
    summarize_xlsx,
)

__all__ = [
    "DATA_SCHEMA_CATEGORY",
    "DATA_SCHEMA_EXTENSIONS",
    "DATA_SCHEMA_EXTRACTOR_NAME",
    "DATA_SCHEMA_EXTRACTOR_VERSION",
    "DATA_SCHEMA_SOURCE_TYPE",
    "SAMPLE_ROW_LIMIT",
    "data_summary_to_chunks",
    "summarize_csv",
    "summarize_data_file",
    "summarize_data_files",
    "summarize_json",
    "summarize_jsonl",
    "summarize_sqlite",
    "summarize_xlsx",
]
