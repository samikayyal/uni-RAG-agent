"""Keyword indexing data contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class KeywordIndexError(RuntimeError):
    """Raised when keyword-index maintenance cannot be completed."""


class KeywordSearchError(ValueError):
    """Raised when a keyword query cannot be safely executed."""


@dataclass(frozen=True)
class KeywordIndexResult:
    rebuild: bool
    rows_removed: int
    chunks_seen: int
    rows_indexed: int
    by_source_type: Mapping[str, int]
    diagnostics: tuple[str, ...]

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "rebuild": self.rebuild,
            "rows_removed": self.rows_removed,
            "chunks_seen": self.chunks_seen,
            "rows_indexed": self.rows_indexed,
            "by_source_type": dict(self.by_source_type),
            "diagnostics": list(self.diagnostics),
        }
