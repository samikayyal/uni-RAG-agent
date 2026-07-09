"""Indexing data contracts for keyword and vector indexing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class KeywordIndexError(RuntimeError):
    """Raised when keyword-index maintenance cannot be completed."""


class KeywordSearchError(ValueError):
    """Raised when a keyword query cannot be safely executed."""


class VectorIndexError(RuntimeError):
    """Raised when vector-index maintenance cannot be completed.

    Covers missing/unknown embedding profiles, missing optional embedding
    dependencies, gated/credential-less real providers, and ChromaDB failures.
    """


class SemanticSearchError(ValueError):
    """Raised when a semantic query cannot be safely executed."""


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


@dataclass(frozen=True)
class VectorIndexResult:
    rebuild: bool
    model: str
    provider: str
    embedding_dim: int
    collections: tuple[str, ...]
    chunks_seen: int
    rows_removed: int
    vectors_indexed: int
    embeddings_total: int
    by_source_type: Mapping[str, int]
    diagnostics: tuple[str, ...]

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "rebuild": self.rebuild,
            "model": self.model,
            "provider": self.provider,
            "embedding_dim": self.embedding_dim,
            "collections": list(self.collections),
            "chunks_seen": self.chunks_seen,
            "rows_removed": self.rows_removed,
            "vectors_indexed": self.vectors_indexed,
            "embeddings_total": self.embeddings_total,
            "by_source_type": dict(self.by_source_type),
            "diagnostics": list(self.diagnostics),
        }
