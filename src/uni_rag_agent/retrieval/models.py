"""Shared retrieval result contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: int
    file_id: int
    course: str | None
    file_path: str
    source_type: str
    location_type: str | None
    location_value: str | None
    rank: int
    score: float
    snippet: str
    retrieval_method: str = "keyword"
    vector_collection: str | None = None
    vector_id: str | None = None

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "file_id": self.file_id,
            "course": self.course,
            "file_path": self.file_path,
            "source_type": self.source_type,
            "location_type": self.location_type,
            "location_value": self.location_value,
            "rank": self.rank,
            "score": self.score,
            "snippet": self.snippet,
            "retrieval_method": self.retrieval_method,
            "vector_collection": self.vector_collection,
            "vector_id": self.vector_id,
        }
