"""JSON-safe contracts shared by query planning, retrieval, and evidence work."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked result returned by a single retrieval backend.

    Metadata results are intentionally allowed to be file-level: they have no
    chunk or source location, while still carrying enough inventory metadata
    for debugging and for Feature 09 to decide whether they can become
    evidence.
    """

    chunk_id: int | None
    file_id: int
    course: str | None
    file_path: str
    source_type: str | None
    location_type: str | None
    location_value: str | None
    rank: int
    score: float
    snippet: str
    retrieval_method: str = "keyword"  # "semantic" or "metadata"
    vector_collection: str | None = None
    vector_id: str | None = None
    file_category: str | None = None
    file_index_status: str | None = None
    reason_not_indexed: str | None = None
    matched_fields: tuple[str, ...] = ()

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
            "file_category": self.file_category,
            "file_index_status": self.file_index_status,
            "reason_not_indexed": self.reason_not_indexed,
            "matched_fields": list(self.matched_fields),
        }


QUERY_TYPES = (
    "concept_explanation",
    "course_summary",
    "cross_course_comparison",
    "find_file",
    "assignment_or_project_lookup",
    "code_question",
    "data_question",
    "study_quiz",
    "portfolio_resume",
    "unknown_or_unsupported",
)

LOGICAL_INDEXES = (
    "document_index",
    "slides_index",
    "notebook_index",
    "code_index",
    "data_schema_index",
    "transcript_index",
)


@dataclass(frozen=True)
class QueryPlan:
    query_type: str
    candidate_courses: tuple[str, ...]
    candidate_indexes: tuple[str, ...]
    keyword_terms: tuple[str, ...]
    semantic_queries: tuple[str, ...]
    needs_file_inspection: bool
    needs_python: bool
    plan_confidence: float
    plan_reason: str

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "query_type": self.query_type,
            "candidate_courses": list(self.candidate_courses),
            "candidate_indexes": list(self.candidate_indexes),
            "keyword_terms": list(self.keyword_terms),
            "semantic_queries": list(self.semantic_queries),
            "needs_file_inspection": self.needs_file_inspection,
            "needs_python": self.needs_python,
            "plan_confidence": self.plan_confidence,
            "plan_reason": self.plan_reason,
        }


@dataclass(frozen=True)
class RetrievalContribution:
    result_set_id: str
    retrieval_method: str
    semantic_query: str | None
    semantic_query_index: int | None
    source_rank: int
    native_score: float
    rrf_contribution: float

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "result_set_id": self.result_set_id,
            "retrieval_method": self.retrieval_method,
            "semantic_query": self.semantic_query,
            "semantic_query_index": self.semantic_query_index,
            "source_rank": self.source_rank,
            "native_score": self.native_score,
            "rrf_contribution": self.rrf_contribution,
        }


@dataclass(frozen=True)
class FusedRetrievalResult:
    chunk_id: int | None
    file_id: int
    course: str | None
    file_path: str
    source_type: str | None
    location_type: str | None
    location_value: str | None
    rank: int
    score: float
    snippet: str
    contributions: tuple[RetrievalContribution, ...]
    retrieval_method: str = "hybrid"
    vector_collection: str | None = None
    vector_id: str | None = None
    file_category: str | None = None
    file_index_status: str | None = None
    reason_not_indexed: str | None = None
    matched_fields: tuple[str, ...] = ()

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
            "file_category": self.file_category,
            "file_index_status": self.file_index_status,
            "reason_not_indexed": self.reason_not_indexed,
            "matched_fields": list(self.matched_fields),
            "contributions": [item.as_safe_dict() for item in self.contributions],
        }


@dataclass(frozen=True)
class RetrievalResultSet:
    result_set_id: str
    retrieval_method: str
    query: str
    results: tuple[RetrievalResult, ...]

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "result_set_id": self.result_set_id,
            "retrieval_method": self.retrieval_method,
            "query": self.query,
            "results": [result.as_safe_dict() for result in self.results],
        }


@dataclass(frozen=True)
class RetrievalRun:
    query: str
    embedding_model: str
    query_plan: QueryPlan
    result_sets: tuple[RetrievalResultSet, ...]
    results: tuple[FusedRetrievalResult, ...]
    searched_courses: tuple[str, ...]
    searched_indexes: tuple[str, ...]
    keyword_terms: tuple[str, ...]
    semantic_queries: tuple[str, ...]
    weaknesses: tuple[str, ...]
    status: str = "completed"

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "embedding_model": self.embedding_model,
            "query_plan": self.query_plan.as_safe_dict(),
            "result_sets": [item.as_safe_dict() for item in self.result_sets],
            "results": [item.as_safe_dict() for item in self.results],
            "searched_courses": list(self.searched_courses),
            "searched_indexes": list(self.searched_indexes),
            "keyword_terms": list(self.keyword_terms),
            "semantic_queries": list(self.semantic_queries),
            "weaknesses": list(self.weaknesses),
            "status": self.status,
        }
