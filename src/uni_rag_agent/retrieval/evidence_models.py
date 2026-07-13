"""Strict, JSON-safe contracts for persisted evidence packets."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .models import (
    LOGICAL_INDEXES,
    QUERY_TYPES,
    QueryPlan,
    RetrievalContribution,
    RetrievalRun,
)

EVIDENCE_SOURCE_TYPES = (
    "document",
    "slides",
    "notebook",
    "code",
    "data_schema",
    "transcript",
)
SEARCHED_KEYS = ("courses", "indexes", "keyword_terms", "semantic_queries")
ANSWER_CONSTRAINTS = (
    "Answer only from evidence.",
    "Cite course and file.",
    "If evidence is insufficient, say so.",
)


class EvidenceModelError(ValueError):
    """Raised when an evidence model or persisted JSON is invalid."""


def canonical_json(value: object) -> str:
    """Serialize a safe value with the one packet-storage representation."""
    if hasattr(value, "as_safe_dict"):
        value = value.as_safe_dict()  # type: ignore[union-attr]
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class RetrievalSettings:
    llm_provider: str
    llm_model: str
    embedding_model: str
    keyword_top_k: int
    semantic_top_k: int
    metadata_top_k: int
    semantic_query_limit: int
    query_plan_min_confidence: float
    filename_fuzzy_threshold: int
    path_fuzzy_threshold: int
    rrf_k: int
    final_top_k: int
    evidence_max_tokens: int
    conversation_context_message_count: int

    def __post_init__(self) -> None:
        for name in ("llm_provider", "llm_model", "embedding_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise EvidenceModelError(f"{name} must be a nonblank string")
        for name in (
            "keyword_top_k",
            "semantic_top_k",
            "metadata_top_k",
            "semantic_query_limit",
            "final_top_k",
            "evidence_max_tokens",
        ):
            _positive_int(getattr(self, name), name)
        _nonnegative_int(self.rrf_k, "rrf_k")
        _nonnegative_int(
            self.conversation_context_message_count,
            "conversation_context_message_count",
        )
        if (
            not isinstance(self.query_plan_min_confidence, (int, float))
            or isinstance(self.query_plan_min_confidence, bool)
            or not 0.0 <= float(self.query_plan_min_confidence) <= 1.0
        ):
            raise EvidenceModelError(
                "query_plan_min_confidence must be a number between 0 and 1"
            )
        for name in ("filename_fuzzy_threshold", "path_fuzzy_threshold"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 0 <= value <= 100
            ):
                raise EvidenceModelError(f"{name} must be an integer between 0 and 100")

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "keyword_top_k": self.keyword_top_k,
            "semantic_top_k": self.semantic_top_k,
            "metadata_top_k": self.metadata_top_k,
            "semantic_query_limit": self.semantic_query_limit,
            "query_plan_min_confidence": self.query_plan_min_confidence,
            "filename_fuzzy_threshold": self.filename_fuzzy_threshold,
            "path_fuzzy_threshold": self.path_fuzzy_threshold,
            "rrf_k": self.rrf_k,
            "final_top_k": self.final_top_k,
            "evidence_max_tokens": self.evidence_max_tokens,
            "conversation_context_message_count": self.conversation_context_message_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RetrievalSettings":
        data = _exact_mapping(
            value, cls.__dataclass_fields__.keys(), "retrieval settings"
        )
        return cls(
            llm_provider=_required_string(data["llm_provider"], "llm_provider"),
            llm_model=_required_string(data["llm_model"], "llm_model"),
            embedding_model=_required_string(
                data["embedding_model"], "embedding_model"
            ),
            keyword_top_k=_required_int(data["keyword_top_k"], "keyword_top_k"),
            semantic_top_k=_required_int(data["semantic_top_k"], "semantic_top_k"),
            metadata_top_k=_required_int(data["metadata_top_k"], "metadata_top_k"),
            semantic_query_limit=_required_int(
                data["semantic_query_limit"], "semantic_query_limit"
            ),
            query_plan_min_confidence=_required_number(
                data["query_plan_min_confidence"], "query_plan_min_confidence"
            ),
            filename_fuzzy_threshold=_required_int(
                data["filename_fuzzy_threshold"], "filename_fuzzy_threshold"
            ),
            path_fuzzy_threshold=_required_int(
                data["path_fuzzy_threshold"], "path_fuzzy_threshold"
            ),
            rrf_k=_required_int(data["rrf_k"], "rrf_k"),
            final_top_k=_required_int(data["final_top_k"], "final_top_k"),
            evidence_max_tokens=_required_int(
                data["evidence_max_tokens"], "evidence_max_tokens"
            ),
            conversation_context_message_count=_required_int(
                data["conversation_context_message_count"],
                "conversation_context_message_count",
            ),
        )


@dataclass(frozen=True)
class EvidenceLocation:
    type: str | None
    value: str | None
    label: str

    def __post_init__(self) -> None:
        for name in ("type", "value"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise EvidenceModelError(
                    f"location {name} must be null or nonblank text"
                )
        if not isinstance(self.label, str) or self.label != location_label(
            self.type, self.value
        ):
            raise EvidenceModelError("location label is not deterministic")

    def as_safe_dict(self) -> dict[str, object]:
        return {"type": self.type, "value": self.value, "label": self.label}

    @classmethod
    def from_dict(cls, value: object) -> "EvidenceLocation":
        data = _exact_mapping(value, ("type", "value", "label"), "evidence location")
        location_type = _optional_string(data["type"], "location type")
        location_value = _optional_string(data["value"], "location value")
        return cls(
            type=location_type,
            value=location_value,
            label=_required_string(data["label"], "location label"),
        )


@dataclass(frozen=True)
class EvidenceItem:
    course: str
    file_id: int
    chunk_id: int
    file: str
    source_type: str
    location: EvidenceLocation
    text: str
    token_count: int
    rank: int
    score: float
    retrieval_method: str
    contributions: tuple[RetrievalContribution, ...]

    def __post_init__(self) -> None:
        for name in ("course", "file", "text", "source_type"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise EvidenceModelError(f"{name} must be nonblank")
        if self.source_type not in EVIDENCE_SOURCE_TYPES:
            raise EvidenceModelError(
                f"unsupported evidence source type: {self.source_type}"
            )
        for name in ("file_id", "chunk_id", "token_count", "rank"):
            _positive_int(getattr(self, name), name)
        if (
            not isinstance(self.score, (int, float))
            or isinstance(self.score, bool)
            or not math.isfinite(float(self.score))
        ):
            raise EvidenceModelError("score must be a finite number")
        if self.retrieval_method != "hybrid":
            raise EvidenceModelError("evidence retrieval_method must be exactly hybrid")
        if not isinstance(self.location, EvidenceLocation):
            raise EvidenceModelError("location must be an EvidenceLocation")
        if any(
            not isinstance(item, RetrievalContribution) for item in self.contributions
        ):
            raise EvidenceModelError(
                "contributions must contain RetrievalContribution values"
            )

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "course": self.course,
            "file_id": self.file_id,
            "chunk_id": self.chunk_id,
            "file": self.file,
            "source_type": self.source_type,
            "location": self.location.as_safe_dict(),
            "text": self.text,
            "token_count": self.token_count,
            "rank": self.rank,
            "score": self.score,
            "retrieval_method": self.retrieval_method,
            "contributions": [item.as_safe_dict() for item in self.contributions],
        }

    @classmethod
    def from_dict(cls, value: object) -> "EvidenceItem":
        data = _exact_mapping(
            value,
            (
                "course",
                "file_id",
                "chunk_id",
                "file",
                "source_type",
                "location",
                "text",
                "token_count",
                "rank",
                "score",
                "retrieval_method",
                "contributions",
            ),
            "evidence item",
        )
        contribution_values = _required_list(data["contributions"], "contributions")
        return cls(
            course=_required_string(data["course"], "course"),
            file_id=_required_int(data["file_id"], "file_id"),
            chunk_id=_required_int(data["chunk_id"], "chunk_id"),
            file=_required_string(data["file"], "file"),
            source_type=_required_string(data["source_type"], "source_type"),
            location=EvidenceLocation.from_dict(data["location"]),
            text=_required_string(data["text"], "text"),
            token_count=_required_int(data["token_count"], "token_count"),
            rank=_required_int(data["rank"], "rank"),
            score=_required_number(data["score"], "score"),
            retrieval_method=_required_string(
                data["retrieval_method"], "retrieval_method"
            ),
            contributions=tuple(
                _contribution_from_dict(item) for item in contribution_values
            ),
        )


@dataclass(frozen=True)
class SearchCoverage:
    search_run_id: int
    status: str
    searched_courses: tuple[str, ...]
    searched_indexes: tuple[str, ...]
    keyword_terms: tuple[str, ...]
    semantic_queries: tuple[str, ...]
    raw_result_count: int
    raw_result_counts_by_method: dict[str, int]
    fused_candidate_count: int
    selectable_candidate_count: int
    evidence_count: int
    evidence_token_count: int
    courses_with_chunk_hits: tuple[str, ...]
    indexes_with_chunk_hits: tuple[str, ...]
    source_types_with_chunk_hits: tuple[str, ...]
    courses_without_chunk_hits: tuple[str, ...]
    indexes_without_chunk_hits: tuple[str, ...]
    semantic_queries_without_hits: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    file_only_candidate_count: int
    token_budget_omission_count: int
    oversized_evidence_omission_count: int
    unselected_selectable_candidate_count: int
    weaknesses: tuple[str, ...]

    def __post_init__(self) -> None:
        _positive_int(self.search_run_id, "search_run_id")
        if not isinstance(self.status, str) or not self.status.strip():
            raise EvidenceModelError("coverage status must be nonblank")
        for name in (
            "searched_courses",
            "searched_indexes",
            "keyword_terms",
            "semantic_queries",
            "courses_with_chunk_hits",
            "indexes_with_chunk_hits",
            "source_types_with_chunk_hits",
            "courses_without_chunk_hits",
            "indexes_without_chunk_hits",
            "semantic_queries_without_hits",
            "missing_capabilities",
            "weaknesses",
        ):
            _string_sequence(getattr(self, name), name)
        for name in (
            "raw_result_count",
            "fused_candidate_count",
            "selectable_candidate_count",
            "evidence_count",
            "evidence_token_count",
            "file_only_candidate_count",
            "token_budget_omission_count",
            "oversized_evidence_omission_count",
            "unselected_selectable_candidate_count",
        ):
            _nonnegative_int(getattr(self, name), name)
        if not isinstance(self.raw_result_counts_by_method, dict):
            raise EvidenceModelError("raw_result_counts_by_method must be an object")
        if any(
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for key, value in self.raw_result_counts_by_method.items()
        ):
            raise EvidenceModelError(
                "raw result method counts must be nonnegative integers"
            )
        if set(self.raw_result_counts_by_method) != {"metadata", "keyword", "semantic"}:
            raise EvidenceModelError(
                "raw result method counts must contain metadata, keyword, and semantic"
            )
        if self.raw_result_count != sum(self.raw_result_counts_by_method.values()):
            raise EvidenceModelError("raw result count does not match method counts")
        if self.selectable_candidate_count > self.fused_candidate_count:
            raise EvidenceModelError(
                "selectable candidates cannot exceed fused candidates"
            )
        if self.evidence_count > self.selectable_candidate_count:
            raise EvidenceModelError(
                "evidence count cannot exceed selectable candidates"
            )
        if self.file_only_candidate_count > self.fused_candidate_count:
            raise EvidenceModelError(
                "file-only candidates cannot exceed fused candidates"
            )
        if self.unselected_selectable_candidate_count > self.selectable_candidate_count:
            raise EvidenceModelError(
                "unselected selectable candidates cannot exceed selectable candidates"
            )

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "search_run_id": self.search_run_id,
            "status": self.status,
            "searched_courses": list(self.searched_courses),
            "searched_indexes": list(self.searched_indexes),
            "keyword_terms": list(self.keyword_terms),
            "semantic_queries": list(self.semantic_queries),
            "raw_result_count": self.raw_result_count,
            "raw_result_counts_by_method": dict(self.raw_result_counts_by_method),
            "fused_candidate_count": self.fused_candidate_count,
            "selectable_candidate_count": self.selectable_candidate_count,
            "evidence_count": self.evidence_count,
            "evidence_token_count": self.evidence_token_count,
            "courses_with_chunk_hits": list(self.courses_with_chunk_hits),
            "indexes_with_chunk_hits": list(self.indexes_with_chunk_hits),
            "source_types_with_chunk_hits": list(self.source_types_with_chunk_hits),
            "courses_without_chunk_hits": list(self.courses_without_chunk_hits),
            "indexes_without_chunk_hits": list(self.indexes_without_chunk_hits),
            "semantic_queries_without_hits": list(self.semantic_queries_without_hits),
            "missing_capabilities": list(self.missing_capabilities),
            "file_only_candidate_count": self.file_only_candidate_count,
            "token_budget_omission_count": self.token_budget_omission_count,
            "oversized_evidence_omission_count": self.oversized_evidence_omission_count,
            "unselected_selectable_candidate_count": self.unselected_selectable_candidate_count,
            "weaknesses": list(self.weaknesses),
        }

    @classmethod
    def from_dict(cls, value: object) -> "SearchCoverage":
        fields = cls.__dataclass_fields__.keys()
        data = _exact_mapping(value, fields, "search coverage")
        return cls(
            search_run_id=_required_int(data["search_run_id"], "search_run_id"),
            status=_required_string(data["status"], "status"),
            searched_courses=_required_string_tuple(
                data["searched_courses"], "searched_courses"
            ),
            searched_indexes=_required_string_tuple(
                data["searched_indexes"], "searched_indexes"
            ),
            keyword_terms=_required_string_tuple(
                data["keyword_terms"], "keyword_terms"
            ),
            semantic_queries=_required_string_tuple(
                data["semantic_queries"], "semantic_queries"
            ),
            raw_result_count=_required_int(
                data["raw_result_count"], "raw_result_count"
            ),
            raw_result_counts_by_method=_required_int_dict(
                data["raw_result_counts_by_method"], "raw_result_counts_by_method"
            ),
            fused_candidate_count=_required_int(
                data["fused_candidate_count"], "fused_candidate_count"
            ),
            selectable_candidate_count=_required_int(
                data["selectable_candidate_count"], "selectable_candidate_count"
            ),
            evidence_count=_required_int(data["evidence_count"], "evidence_count"),
            evidence_token_count=_required_int(
                data["evidence_token_count"], "evidence_token_count"
            ),
            courses_with_chunk_hits=_required_string_tuple(
                data["courses_with_chunk_hits"], "courses_with_chunk_hits"
            ),
            indexes_with_chunk_hits=_required_string_tuple(
                data["indexes_with_chunk_hits"], "indexes_with_chunk_hits"
            ),
            source_types_with_chunk_hits=_required_string_tuple(
                data["source_types_with_chunk_hits"], "source_types_with_chunk_hits"
            ),
            courses_without_chunk_hits=_required_string_tuple(
                data["courses_without_chunk_hits"], "courses_without_chunk_hits"
            ),
            indexes_without_chunk_hits=_required_string_tuple(
                data["indexes_without_chunk_hits"], "indexes_without_chunk_hits"
            ),
            semantic_queries_without_hits=_required_string_tuple(
                data["semantic_queries_without_hits"], "semantic_queries_without_hits"
            ),
            missing_capabilities=_required_string_tuple(
                data["missing_capabilities"], "missing_capabilities"
            ),
            file_only_candidate_count=_required_int(
                data["file_only_candidate_count"], "file_only_candidate_count"
            ),
            token_budget_omission_count=_required_int(
                data["token_budget_omission_count"], "token_budget_omission_count"
            ),
            oversized_evidence_omission_count=_required_int(
                data["oversized_evidence_omission_count"],
                "oversized_evidence_omission_count",
            ),
            unselected_selectable_candidate_count=_required_int(
                data["unselected_selectable_candidate_count"],
                "unselected_selectable_candidate_count",
            ),
            weaknesses=_required_string_tuple(data["weaknesses"], "weaknesses"),
        )


@dataclass(frozen=True)
class EvidencePacket:
    search_run_id: int
    query: str
    interpreted_intent: str
    query_plan: QueryPlan
    retrieval_settings: RetrievalSettings
    searched: dict[str, tuple[str, ...]]
    coverage: SearchCoverage
    evidence: tuple[EvidenceItem, ...]
    weaknesses: tuple[str, ...]
    answer_constraints: tuple[str, ...]

    def __post_init__(self) -> None:
        _positive_int(self.search_run_id, "search_run_id")
        if not isinstance(self.query, str) or not self.query.strip():
            raise EvidenceModelError("packet query must be nonblank")
        if self.interpreted_intent != self.query_plan.query_type:
            raise EvidenceModelError("packet intent does not match query plan")
        if self.coverage.search_run_id != self.search_run_id:
            raise EvidenceModelError("packet coverage run id does not match packet")
        if self.coverage.evidence_count != len(self.evidence):
            raise EvidenceModelError("packet evidence count does not match coverage")
        if self.coverage.evidence_token_count != sum(
            item.token_count for item in self.evidence
        ):
            raise EvidenceModelError(
                "packet evidence token count does not match coverage"
            )
        _validate_searched(self.searched)
        _string_sequence(self.weaknesses, "weaknesses")
        if tuple(self.weaknesses) != tuple(self.coverage.weaknesses):
            raise EvidenceModelError(
                "packet weaknesses do not match coverage weaknesses"
            )
        if tuple(self.answer_constraints) != ANSWER_CONSTRAINTS:
            raise EvidenceModelError(
                "packet answer constraints do not match the contract"
            )
        if any(not isinstance(item, EvidenceItem) for item in self.evidence):
            raise EvidenceModelError("packet evidence must contain EvidenceItem values")

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "search_run_id": self.search_run_id,
            "query": self.query,
            "interpreted_intent": self.interpreted_intent,
            "query_plan": self.query_plan.as_safe_dict(),
            "retrieval_settings": self.retrieval_settings.as_safe_dict(),
            "searched": {key: list(self.searched[key]) for key in SEARCHED_KEYS},
            "coverage": self.coverage.as_safe_dict(),
            "evidence": [item.as_safe_dict() for item in self.evidence],
            "weaknesses": list(self.weaknesses),
            "answer_constraints": list(self.answer_constraints),
        }

    @classmethod
    def from_dict(cls, value: object) -> "EvidencePacket":
        data = _exact_mapping(
            value,
            (
                "search_run_id",
                "query",
                "interpreted_intent",
                "query_plan",
                "retrieval_settings",
                "searched",
                "coverage",
                "evidence",
                "weaknesses",
                "answer_constraints",
            ),
            "evidence packet",
        )
        evidence_values = _required_list(data["evidence"], "evidence")
        return cls(
            search_run_id=_required_int(data["search_run_id"], "search_run_id"),
            query=_required_string(data["query"], "query"),
            interpreted_intent=_required_string(
                data["interpreted_intent"], "interpreted_intent"
            ),
            query_plan=_query_plan_from_dict(data["query_plan"]),
            retrieval_settings=RetrievalSettings.from_dict(data["retrieval_settings"]),
            searched=_searched_from_dict(data["searched"]),
            coverage=SearchCoverage.from_dict(data["coverage"]),
            evidence=tuple(EvidenceItem.from_dict(item) for item in evidence_values),
            weaknesses=_required_string_tuple(data["weaknesses"], "weaknesses"),
            answer_constraints=_required_string_tuple(
                data["answer_constraints"], "answer_constraints"
            ),
        )


@dataclass(frozen=True)
class EvidenceBuildResult:
    search_run_id: int
    evidence_packet_id: int
    retrieval_run: RetrievalRun
    coverage: SearchCoverage
    packet: EvidencePacket

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "search_run_id": self.search_run_id,
            "evidence_packet_id": self.evidence_packet_id,
            "retrieval_run": self.retrieval_run.as_safe_dict(),
            "coverage": self.coverage.as_safe_dict(),
            "packet": self.packet.as_safe_dict(),
        }


def location_label(location_type: str | None, location_value: str | None) -> str:
    """Build the one deterministic label for an evidence location.

    This is the canonical definition; ``EvidenceLocation`` validates against it
    and packet assembly builds labels with it, so there is exactly one source of
    truth for the determinism guarantee.
    """
    if location_type is None and location_value is None:
        return "location unavailable"
    display_type = (
        location_type.replace("_", " ") if location_type is not None else None
    )
    if location_type is None:
        return str(location_value)
    if location_value is None:
        return str(display_type)
    return f"{display_type} {location_value}"


def _query_plan_from_dict(value: object) -> QueryPlan:
    data = _exact_mapping(
        value,
        (
            "query_type",
            "candidate_courses",
            "candidate_indexes",
            "keyword_terms",
            "semantic_queries",
            "needs_file_inspection",
            "needs_python",
            "plan_confidence",
            "plan_reason",
        ),
        "query plan",
    )
    query_type = _required_string(data["query_type"], "query_type")
    if query_type not in QUERY_TYPES:
        raise EvidenceModelError(f"unknown persisted query type: {query_type}")
    indexes = _required_string_tuple(data["candidate_indexes"], "candidate_indexes")
    if any(item not in LOGICAL_INDEXES for item in indexes):
        raise EvidenceModelError(
            "persisted query plan contains an unknown logical index"
        )
    courses = _required_string_tuple(data["candidate_courses"], "candidate_courses")
    keyword_terms = _required_string_tuple(data["keyword_terms"], "keyword_terms")
    semantic_queries = _required_string_tuple(
        data["semantic_queries"], "semantic_queries"
    )
    if type(data["needs_file_inspection"]) is not bool:
        raise EvidenceModelError("needs_file_inspection must be boolean")
    if type(data["needs_python"]) is not bool:
        raise EvidenceModelError("needs_python must be boolean")
    confidence = _required_number(data["plan_confidence"], "plan_confidence")
    if not 0.0 <= confidence <= 1.0:
        raise EvidenceModelError("plan_confidence must be between 0 and 1")
    reason = _required_string(data["plan_reason"], "plan_reason")
    if query_type == "unknown_or_unsupported":
        if courses or indexes or keyword_terms or semantic_queries:
            raise EvidenceModelError(
                "unsupported persisted plans must have empty scopes"
            )
    elif not courses or not indexes or not keyword_terms or not semantic_queries:
        raise EvidenceModelError(
            "supported persisted plans must have nonempty retrieval scopes"
        )
    return QueryPlan(
        query_type=query_type,
        candidate_courses=courses,
        candidate_indexes=indexes,
        keyword_terms=keyword_terms,
        semantic_queries=semantic_queries,
        needs_file_inspection=data["needs_file_inspection"],
        needs_python=data["needs_python"],
        plan_confidence=confidence,
        plan_reason=reason,
    )


def _contribution_from_dict(value: object) -> RetrievalContribution:
    data = _exact_mapping(
        value,
        (
            "result_set_id",
            "retrieval_method",
            "semantic_query",
            "semantic_query_index",
            "source_rank",
            "native_score",
            "rrf_contribution",
        ),
        "retrieval contribution",
    )
    semantic_query = _optional_string(data["semantic_query"], "semantic_query")
    semantic_index = data["semantic_query_index"]
    if semantic_index is not None:
        semantic_index = _required_int(semantic_index, "semantic_query_index")
    return RetrievalContribution(
        result_set_id=_required_string(data["result_set_id"], "result_set_id"),
        retrieval_method=_required_string(data["retrieval_method"], "retrieval_method"),
        semantic_query=semantic_query,
        semantic_query_index=semantic_index,
        source_rank=_required_int(data["source_rank"], "source_rank"),
        native_score=_required_number(data["native_score"], "native_score"),
        rrf_contribution=_required_number(data["rrf_contribution"], "rrf_contribution"),
    )


def _validate_searched(value: Mapping[str, object]) -> None:
    if set(value) != set(SEARCHED_KEYS):
        raise EvidenceModelError("searched must contain exactly the contract keys")
    for key in SEARCHED_KEYS:
        _string_sequence(value[key], f"searched.{key}")


def _searched_from_dict(value: object) -> dict[str, tuple[str, ...]]:
    data = _exact_mapping(value, SEARCHED_KEYS, "searched")
    _validate_searched(data)
    return {
        key: _required_string_tuple(data[key], f"searched.{key}")
        for key in SEARCHED_KEYS
    }


def _exact_mapping(value: object, keys: Sequence[str], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise EvidenceModelError(f"{label} must contain exactly the required fields")
    return {str(key): value[key] for key in keys}


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceModelError(f"{label} must be nonblank text")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise EvidenceModelError(f"{label} must be null or nonblank text")
    return value


def _required_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise EvidenceModelError(f"{label} must be an integer")
    return value


def _required_number(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise EvidenceModelError(f"{label} must be a finite number")
    return float(value)


def _required_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise EvidenceModelError(f"{label} must be a JSON array")
    return value


def _required_string_tuple(value: object, label: str) -> tuple[str, ...]:
    return tuple(_required_string(item, label) for item in _required_list(value, label))


def _string_sequence(value: object, label: str) -> None:
    if not isinstance(value, (tuple, list)) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise EvidenceModelError(f"{label} must contain only nonblank strings")


def _required_int_dict(value: object, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise EvidenceModelError(f"{label} must be an object")
    result: dict[str, int] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(item, int)
            or isinstance(item, bool)
        ):
            raise EvidenceModelError(f"{label} must contain integer values")
        result[key] = item
    return result


def _positive_int(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise EvidenceModelError(f"{label} must be a positive integer")


def _nonnegative_int(value: object, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise EvidenceModelError(f"{label} must be a nonnegative integer")
