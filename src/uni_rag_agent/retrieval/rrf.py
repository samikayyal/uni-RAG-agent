"""Deterministic, unweighted Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .models import (
    FusedRetrievalResult,
    RetrievalContribution,
    RetrievalResult,
    RetrievalResultSet,
)


@dataclass
class _FusedBuilder:
    result: RetrievalResult
    score: float = 0.0
    contributions: list[RetrievalContribution] = field(default_factory=list)

    @property
    def best_source_rank(self) -> int:
        return min((item.source_rank for item in self.contributions), default=10**9)


def merge_with_rrf(
    result_sets: Sequence[RetrievalResultSet | Sequence[RetrievalResult]],
    k: int = 60,
    final_top_k: int | None = 10,
) -> list[FusedRetrievalResult]:
    """Fuse raw ranked lists while preserving every contributing source.

    ``final_top_k=None`` returns the complete deterministic ordering. Existing
    callers retain the bounded default, while Feature 09 can persist every
    fused candidate before applying evidence eligibility and token limits.
    """
    if k < 0:
        raise ValueError("rrf k must not be negative")
    if final_top_k is not None and final_top_k <= 0:
        raise ValueError("final_top_k must be greater than zero")
    normalized = _normalize_result_sets(result_sets)
    builders: dict[tuple[str, int], _FusedBuilder] = {}
    metadata_sets: list[RetrievalResultSet] = []

    for result_set in normalized:
        if result_set.retrieval_method == "metadata":
            metadata_sets.append(result_set)
            continue
        for source_rank, result in enumerate(result_set.results, start=1):
            identity = _identity(result)
            contribution = _contribution(result_set, result, source_rank, k)
            builder = builders.get(identity)
            if builder is None:
                builder = _FusedBuilder(result=result)
                builders[identity] = builder
            else:
                builder.result = _preferred_result(builder.result, result)
            builder.score += contribution.rrf_contribution
            builder.contributions.append(contribution)

    for result_set in metadata_sets:
        for source_rank, metadata in enumerate(result_set.results, start=1):
            file_builders = [
                builder
                for builder in builders.values()
                if builder.result.file_id == metadata.file_id
                and builder.result.chunk_id is not None
            ]
            contribution = _contribution(result_set, metadata, source_rank, k)
            if file_builders:
                chosen = min(
                    file_builders,
                    key=lambda builder: (
                        -builder.score,
                        builder.best_source_rank,
                        -len(builder.contributions),
                        int(builder.result.chunk_id or 0),
                    ),
                )
                chosen.score += contribution.rrf_contribution
                chosen.contributions.append(contribution)
                chosen.result = _merge_metadata_fields(chosen.result, metadata)
                continue

            identity = _identity(metadata)
            builder = builders.get(identity)
            if builder is None:
                builder = _FusedBuilder(result=metadata)
                builders[identity] = builder
            else:
                builder.result = _preferred_result(builder.result, metadata)
            builder.score += contribution.rrf_contribution
            builder.contributions.append(contribution)

    ordered = sorted(
        builders.values(),
        key=lambda builder: (
            -builder.score,
            builder.best_source_rank,
            -len(builder.contributions),
            0 if builder.result.chunk_id is not None else 1,
            builder.result.file_id,
            builder.result.chunk_id if builder.result.chunk_id is not None else 10**18,
        ),
    )
    selected = ordered if final_top_k is None else ordered[:final_top_k]
    return [_to_fused(builder, rank) for rank, builder in enumerate(selected, start=1)]


def _normalize_result_sets(
    result_sets: Sequence[RetrievalResultSet | Sequence[RetrievalResult]],
) -> tuple[RetrievalResultSet, ...]:
    normalized: list[RetrievalResultSet] = []
    semantic_index = 0
    for index, result_set in enumerate(result_sets, start=1):
        if isinstance(result_set, RetrievalResultSet):
            normalized.append(result_set)
            continue
        method = "keyword" if index == 1 else "semantic"
        if method == "semantic":
            semantic_index += 1
        normalized.append(
            RetrievalResultSet(
                result_set_id=f"{method}:{semantic_index}"
                if method == "semantic"
                else method,
                retrieval_method=method,
                query="",
                results=tuple(result_set),
            )
        )
    return tuple(normalized)


def _identity(result: RetrievalResult) -> tuple[str, int]:
    if result.chunk_id is not None:
        return "chunk", result.chunk_id
    return "file", result.file_id


def _contribution(
    result_set: RetrievalResultSet,
    result: RetrievalResult,
    source_rank: int,
    k: int,
) -> RetrievalContribution:
    semantic_index: int | None = None
    semantic_query: str | None = None
    if result_set.retrieval_method == "semantic":
        semantic_query = result_set.query
        try:
            semantic_index = int(result_set.result_set_id.split(":", 1)[1])
        except (IndexError, ValueError):
            semantic_index = None
    return RetrievalContribution(
        result_set_id=result_set.result_set_id,
        retrieval_method=result_set.retrieval_method,
        semantic_query=semantic_query,
        semantic_query_index=semantic_index,
        source_rank=source_rank,
        native_score=float(result.score),
        rrf_contribution=1.0 / (k + source_rank),
    )


def _preferred_result(left: RetrievalResult, right: RetrievalResult) -> RetrievalResult:
    # Backends own their native rank; use the first result's text/metadata as
    # the stable base and merge additive metadata fields from later sources.
    matched = tuple(dict.fromkeys((*left.matched_fields, *right.matched_fields)))
    return RetrievalResult(
        chunk_id=left.chunk_id,
        file_id=left.file_id,
        course=left.course or right.course,
        file_path=left.file_path or right.file_path,
        source_type=left.source_type or right.source_type,
        location_type=left.location_type or right.location_type,
        location_value=left.location_value or right.location_value,
        rank=min(left.rank, right.rank),
        score=left.score,
        snippet=left.snippet or right.snippet,
        retrieval_method=left.retrieval_method,
        vector_collection=left.vector_collection or right.vector_collection,
        vector_id=left.vector_id or right.vector_id,
        file_category=left.file_category or right.file_category,
        file_index_status=left.file_index_status or right.file_index_status,
        reason_not_indexed=left.reason_not_indexed or right.reason_not_indexed,
        matched_fields=matched,
    )


def _merge_metadata_fields(
    result: RetrievalResult,
    metadata: RetrievalResult,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=result.chunk_id,
        file_id=result.file_id,
        course=result.course or metadata.course,
        file_path=result.file_path or metadata.file_path,
        source_type=result.source_type,
        location_type=result.location_type,
        location_value=result.location_value,
        rank=result.rank,
        score=result.score,
        snippet=result.snippet or metadata.snippet,
        retrieval_method=result.retrieval_method,
        vector_collection=result.vector_collection,
        vector_id=result.vector_id,
        file_category=result.file_category or metadata.file_category,
        file_index_status=result.file_index_status or metadata.file_index_status,
        reason_not_indexed=result.reason_not_indexed or metadata.reason_not_indexed,
        matched_fields=tuple(
            dict.fromkeys((*result.matched_fields, *metadata.matched_fields))
        ),
    )


def _to_fused(builder: _FusedBuilder, rank: int) -> FusedRetrievalResult:
    result = builder.result
    contributions = tuple(
        sorted(
            builder.contributions,
            key=lambda item: (
                item.result_set_id,
                item.source_rank,
                item.semantic_query_index
                if item.semantic_query_index is not None
                else -1,
            ),
        )
    )
    return FusedRetrievalResult(
        chunk_id=result.chunk_id,
        file_id=result.file_id,
        course=result.course,
        file_path=result.file_path,
        source_type=result.source_type,
        location_type=result.location_type,
        location_value=result.location_value,
        rank=rank,
        score=builder.score,
        snippet=result.snippet,
        contributions=contributions,
        vector_collection=result.vector_collection,
        vector_id=result.vector_id,
        file_category=result.file_category,
        file_index_status=result.file_index_status,
        reason_not_indexed=result.reason_not_indexed,
        matched_fields=result.matched_fields,
    )
