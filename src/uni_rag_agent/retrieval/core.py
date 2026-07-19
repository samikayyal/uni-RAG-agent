"""Read-only hybrid retrieval orchestration."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from uni_rag_agent.config import Config, validate_config
from uni_rag_agent.indexing import (
    SemanticSearchError,
    keyword_search_terms,
    semantic_search_many,
)
from uni_rag_agent.indexing.models import KeywordSearchError
from uni_rag_agent.indexing.profiles import resolve_embedding_profile
from uni_rag_agent.storage import StorageError, check_storage

from .metadata import MetadataSearchError, metadata_search
from .models import FusedRetrievalResult, RetrievalResultSet, RetrievalRun
from .planner import (
    MAX_QUERY_PLAN_CONTEXT_MESSAGES,
    QueryPlanningError,
    normalize_query,
    plan_query,
)
from .rrf import merge_with_rrf


class RetrievalError(RuntimeError):
    """Raised when a configured retrieval backend fails."""

    def __init__(self, message: str, *, search_run_id: int | None = None) -> None:
        super().__init__(message)
        self.search_run_id = search_run_id


@dataclass(frozen=True)
class _RetrievalExecution:
    """Public retrieval output plus the complete in-memory RRF ordering."""

    run: RetrievalRun
    all_fused_candidates: tuple[FusedRetrievalResult, ...]


class _SearchRunRecorder(Protocol):
    search_run_id: int

    def start(
        self,
        *,
        query: str,
        query_plan: object,
        embedding_model: str,
        conversation_message_count: int,
    ) -> None: ...

    def record_result_set(self, result_set: RetrievalResultSet) -> None: ...

    def record_fused_results(self, results: Sequence[FusedRetrievalResult]) -> None: ...

    def mark_failed(self, error: Exception) -> None: ...


def retrieve(
    config: Config,
    query: str,
    conversation_context: Sequence[dict[str, str]] | None = None,
    model: str | None = None,
    *,
    chat_model: object | None = None,
) -> RetrievalRun:
    """Run metadata, keyword, and semantic retrieval without persistence."""
    return _execute_retrieval(
        config,
        query,
        conversation_context=conversation_context,
        model=model,
        chat_model=chat_model,
    ).run


def _execute_retrieval(
    config: Config,
    query: str,
    conversation_context: Sequence[dict[str, str]] | None = None,
    model: str | None = None,
    *,
    chat_model: object | None = None,
    recorder: _SearchRunRecorder | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> _RetrievalExecution:
    """Execute planner, backends, and RRF with an optional persistence seam."""
    normalized_query = normalize_query(query)
    validate_config(config)
    profile = resolve_embedding_profile(config, model, error=RetrievalError)
    storage = check_storage(config)
    if not storage.ok:
        details = "; ".join(storage.diagnostics) or "storage is not ready"
        raise StorageError(f"Retrieval storage check failed: {details}")

    _report_progress(progress_callback, "planning")
    query_plan = plan_query(
        config,
        query,
        conversation_context=conversation_context,
        chat_model=chat_model,
    )
    if recorder is not None:
        recorder.start(
            query=query,
            query_plan=query_plan,
            embedding_model=profile.model_name,
            conversation_message_count=min(
                len(conversation_context or ()), MAX_QUERY_PLAN_CONTEXT_MESSAGES
            ),
        )
    if query_plan.query_type == "unknown_or_unsupported":
        run = RetrievalRun(
            query=query,
            embedding_model=profile.model_name,
            query_plan=query_plan,
            result_sets=(),
            results=(),
            searched_courses=(),
            searched_indexes=(),
            keyword_terms=(),
            semantic_queries=(),
            weaknesses=(query_plan.plan_reason,),
            status="unsupported",
        )
        return _RetrievalExecution(run=run, all_fused_candidates=())

    result_sets: list[RetrievalResultSet] = []
    try:
        metadata_results = metadata_search(
            config,
            normalized_query,
            courses=query_plan.candidate_courses,
            indexes=query_plan.candidate_indexes,
            extensions=_query_extensions(normalized_query),
        )
        result_sets.append(
            RetrievalResultSet(
                result_set_id="metadata",
                retrieval_method="metadata",
                query=normalized_query,
                results=tuple(metadata_results),
            )
        )
        if recorder is not None:
            recorder.record_result_set(result_sets[-1])

        _report_progress(progress_callback, "keyword_search")
        keyword_results = keyword_search_terms(
            config,
            query_plan.keyword_terms,
            courses=query_plan.candidate_courses,
            indexes=query_plan.candidate_indexes,
        )
        result_sets.append(
            RetrievalResultSet(
                result_set_id="keyword",
                retrieval_method="keyword",
                query=" OR ".join(query_plan.keyword_terms),
                results=tuple(keyword_results),
            )
        )
        if recorder is not None:
            recorder.record_result_set(result_sets[-1])

        _report_progress(progress_callback, "semantic_search")
        semantic_result_sets = semantic_search_many(
            config,
            query_plan.semantic_queries,
            courses=query_plan.candidate_courses,
            indexes=query_plan.candidate_indexes,
            model=profile.model_name,
        )
        if len(semantic_result_sets) != len(query_plan.semantic_queries):
            raise SemanticSearchError(
                "Semantic search returned an unexpected number of result sets."
            )
        for semantic_index, (semantic_query, semantic_results) in enumerate(
            zip(query_plan.semantic_queries, semantic_result_sets),
            start=1,
        ):
            result_sets.append(
                RetrievalResultSet(
                    result_set_id=f"semantic:{semantic_index}",
                    retrieval_method="semantic",
                    query=semantic_query,
                    results=tuple(semantic_results),
                )
            )
            if recorder is not None:
                recorder.record_result_set(result_sets[-1])
    except (MetadataSearchError, KeywordSearchError, SemanticSearchError) as exc:
        _mark_recorder_failed(recorder, exc)
        raise RetrievalError(
            f"Retrieval backend failed: {exc}",
            search_run_id=_recorder_search_run_id(recorder),
        ) from exc
    except (StorageError, QueryPlanningError) as exc:
        _mark_recorder_failed(recorder, exc)
        _attach_search_run_id(exc, recorder)
        raise
    except Exception as exc:  # noqa: BLE001 - backend failures are fatal
        _mark_recorder_failed(recorder, exc)
        raise RetrievalError(
            f"Retrieval backend failed: {exc}",
            search_run_id=_recorder_search_run_id(recorder),
        ) from exc

    try:
        all_results = merge_with_rrf(
            result_sets,
            k=config.rrf_k,
            final_top_k=None,
        )
    except Exception as exc:  # noqa: BLE001 - fusion failures are fatal
        _mark_recorder_failed(recorder, exc)
        raise RetrievalError(
            f"Retrieval fusion failed: {exc}",
            search_run_id=_recorder_search_run_id(recorder),
        ) from exc
    if recorder is not None:
        try:
            recorder.record_fused_results(all_results)
        except Exception as exc:  # noqa: BLE001 - persistence failures are fatal
            _mark_recorder_failed(recorder, exc)
            _attach_search_run_id(exc, recorder)
            raise
    results = all_results[: config.final_top_k]
    weaknesses = _weaknesses(
        result_sets,
        final_count=len(results),
        final_top_k=config.final_top_k,
    )
    run = RetrievalRun(
        query=query,
        embedding_model=profile.model_name,
        query_plan=query_plan,
        result_sets=tuple(result_sets),
        results=tuple(results),
        searched_courses=query_plan.candidate_courses,
        searched_indexes=query_plan.candidate_indexes,
        keyword_terms=query_plan.keyword_terms,
        semantic_queries=query_plan.semantic_queries,
        weaknesses=tuple(weaknesses),
        status="completed",
    )
    return _RetrievalExecution(
        run=run,
        all_fused_candidates=tuple(all_results),
    )


def _mark_recorder_failed(
    recorder: _SearchRunRecorder | None,
    error: Exception,
) -> None:
    if recorder is None:
        return
    try:
        recorder.mark_failed(error)
    except Exception:
        # Preserve the original retrieval/storage failure. The recorder owns
        # its own best-effort diagnostics and must not mask it.
        return


def _report_progress(
    progress_callback: Callable[[str], None] | None,
    phase: str,
) -> None:
    """Report an optional UI-only phase without affecting retrieval behavior."""
    if progress_callback is not None:
        progress_callback(phase)


def _recorder_search_run_id(recorder: _SearchRunRecorder | None) -> int | None:
    """Return the persisted run id when a failure happens after planning."""

    if recorder is None:
        return None
    value = getattr(recorder, "search_run_id", None)
    return value if isinstance(value, int) and value > 0 else None


def _attach_search_run_id(
    error: Exception,
    recorder: _SearchRunRecorder | None,
) -> None:
    run_id = _recorder_search_run_id(recorder)
    if run_id is None or getattr(error, "search_run_id", None) is not None:
        return
    try:
        error.search_run_id = run_id  # type: ignore[attr-defined]
    except Exception:
        return


def _query_extensions(query: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            token.casefold() for token in re.findall(r"(?i)\.[a-z0-9]+\b", query)
        )
    )


def _weaknesses(
    result_sets: Sequence[RetrievalResultSet],
    *,
    final_count: int,
    final_top_k: int,
) -> list[str]:
    weaknesses: list[str] = []
    by_method = {result_set.retrieval_method: result_set for result_set in result_sets}
    if not by_method.get("metadata", RetrievalResultSet("", "", "", ())).results:
        weaknesses.append("No current metadata files matched the query.")
    if not by_method.get("keyword", RetrievalResultSet("", "", "", ())).results:
        weaknesses.append("Keyword search returned no hits.")
    semantic_sets = [
        item for item in result_sets if item.retrieval_method == "semantic"
    ]
    for item in semantic_sets:
        if not item.results:
            weaknesses.append(f"Semantic query returned no hits: {item.query}")
    if final_count == 0:
        weaknesses.append("All retrieval result lists were empty.")
    elif final_count < final_top_k:
        weaknesses.append(
            f"Only {final_count} fused result(s) were available; requested {final_top_k}."
        )
    if final_count and all(
        result_set.retrieval_method == "metadata"
        for result_set in result_sets
        if result_set.results
    ):
        weaknesses.append("Retrieval produced only file-level metadata results.")
    if any(
        result.chunk_id is None
        and result.file_index_status
        in {"pending", "failed", "skipped", "metadata_only"}
        for result_set in result_sets
        for result in result_set.results
    ):
        weaknesses.append(
            "A matched file has no selectable evidence chunk because it is pending, failed, skipped, or metadata-only."
        )
    return weaknesses
