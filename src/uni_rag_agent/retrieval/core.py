"""Read-only hybrid retrieval orchestration."""

from __future__ import annotations

import re
from collections.abc import Sequence

from uni_rag_agent.config import Config, validate_config
from uni_rag_agent.indexing import (
    SemanticSearchError,
    keyword_search_terms,
    semantic_search,
)
from uni_rag_agent.indexing.models import KeywordSearchError
from uni_rag_agent.indexing.profiles import resolve_embedding_profile
from uni_rag_agent.storage import StorageError, check_storage

from .metadata import MetadataSearchError, metadata_search
from .models import RetrievalResultSet, RetrievalRun, RouterOutput
from .router import RoutingError, normalize_query, route_query, validate_router_output
from .rrf import merge_with_rrf


class RetrievalError(RuntimeError):
    """Raised when a configured retrieval backend fails."""


def retrieve(
    config: Config,
    query: str,
    router_output: RouterOutput | None = None,
    conversation_context: Sequence[dict[str, str]] | None = None,
    model: str | None = None,
) -> RetrievalRun:
    """Run metadata, keyword, and semantic retrieval without persistence."""
    normalized_query = normalize_query(query)
    validate_config(config)
    profile = resolve_embedding_profile(config, model, error=RetrievalError)
    storage = check_storage(config)
    if not storage.ok:
        details = "; ".join(storage.diagnostics) or "storage is not ready"
        raise StorageError(f"Retrieval storage check failed: {details}")

    output = router_output or route_query(
        config,
        query,
        conversation_context=conversation_context,
    )
    validate_router_output(config, output)
    if (
        output.route_source == "unsupported"
        or output.query_type == "unknown_or_unsupported"
    ):
        return RetrievalRun(
            query=query,
            embedding_model=profile.model_name,
            router_output=output,
            result_sets=(),
            results=(),
            searched_courses=(),
            searched_indexes=(),
            keyword_terms=(),
            semantic_queries=(),
            weaknesses=(output.route_reason,),
            status="unsupported",
        )

    result_sets: list[RetrievalResultSet] = []
    try:
        metadata_results = metadata_search(
            config,
            normalized_query,
            courses=output.candidate_courses,
            indexes=output.candidate_indexes,
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

        keyword_results = keyword_search_terms(
            config,
            output.keyword_terms,
            courses=output.candidate_courses,
            indexes=output.candidate_indexes,
        )
        result_sets.append(
            RetrievalResultSet(
                result_set_id="keyword",
                retrieval_method="keyword",
                query=" OR ".join(output.keyword_terms),
                results=tuple(keyword_results),
            )
        )

        for semantic_index, semantic_query in enumerate(
            output.semantic_queries, start=1
        ):
            semantic_results = semantic_search(
                config,
                semantic_query,
                courses=output.candidate_courses,
                indexes=output.candidate_indexes,
                model=model or profile.model_name,
            )
            result_sets.append(
                RetrievalResultSet(
                    result_set_id=f"semantic:{semantic_index}",
                    retrieval_method="semantic",
                    query=semantic_query,
                    results=tuple(semantic_results),
                )
            )
    except (MetadataSearchError, KeywordSearchError, SemanticSearchError) as exc:
        raise RetrievalError(f"Retrieval backend failed: {exc}") from exc
    except (StorageError, RoutingError):
        raise
    except Exception as exc:  # noqa: BLE001 - backend failures are fatal
        raise RetrievalError(f"Retrieval backend failed: {exc}") from exc

    results = merge_with_rrf(
        result_sets,
        k=config.rrf_k,
        final_top_k=config.final_top_k,
    )
    weaknesses = _weaknesses(
        result_sets,
        final_count=len(results),
        final_top_k=config.final_top_k,
    )
    return RetrievalRun(
        query=query,
        embedding_model=profile.model_name,
        router_output=output,
        result_sets=tuple(result_sets),
        results=tuple(results),
        searched_courses=output.candidate_courses,
        searched_indexes=output.candidate_indexes,
        keyword_terms=output.keyword_terms,
        semantic_queries=output.semantic_queries,
        weaknesses=tuple(weaknesses),
        status="completed",
    )


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
