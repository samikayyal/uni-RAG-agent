from __future__ import annotations

import json
from pathlib import Path

import pytest

from uni_rag_agent.indexing import (
    KeywordSearchError,
    SemanticSearchError,
    keyword_search,
    keyword_search_terms,
    sync_keyword_index,
)
from uni_rag_agent.retrieval import (
    FusedRetrievalResult,
    QueryPlan,
    RetrievalResult,
    RetrievalResultSet,
    RetrievalError,
    MetadataSearchError,
    merge_with_rrf,
    metadata_search,
    retrieve,
)
from uni_rag_agent.storage import connect_sqlite
from tests.sqlite_helpers import insert_minimal_chunk
from tests.support import make_initialized_config


def test_plural_keyword_courses_are_filtered_before_global_limit(
    tmp_path: Path,
) -> None:
    config = make_initialized_config(
        tmp_path,
        llm_provider="ollama",
        llm_model="test-planner",
    )
    with connect_sqlite(config) as connection:
        first = insert_minimal_chunk(
            connection,
            config,
            course_name="Information Retrieval",
            filename="ir.md",
            text="BM25 retrieval",
        )
        second = insert_minimal_chunk(
            connection,
            config,
            course_name="High Preformance Computing for Big Data",
            filename="hpc.md",
            text="BM25 MapReduce",
        )
        connection.commit()
    sync_keyword_index(config)

    results = keyword_search(
        config,
        "BM25",
        courses=["information retrieval", "High Preformance Computing for Big Data"],
        top_k=2,
    )

    assert {result.chunk_id for result in results} == {first.chunk_id, second.chunk_id}
    with pytest.raises(Exception, match="either course or courses"):
        keyword_search(
            config,
            "BM25",
            course="Information Retrieval",
            courses=["Information Retrieval"],
        )
    assert keyword_search_terms(config, ["does-not-exist"], courses=[]) == []


def test_metadata_search_returns_file_level_rows_and_excludes_historical_missing(
    tmp_path: Path,
) -> None:
    config = make_initialized_config(
        tmp_path,
        llm_provider="ollama",
        llm_model="test-planner",
    )
    with connect_sqlite(config) as connection:
        image = insert_minimal_chunk(
            connection,
            config,
            course_name="Information Retrieval",
            filename="diagram.png",
            extension=".png",
            category="image_metadata_only",
            index_status="metadata_only",
            source_type="document",
            text="metadata only",
        )
        insert_minimal_chunk(
            connection,
            config,
            course_name="Information Retrieval",
            filename="old.png",
            extension=".png",
            category="image_metadata_only",
            index_status="skipped",
            source_type="document",
            text="historical",
        )
        connection.execute(
            "UPDATE files SET reason_not_indexed = ? WHERE filename = ?",
            ("missing from latest inventory run", "old.png"),
        )
        connection.commit()

    results = metadata_search(
        config,
        "Find diagram.png",
        courses=["information retrieval"],
        indexes=["document_index"],
        extensions=[".png"],
    )

    assert [result.file_id for result in results] == [image.file_id]
    assert results[0].chunk_id is None
    assert results[0].source_type is None
    assert results[0].file_index_status == "metadata_only"
    assert "filename" in results[0].matched_fields


def _result(
    *,
    chunk_id: int,
    file_id: int = 1,
    method: str = "keyword",
    rank: int = 1,
    score: float = 1.0,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        file_id=file_id,
        course="Information Retrieval",
        file_path=f"Courses/Information Retrieval/{file_id}.md",
        source_type="document",
        location_type="page",
        location_value=str(chunk_id),
        rank=rank,
        score=score,
        snippet=f"chunk {chunk_id}",
        retrieval_method=method,
    )


def _supported_retrieval_plan() -> QueryPlan:
    return QueryPlan(
        query_type="concept_explanation",
        candidate_courses=("Information Retrieval",),
        candidate_indexes=("document_index",),
        keyword_terms=("BM25",),
        semantic_queries=("BM25 explanation",),
        needs_file_inspection=False,
        needs_python=False,
        plan_confidence=1.0,
        plan_reason="test plan",
    )


def test_rrf_preserves_semantic_expansion_and_metadata_provenance() -> None:
    metadata = RetrievalResult(
        chunk_id=None,
        file_id=1,
        course="Information Retrieval",
        file_path="Courses/Information Retrieval/1.md",
        source_type=None,
        location_type=None,
        location_value=None,
        rank=1,
        score=100.0,
        snippet="1.md | document | indexed",
        retrieval_method="metadata",
        matched_fields=("filename",),
    )
    fused = merge_with_rrf(
        [
            RetrievalResultSet(
                "keyword", "keyword", "BM25", (_result(chunk_id=1, score=0.1),)
            ),
            RetrievalResultSet(
                "semantic:1",
                "semantic",
                "BM25 explanation",
                (_result(chunk_id=1, method="semantic", score=0.2),),
            ),
            RetrievalResultSet(
                "semantic:2",
                "semantic",
                "distributed ranking",
                (_result(chunk_id=1, method="semantic", score=0.3),),
            ),
            RetrievalResultSet("metadata", "metadata", "Find 1.md", (metadata,)),
        ],
        k=60,
        final_top_k=10,
    )

    assert len(fused) == 1
    result = fused[0]
    assert isinstance(result, FusedRetrievalResult)
    assert result.retrieval_method == "hybrid"
    assert result.score == pytest.approx(4 / 61)
    assert {item.result_set_id for item in result.contributions} == {
        "keyword",
        "semantic:1",
        "semantic:2",
        "metadata",
    }
    semantic = [
        item for item in result.contributions if item.retrieval_method == "semantic"
    ]
    assert {item.semantic_query_index for item in semantic} == {1, 2}
    assert result.matched_fields == ("filename",)


def test_rrf_ties_have_a_deterministic_file_id_order() -> None:
    first = _result(chunk_id=20, file_id=20)
    second = _result(chunk_id=10, file_id=10)

    fused = merge_with_rrf(
        [
            RetrievalResultSet("keyword", "keyword", "query", (first,)),
            RetrievalResultSet("semantic:1", "semantic", "query", (second,)),
        ],
        k=60,
        final_top_k=10,
    )

    assert [result.chunk_id for result in fused] == [10, 20]
    assert [result.rank for result in fused] == [1, 2]


def test_retrieve_runs_each_semantic_query_without_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_initialized_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="test-planner",
    )
    with connect_sqlite(config) as connection:
        inserted = insert_minimal_chunk(
            connection,
            config,
            course_name="Information Retrieval",
            filename="notes.md",
            text="BM25 retrieval",
        )
        connection.commit()

    class FakeChat:
        def invoke(self, _: str) -> str:
            return json.dumps(
                {
                    "query_type": "concept_explanation",
                    "candidate_courses": ["Information Retrieval"],
                    "candidate_indexes": ["document_index"],
                    "keyword_terms": ["BM25"],
                    "semantic_queries": ["BM25 explanation", "ranking explanation"],
                    "needs_file_inspection": False,
                    "needs_python": False,
                    "plan_confidence": 1.0,
                    "plan_reason": "test plan",
                }
            )

    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.metadata_search",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.keyword_search_terms",
        lambda *args, **kwargs: [_result(chunk_id=inserted.chunk_id)],
    )
    semantic_queries: list[str] = []
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.semantic_search",
        lambda _config, query, **kwargs: (
            semantic_queries.append(query)
            or [_result(chunk_id=inserted.chunk_id, method="semantic")]
        ),
    )

    run = retrieve(
        config,
        "BM25",
        model="BAAI/bge-m3",
        chat_model=FakeChat(),
    )

    assert run.status == "completed"
    assert [item.result_set_id for item in run.result_sets] == [
        "metadata",
        "keyword",
        "semantic:1",
        "semantic:2",
    ]
    assert semantic_queries == ["BM25 explanation", "ranking explanation"]
    assert run.results[0].chunk_id == inserted.chunk_id
    with connect_sqlite(config) as connection:
        assert connection.execute("SELECT COUNT(*) FROM search_runs").fetchone()[0] == 0


def test_retrieve_passes_canonical_embedding_model_to_semantic_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_initialized_config(
        tmp_path,
        embedding_model="gemini-embedding-001",
        llm_provider="ollama",
        llm_model="test-planner",
    )
    with connect_sqlite(config) as connection:
        inserted = insert_minimal_chunk(
            connection,
            config,
            course_name="Information Retrieval",
            filename="notes.md",
            text="BM25 retrieval",
        )
        connection.commit()

    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.plan_query",
        lambda *args, **kwargs: _supported_retrieval_plan(),
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.metadata_search",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.keyword_search_terms",
        lambda *args, **kwargs: [],
    )
    models: list[str | None] = []

    def semantic_backend(
        _config: object, _query: str, **kwargs: object
    ) -> list[RetrievalResult]:
        models.append(kwargs.get("model"))  # type: ignore[arg-type]
        return [_result(chunk_id=inserted.chunk_id, method="semantic")]

    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.semantic_search",
        semantic_backend,
    )

    run = retrieve(config, "BM25", model="gemini-embedding-001")

    assert run.embedding_model == "google/gemini-embedding-001"
    assert models == ["google/gemini-embedding-001"]


def test_retrieve_requires_model_before_unsupported_routing(tmp_path: Path) -> None:
    config = make_initialized_config(
        tmp_path,
        llm_provider="ollama",
        llm_model="test-planner",
    )

    with pytest.raises(RetrievalError, match="No embedding model selected"):
        retrieve(config, "unsupported nonsense")


def test_retrieve_uses_planner_and_skips_backends_for_supported_unsupported_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_initialized_config(
        tmp_path,
        llm_provider="ollama",
        llm_model="test-planner",
    )
    planned: list[str] = []

    def planned_unsupported(*args: object, **kwargs: object) -> QueryPlan:
        planned.append(str(args[1]))
        return QueryPlan(
            query_type="unknown_or_unsupported",
            candidate_courses=(),
            candidate_indexes=(),
            keyword_terms=(),
            semantic_queries=(),
            needs_file_inspection=False,
            needs_python=False,
            plan_confidence=0.9,
            plan_reason="Outside the indexed course archive.",
        )

    monkeypatch.setattr("uni_rag_agent.retrieval.core.plan_query", planned_unsupported)
    for backend in ("metadata_search", "keyword_search_terms", "semantic_search"):
        monkeypatch.setattr(
            f"uni_rag_agent.retrieval.core.{backend}",
            lambda *args, **kwargs: pytest.fail("unsupported plans must not search"),
        )

    run = retrieve(config, "Write a poem", model="BAAI/bge-m3")

    assert planned == ["Write a poem"]
    assert run.status == "unsupported"
    assert run.results == ()
    assert run.weaknesses == ("Outside the indexed course archive.",)


@pytest.mark.parametrize(
    ("backend_name", "backend_error"),
    [
        ("metadata_search", MetadataSearchError("metadata unavailable")),
        ("keyword_search_terms", KeywordSearchError("keyword unavailable")),
        ("semantic_search", SemanticSearchError("semantic unavailable")),
    ],
    ids=["metadata", "keyword", "semantic"],
)
def test_retrieve_translates_each_backend_failure_to_retrieval_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
    backend_error: Exception,
) -> None:
    config = make_initialized_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="test-planner",
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.plan_query",
        lambda *args, **kwargs: _supported_retrieval_plan(),
    )

    def fail(*args: object, **kwargs: object) -> object:
        raise backend_error

    backend_functions = {
        "metadata_search": lambda *args, **kwargs: [],
        "keyword_search_terms": lambda *args, **kwargs: [],
        "semantic_search": lambda *args, **kwargs: [],
    }
    backend_functions[backend_name] = fail
    for name, function in backend_functions.items():
        monkeypatch.setattr(f"uni_rag_agent.retrieval.core.{name}", function)

    with pytest.raises(
        RetrievalError,
        match=f"^Retrieval backend failed: {backend_error}$",
    ):
        retrieve(config, "BM25", model="BAAI/bge-m3")


def test_retrieve_reports_complete_zero_hit_weaknesses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_initialized_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="test-planner",
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.plan_query",
        lambda *args, **kwargs: _supported_retrieval_plan(),
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.metadata_search",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.keyword_search_terms",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.semantic_search",
        lambda *args, **kwargs: [],
    )

    run = retrieve(config, "BM25", model="BAAI/bge-m3")

    assert run.status == "completed"
    assert run.results == ()
    assert run.weaknesses == (
        "No current metadata files matched the query.",
        "Keyword search returned no hits.",
        "Semantic query returned no hits: BM25 explanation",
        "All retrieval result lists were empty.",
    )


def test_retrieve_reports_partial_fusion_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_initialized_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="test-planner",
        final_top_k=3,
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.plan_query",
        lambda *args, **kwargs: _supported_retrieval_plan(),
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.metadata_search",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.keyword_search_terms",
        lambda *args, **kwargs: [_result(chunk_id=1)],
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.semantic_search",
        lambda *args, **kwargs: [],
    )

    run = retrieve(config, "BM25", model="BAAI/bge-m3")

    assert run.status == "completed"
    assert len(run.results) == 1
    assert run.weaknesses == (
        "No current metadata files matched the query.",
        "Semantic query returned no hits: BM25 explanation",
        "Only 1 fused result(s) were available; requested 3.",
    )


def test_retrieve_reports_metadata_only_non_selectable_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_initialized_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="test-planner",
        final_top_k=3,
    )
    metadata = RetrievalResult(
        chunk_id=None,
        file_id=1,
        course="Information Retrieval",
        file_path="Courses/Information Retrieval/diagram.png",
        source_type=None,
        location_type=None,
        location_value=None,
        rank=1,
        score=1.0,
        snippet="diagram.png | image_metadata_only | metadata_only",
        retrieval_method="metadata",
        file_category="image_metadata_only",
        file_index_status="metadata_only",
        reason_not_indexed="standalone image",
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.plan_query",
        lambda *args, **kwargs: _supported_retrieval_plan(),
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.metadata_search",
        lambda *args, **kwargs: [metadata],
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.keyword_search_terms",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.semantic_search",
        lambda *args, **kwargs: [],
    )

    run = retrieve(config, "diagram.png", model="BAAI/bge-m3")

    assert run.status == "completed"
    assert len(run.results) == 1
    assert run.weaknesses == (
        "Keyword search returned no hits.",
        "Semantic query returned no hits: BM25 explanation",
        "Only 1 fused result(s) were available; requested 3.",
        "Retrieval produced only file-level metadata results.",
        "A matched file has no selectable evidence chunk because it is pending, failed, skipped, or metadata-only.",
    )
