from __future__ import annotations

from pathlib import Path

import pytest

from uni_rag_agent.config import Config, load_config
from uni_rag_agent.indexing import (
    keyword_search,
    keyword_search_terms,
    sync_keyword_index,
)
from uni_rag_agent.retrieval import (
    FusedRetrievalResult,
    RetrievalResult,
    RetrievalResultSet,
    RouterOutput,
    RetrievalError,
    merge_with_rrf,
    metadata_search,
    retrieve,
)
from uni_rag_agent.storage import connect_sqlite, ensure_data_dirs, initialize_schema
from tests.sqlite_helpers import insert_minimal_chunk


def _config(tmp_path: Path) -> Config:
    (tmp_path / "Courses").mkdir()
    config = load_config(repo_root=tmp_path, env_file=tmp_path / "missing.env")
    ensure_data_dirs(config)
    with connect_sqlite(config) as connection:
        initialize_schema(connection)
    return config


def test_plural_keyword_courses_are_filtered_before_global_limit(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
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
    config = _config(tmp_path)
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
    config = _config(tmp_path)
    with connect_sqlite(config) as connection:
        inserted = insert_minimal_chunk(
            connection,
            config,
            course_name="Information Retrieval",
            filename="notes.md",
            text="BM25 retrieval",
        )
        connection.commit()
    route = RouterOutput(
        query_type="concept_explanation",
        candidate_courses=("Information Retrieval",),
        candidate_indexes=("document_index",),
        keyword_terms=("BM25",),
        semantic_queries=("BM25 explanation", "ranking explanation"),
        needs_keyword_search=True,
        needs_semantic_search=True,
        needs_file_inspection=False,
        needs_python=False,
        route_confidence=1.0,
        route_reason="test route",
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

    run = retrieve(config, "BM25", router_output=route, model="BAAI/bge-m3")

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


def test_retrieve_requires_model_before_unsupported_routing(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(RetrievalError, match="No embedding model selected"):
        retrieve(config, "unsupported nonsense")
