from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tests.sqlite_helpers import insert_minimal_chunk
from tests.support import make_config, make_initialized_config
from uni_rag_agent.config import ConfigError, load_config
from uni_rag_agent.retrieval import (
    EvidenceError,
    RetrievalError,
    build_evidence,
    explain_search_coverage,
    load_evidence_packet,
)
from uni_rag_agent.retrieval import core as retrieval_core
from uni_rag_agent.retrieval.evidence_models import (
    EvidenceLocation,
    EvidenceModelError,
    canonical_json,
)
from uni_rag_agent.retrieval.evidence_persistence import sanitize_error
from uni_rag_agent.retrieval.models import (
    QueryPlan,
    RetrievalResult,
)
from uni_rag_agent.storage import (
    StorageError,
    connect_sqlite,
    ensure_data_dirs,
    initialize_schema,
)
from uni_rag_agent.indexing.models import KeywordSearchError


def _plan(
    course: str = "Course A",
    *,
    indexes: tuple[str, ...] = ("document_index",),
    needs_file_inspection: bool = False,
    needs_python: bool = False,
) -> QueryPlan:
    return QueryPlan(
        query_type="concept_explanation",
        candidate_courses=(course,),
        candidate_indexes=indexes,
        keyword_terms=("BM25",),
        semantic_queries=("BM25 explanation",),
        needs_file_inspection=needs_file_inspection,
        needs_python=needs_python,
        plan_confidence=0.9,
        plan_reason="The fixture asks for a course concept.",
    )


def _result(
    config,
    *,
    chunk_id: int | None,
    file_id: int,
    rank: int,
    method: str,
    filename: str = "notes.md",
    text: str = "snippet only",
    course: str = "Course A",
    token_count: int | None = None,
    location_type: str | None = "page",
    location_value: str | None = "1",
) -> RetrievalResult:
    del token_count
    return RetrievalResult(
        chunk_id=chunk_id,
        file_id=file_id,
        course=course,
        file_path=str(config.courses_root / course / filename),
        source_type="document" if chunk_id is not None else None,
        location_type=location_type if chunk_id is not None else None,
        location_value=location_value if chunk_id is not None else None,
        rank=rank,
        score=float(rank),
        snippet=text,
        retrieval_method=method,
        file_category="document" if chunk_id is not None else "image_metadata_only",
        file_index_status="indexed" if chunk_id is not None else "metadata_only",
        reason_not_indexed=None if chunk_id is not None else "standalone image",
    )


def _patch_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    plan: QueryPlan,
    *,
    metadata_results=(),
    keyword_results=(),
    semantic_results=(),
) -> None:
    monkeypatch.setattr(retrieval_core, "plan_query", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        retrieval_core,
        "metadata_search",
        lambda *args, **kwargs: list(metadata_results),
    )
    monkeypatch.setattr(
        retrieval_core,
        "keyword_search_terms",
        lambda *args, **kwargs: list(keyword_results),
    )
    monkeypatch.setattr(
        retrieval_core,
        "semantic_search_many",
        lambda _config, queries, **kwargs: [list(semantic_results) for _ in queries],
    )


def test_evidence_budget_defaults_and_safe_override(tmp_path: Path) -> None:
    (tmp_path / "Courses").mkdir()
    config = load_config(repo_root=tmp_path, env_file=tmp_path / "missing.env")
    assert config.evidence_max_tokens == 12_000
    assert config.as_safe_dict()["evidence_max_tokens"] == 12_000

    env_file = tmp_path / ".env"
    env_file.write_text("UNI_RAG_EVIDENCE_MAX_TOKENS=321\n", encoding="utf-8")
    overridden = load_config(repo_root=tmp_path, env_file=env_file)
    assert overridden.evidence_max_tokens == 321


@pytest.mark.parametrize("value", ("", "abc", "0", "-1"))
def test_invalid_evidence_budget_fails_clearly(tmp_path: Path, value: str) -> None:
    (tmp_path / "Courses").mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(f"UNI_RAG_EVIDENCE_MAX_TOKENS={value}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="UNI_RAG_EVIDENCE_MAX_TOKENS"):
        load_config(repo_root=tmp_path, env_file=env_file)


def test_legacy_router_column_migrates_without_changing_rows(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    ensure_data_dirs(config)
    plan_json = '{"legacy":true}'
    with sqlite3.connect(config.sqlite_path) as connection:
        connection.execute(
            """
            CREATE TABLE search_runs (
                id INTEGER PRIMARY KEY,
                query TEXT NOT NULL,
                query_type TEXT,
                router_output_json TEXT NOT NULL,
                searched_courses_json TEXT NOT NULL,
                searched_indexes_json TEXT NOT NULL,
                keyword_terms_json TEXT NOT NULL,
                semantic_queries_json TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                weaknesses_json TEXT,
                error TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO search_runs (
                query, query_type, router_output_json, searched_courses_json,
                searched_indexes_json, keyword_terms_json, semantic_queries_json,
                started_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "q",
                "concept_explanation",
                plan_json,
                "[]",
                "[]",
                "[]",
                "[]",
                "now",
                "completed",
            ),
        )
        connection.commit()
    with connect_sqlite(config) as connection:
        initialize_schema(connection)
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(search_runs)")
        }
        row = connection.execute(
            "SELECT id, query_plan_json, retrieval_settings_json FROM search_runs"
        ).fetchone()
    assert "router_output_json" not in columns
    assert "query_plan_json" in columns
    assert row["id"] == 1
    assert row["query_plan_json"] == plan_json
    assert row["retrieval_settings_json"] == "{}"


def test_duplicate_packets_block_migration_without_deleting_rows(
    tmp_path: Path,
) -> None:
    config = make_initialized_config(tmp_path)
    with connect_sqlite(config) as connection:
        connection.execute("DROP INDEX idx_evidence_packets_search_run")
        run_id = connection.execute(
            """
            INSERT INTO search_runs (
                query, query_type, query_plan_json, searched_courses_json,
                searched_indexes_json, keyword_terms_json, semantic_queries_json,
                started_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "q",
                "concept_explanation",
                "{}",
                "[]",
                "[]",
                "[]",
                "[]",
                "now",
                "completed",
            ),
        ).lastrowid
        for _ in range(2):
            connection.execute(
                "INSERT INTO evidence_packets (search_run_id, packet_json, evidence_count, created_at) VALUES (?, ?, ?, ?)",
                (run_id, "{}", 0, "now"),
            )
        connection.commit()
        with pytest.raises(StorageError, match="duplicate packets"):
            initialize_schema(connection)
        assert (
            connection.execute("SELECT COUNT(*) FROM evidence_packets").fetchone()[0]
            == 2
        )


def test_supported_build_persists_raw_fused_and_authoritative_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_initialized_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="test-planner",
        final_top_k=2,
    )
    with connect_sqlite(config) as connection:
        rows = insert_minimal_chunk(
            connection,
            config,
            course_name="Course A",
            filename="notes.md",
            text="authoritative full chunk text",
            location_type="page",
            location_value="1",
        )
        connection.commit()
    plan = _plan()
    metadata = _result(
        config, chunk_id=None, file_id=rows.file_id, rank=1, method="metadata"
    )
    keyword = _result(
        config,
        chunk_id=rows.chunk_id,
        file_id=rows.file_id,
        rank=1,
        method="keyword",
        text="keyword snippet",
    )
    semantic = _result(
        config,
        chunk_id=rows.chunk_id,
        file_id=rows.file_id,
        rank=1,
        method="semantic",
        text="semantic snippet",
    )
    _patch_retrieval(
        monkeypatch,
        plan,
        metadata_results=(metadata,),
        keyword_results=(keyword,),
        semantic_results=(semantic,),
    )

    result = build_evidence(config, "Explain BM25", model="BAAI/bge-m3")

    assert result.coverage.raw_result_counts_by_method == {
        "metadata": 1,
        "keyword": 1,
        "semantic": 1,
    }
    assert result.coverage.fused_candidate_count == 1
    assert result.coverage.evidence_count == 1
    assert result.packet.evidence[0].text == "authoritative full chunk text"
    assert result.packet.evidence[0].contributions[0].retrieval_method in {
        "metadata",
        "keyword",
        "semantic",
    }
    loaded = load_evidence_packet(config, search_run_id=result.search_run_id)
    assert loaded == result.packet
    assert canonical_json(loaded) == canonical_json(result.packet)
    with connect_sqlite(config) as connection:
        rows_by_method = connection.execute(
            "SELECT retrieval_method, selected_for_evidence FROM search_results WHERE search_run_id = ? ORDER BY id",
            (result.search_run_id,),
        ).fetchall()
    assert [row["retrieval_method"] for row in rows_by_method] == [
        "metadata",
        "keyword",
        "semantic",
        "hybrid",
    ]
    assert [row["selected_for_evidence"] for row in rows_by_method] == [0, 0, 0, 1]


def test_supported_zero_hit_build_persists_empty_result_set_envelopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_initialized_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="test-planner",
    )
    _patch_retrieval(monkeypatch, _plan())

    result = build_evidence(config, "BM25", model="BAAI/bge-m3")

    assert result.coverage.status == "completed"
    assert result.coverage.evidence_count == 0
    assert result.coverage.raw_result_count == 0
    assert result.coverage.semantic_queries_without_hits == ("BM25 explanation",)
    with connect_sqlite(config) as connection:
        envelopes = connection.execute(
            """
            SELECT result_set_id, retrieval_method, query, result_count
            FROM search_result_sets
            WHERE search_run_id = ?
            ORDER BY id
            """,
            (result.search_run_id,),
        ).fetchall()
        assert [tuple(row) for row in envelopes] == [
            ("metadata", "metadata", "BM25", 0),
            ("keyword", "keyword", "BM25", 0),
            ("semantic:1", "semantic", "BM25 explanation", 0),
        ]
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM search_results WHERE search_run_id = ?",
                (result.search_run_id,),
            ).fetchone()[0]
            == 0
        )
    assert explain_search_coverage(config, result.search_run_id) == result.coverage


def test_partial_failure_keeps_empty_completed_envelope_and_marks_unreached_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_initialized_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="test-planner",
    )
    _patch_retrieval(monkeypatch, _plan())
    monkeypatch.setattr(
        retrieval_core,
        "keyword_search_terms",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            KeywordSearchError("keyword backend down")
        ),
    )

    with pytest.raises(RetrievalError, match="Retrieval backend failed"):
        build_evidence(config, "BM25", model="BAAI/bge-m3")
    with connect_sqlite(config) as connection:
        envelopes = connection.execute(
            """
            SELECT result_set_id, result_count
            FROM search_result_sets
            ORDER BY id
            """
        ).fetchall()
        assert [tuple(row) for row in envelopes] == [("metadata", 0)]
    coverage = explain_search_coverage(config, 1)
    assert coverage.status == "failed"
    assert coverage.raw_result_counts_by_method == {
        "metadata": 0,
        "keyword": 0,
        "semantic": 0,
    }
    assert coverage.semantic_queries_without_hits == ()


def test_coverage_does_not_call_metadata_only_chunk_candidates_chunk_hits(
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
        rows = insert_minimal_chunk(connection, config, course_name="Course A")
        connection.commit()
    metadata = _result(
        config,
        chunk_id=rows.chunk_id,
        file_id=rows.file_id,
        rank=1,
        method="metadata",
        location_type=None,
        location_value=None,
    )
    _patch_retrieval(monkeypatch, _plan(), metadata_results=(metadata,))

    result = build_evidence(config, "BM25", model="BAAI/bge-m3")

    assert result.coverage.selectable_candidate_count == 1
    assert result.coverage.courses_with_chunk_hits == ()
    assert result.coverage.indexes_with_chunk_hits == ()
    assert result.coverage.source_types_with_chunk_hits == ()
    assert result.coverage.courses_without_chunk_hits == ("Course A",)


@pytest.mark.parametrize(
    ("message", "secret"),
    (
        ("Bearer opaque-secret-value", "opaque-secret-value"),
        ("https://provider.test/search?token=url-secret&keep=1", "url-secret"),
        ("api_key=key-secret", "key-secret"),
        ("provider returned sk-live-secret", "sk-live-secret"),
    ),
)
def test_sanitize_error_redacts_credential_shapes(message: str, secret: str) -> None:
    sanitized = sanitize_error(message)

    assert secret not in sanitized
    assert "Traceback" not in sanitized
    assert len(sanitized) <= 500


def test_unsupported_build_creates_zero_evidence_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_initialized_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="test-planner",
    )
    plan = QueryPlan(
        query_type="unknown_or_unsupported",
        candidate_courses=(),
        candidate_indexes=(),
        keyword_terms=(),
        semantic_queries=(),
        needs_file_inspection=False,
        needs_python=False,
        plan_confidence=0.9,
        plan_reason="The request is outside the indexed course archive.",
    )
    _patch_retrieval(monkeypatch, plan)

    result = build_evidence(config, "unsupported", model="BAAI/bge-m3")

    assert result.coverage.status == "unsupported"
    assert result.coverage.evidence_count == 0
    assert result.packet.weaknesses[0] == plan.plan_reason
    with connect_sqlite(config) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM search_results").fetchone()[0] == 0
        )
        assert (
            connection.execute("SELECT status FROM search_runs").fetchone()[0]
            == "unsupported"
        )


def test_packet_loader_rejects_wrong_run_identity_and_invalid_run_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_initialized_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="test-planner",
    )
    _patch_retrieval(monkeypatch, _plan())
    result = build_evidence(config, "BM25", model="BAAI/bge-m3")

    with connect_sqlite(config) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT packet_json FROM evidence_packets WHERE id = ?",
                (result.evidence_packet_id,),
            ).fetchone()[0]
        )
        payload["search_run_id"] = result.search_run_id + 100
        payload["coverage"]["search_run_id"] = result.search_run_id + 100
        connection.execute(
            "UPDATE evidence_packets SET packet_json = ? WHERE id = ?",
            (json.dumps(payload), result.evidence_packet_id),
        )
        connection.commit()
    with pytest.raises(EvidenceError, match="wrong search run"):
        load_evidence_packet(config, search_run_id=result.search_run_id)

    with connect_sqlite(config) as connection:
        mismatched_payload = json.loads(canonical_json(result.packet))
        mismatched_payload["coverage"]["status"] = "completed"
        connection.execute(
            "UPDATE evidence_packets SET packet_json = ? WHERE id = ?",
            (canonical_json(mismatched_payload), result.evidence_packet_id),
        )
        connection.execute(
            "UPDATE search_runs SET status = 'unsupported' WHERE id = ?",
            (result.search_run_id,),
        )
        connection.commit()
    with pytest.raises(EvidenceError, match="does not match owning run status"):
        load_evidence_packet(config, search_run_id=result.search_run_id)
    with pytest.raises(EvidenceError, match="does not match owning run status"):
        explain_search_coverage(config, result.search_run_id)

    with connect_sqlite(config) as connection:
        mismatched_payload["coverage"]["status"] = "unsupported"
        connection.execute(
            "UPDATE evidence_packets SET packet_json = ? WHERE id = ?",
            (canonical_json(mismatched_payload), result.evidence_packet_id),
        )
        connection.execute(
            "UPDATE search_runs SET status = 'completed' WHERE id = ?",
            (result.search_run_id,),
        )
        connection.commit()
    with pytest.raises(EvidenceError, match="does not match owning run status"):
        load_evidence_packet(config, search_run_id=result.search_run_id)
    with pytest.raises(EvidenceError, match="does not match owning run status"):
        explain_search_coverage(config, result.search_run_id)

    with connect_sqlite(config) as connection:
        connection.execute(
            "UPDATE evidence_packets SET packet_json = ? WHERE id = ?",
            (canonical_json(result.packet), result.evidence_packet_id),
        )
        connection.execute(
            "UPDATE search_runs SET status = 'running' WHERE id = ?",
            (result.search_run_id,),
        )
        connection.commit()
    with pytest.raises(EvidenceError, match="invalid status"):
        load_evidence_packet(config, search_run_id=result.search_run_id)


def test_location_and_packet_models_reject_noncanonical_values() -> None:
    assert (
        EvidenceLocation(
            type="notebook_cell",
            value="23",
            label="notebook cell 23",
        ).label
        == "notebook cell 23"
    )
    with pytest.raises(EvidenceModelError, match="deterministic"):
        EvidenceLocation(
            type="notebook_cell",
            value="23",
            label="notebook_cell 23",
        )


def test_planning_failure_creates_no_persisted_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_initialized_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="test-planner",
    )
    from uni_rag_agent.retrieval.planner import QueryPlanningError

    monkeypatch.setattr(
        retrieval_core,
        "plan_query",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            QueryPlanningError("invalid plan")
        ),
    )
    with pytest.raises(QueryPlanningError):
        build_evidence(config, "bad", model="BAAI/bge-m3")
    with connect_sqlite(config) as connection:
        assert connection.execute("SELECT COUNT(*) FROM search_runs").fetchone()[0] == 0


def test_backend_failure_retains_completed_partial_result_sets(
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
        rows = insert_minimal_chunk(connection, config, course_name="Course A")
        connection.commit()
    plan = _plan()
    metadata = _result(
        config, chunk_id=None, file_id=rows.file_id, rank=1, method="metadata"
    )
    _patch_retrieval(monkeypatch, plan, metadata_results=(metadata,))
    monkeypatch.setattr(
        retrieval_core,
        "keyword_search_terms",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            KeywordSearchError("keyword backend down")
        ),
    )

    with pytest.raises(Exception, match="Retrieval backend failed"):
        build_evidence(config, "BM25", model="BAAI/bge-m3")
    with connect_sqlite(config) as connection:
        run = connection.execute("SELECT status, error FROM search_runs").fetchone()
        assert run["status"] == "failed"
        assert "Traceback" not in (run["error"] or "")
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM search_results WHERE retrieval_method = 'metadata'"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM search_results WHERE retrieval_method = 'hybrid'"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM evidence_packets").fetchone()[0]
            == 0
        )


def test_authoritative_drift_fails_without_packet(
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
        rows = insert_minimal_chunk(connection, config, course_name="Course A")
        connection.commit()
    result_row = _result(
        config, chunk_id=rows.chunk_id, file_id=rows.file_id, rank=1, method="keyword"
    )
    _patch_retrieval(monkeypatch, _plan(), keyword_results=(result_row,))
    from uni_rag_agent.retrieval import evidence as evidence_service

    original_hydrate = evidence_service._hydrate_candidate

    def delete_before_hydration(connection, candidate):
        connection.execute("DELETE FROM chunks WHERE id = ?", (rows.chunk_id,))
        return original_hydrate(connection, candidate)

    monkeypatch.setattr(evidence_service, "_hydrate_candidate", delete_before_hydration)

    with pytest.raises(EvidenceError, match="drift"):
        build_evidence(config, "BM25", model="BAAI/bge-m3")
    with connect_sqlite(config) as connection:
        run = connection.execute("SELECT status FROM search_runs").fetchone()
        assert run["status"] == "failed"
        assert (
            connection.execute("SELECT COUNT(*) FROM evidence_packets").fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT MAX(selected_for_evidence) FROM search_results"
            ).fetchone()[0]
            == 0
        )


def test_token_budget_skips_overflow_and_backfills_lower_ranked_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_initialized_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="test-planner",
        final_top_k=3,
        evidence_max_tokens=5,
    )
    with connect_sqlite(config) as connection:
        first = insert_minimal_chunk(
            connection, config, course_name="Course A", filename="one.md", text="one"
        )
        second = insert_minimal_chunk(
            connection, config, course_name="Course A", filename="two.md", text="two"
        )
        third = insert_minimal_chunk(
            connection,
            config,
            course_name="Course A",
            filename="three.md",
            text="three",
        )
        connection.executemany(
            "UPDATE chunks SET token_count = ? WHERE id = ?",
            ((4, first.chunk_id), (2, second.chunk_id), (1, third.chunk_id)),
        )
        connection.commit()
    results = tuple(
        _result(
            config,
            chunk_id=chunk_id,
            file_id=file_id,
            rank=rank,
            method="keyword",
            filename=filename,
            location_type=None,
            location_value=None,
        )
        for rank, (chunk_id, file_id, filename) in enumerate(
            (
                (first.chunk_id, first.file_id, "one.md"),
                (second.chunk_id, second.file_id, "two.md"),
                (third.chunk_id, third.file_id, "three.md"),
            ),
            start=1,
        )
    )
    _patch_retrieval(monkeypatch, _plan(), keyword_results=results)

    result = build_evidence(config, "BM25", model="BAAI/bge-m3")

    assert [item.chunk_id for item in result.packet.evidence] == [
        first.chunk_id,
        third.chunk_id,
    ]
    assert result.coverage.evidence_token_count == 5
    assert result.coverage.token_budget_omission_count == 1
    assert result.coverage.evidence_count <= config.final_top_k
    assert explain_search_coverage(config, result.search_run_id) == result.coverage
