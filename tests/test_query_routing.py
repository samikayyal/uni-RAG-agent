from __future__ import annotations

import json
from pathlib import Path

import pytest

from uni_rag_agent.config import load_config
from uni_rag_agent.retrieval import (
    RetrievalError,
    RoutingError,
    RouterOutput,
    retrieve,
    route_query,
)
from uni_rag_agent.storage import connect_sqlite, ensure_data_dirs, initialize_schema
from tests.sqlite_helpers import insert_minimal_chunk


ALL_INDEXES = (
    "document_index",
    "slides_index",
    "notebook_index",
    "code_index",
    "data_schema_index",
    "transcript_index",
)


def _llm_config(config):
    return config.__class__(
        **{
            **config.__dict__,
            "llm_provider": "openai",
            "llm_model": "test-router",
        }
    )


def _supported_route(
    *,
    query_type: str = "concept_explanation",
    courses: tuple[str, ...] = ("Information Retrieval",),
    indexes: tuple[str, ...] = ("document_index",),
) -> RouterOutput:
    return RouterOutput(
        query_type=query_type,
        candidate_courses=courses,
        candidate_indexes=indexes,
        keyword_terms=("MapReduce",),
        semantic_queries=("MapReduce explanation",),
        needs_keyword_search=True,
        needs_semantic_search=True,
        needs_file_inspection=False,
        needs_python=False,
        route_confidence=1.0,
        route_reason="test route",
    )


def _config_with_courses(tmp_path: Path):
    (tmp_path / "Courses").mkdir()
    config = load_config(repo_root=tmp_path, env_file=tmp_path / "missing.env")
    ensure_data_dirs(config)
    with connect_sqlite(config) as connection:
        initialize_schema(connection)
        insert_minimal_chunk(
            connection,
            config,
            course_name="Information Retrieval",
            filename="syllabus.txt",
            text="BM25 MapReduce retrieval syllabus",
        )
        insert_minimal_chunk(
            connection,
            config,
            course_name="High Preformance Computing for Big Data",
            filename="mapreduce.md",
            text="MapReduce distributed computation",
        )
        connection.commit()
    return config


@pytest.mark.parametrize(
    ("query, query_type, courses, indexes, needs_file_inspection, needs_python"),
    [
        (
            "Explain MapReduce from Information Retrieval",
            "concept_explanation",
            ("Information Retrieval",),
            ALL_INDEXES,
            False,
            False,
        ),
        (
            "Give a course summary for Information Retrieval",
            "course_summary",
            ("Information Retrieval",),
            ALL_INDEXES,
            False,
            False,
        ),
        (
            "Compare Information Retrieval vs High Preformance Computing for Big Data",
            "cross_course_comparison",
            (
                "High Preformance Computing for Big Data",
                "Information Retrieval",
            ),
            ALL_INDEXES,
            False,
            False,
        ),
        (
            "Find lecture.pdf in Information Retrieval",
            "find_file",
            ("Information Retrieval",),
            ("document_index",),
            True,
            False,
        ),
        (
            "Show my assignment on MapReduce in Information Retrieval",
            "assignment_or_project_lookup",
            ("Information Retrieval",),
            (
                "document_index",
                "slides_index",
                "notebook_index",
                "code_index",
                "data_schema_index",
            ),
            True,
            False,
        ),
        (
            "Show code implementation in Information Retrieval",
            "code_question",
            ("Information Retrieval",),
            ("code_index", "notebook_index"),
            True,
            False,
        ),
        (
            "Which dataset columns are in Information Retrieval",
            "data_question",
            ("Information Retrieval",),
            ("data_schema_index", "notebook_index", "code_index"),
            True,
            False,
        ),
        (
            "Create flashcards for Information Retrieval",
            "study_quiz",
            ("Information Retrieval",),
            ALL_INDEXES,
            False,
            False,
        ),
        (
            "Draft a resume bullet for Information Retrieval",
            "portfolio_resume",
            ("Information Retrieval",),
            ALL_INDEXES,
            False,
            False,
        ),
    ],
    ids=(
        "concept",
        "course-summary",
        "cross-course",
        "find-file",
        "assignment",
        "code",
        "data",
        "study",
        "portfolio",
    ),
)
def test_rule_routing_matrix_covers_every_supported_query_type(
    tmp_path: Path,
    query: str,
    query_type: str,
    courses: tuple[str, ...],
    indexes: tuple[str, ...],
    needs_file_inspection: bool,
    needs_python: bool,
) -> None:
    """Each supported type has an unambiguous, no-LLM routing regression."""
    output = route_query(_config_with_courses(tmp_path), query)

    assert output.query_type == query_type
    assert output.route_source == "rule"
    assert output.candidate_courses == courses
    assert output.candidate_indexes == indexes
    assert output.needs_keyword_search is True
    assert output.needs_semantic_search is True
    assert output.needs_file_inspection is needs_file_inspection
    assert output.needs_python is needs_python


@pytest.mark.parametrize(
    ("query, query_type, courses, indexes, needs_file_inspection"),
    [
        (
            "Explain the code implementation in Information Retrieval",
            "concept_explanation",
            ("Information Retrieval",),
            ALL_INDEXES,
            False,
        ),
        (
            "Find the course summary for Information Retrieval",
            "course_summary",
            ("Information Retrieval",),
            ALL_INDEXES,
            False,
        ),
        (
            "Compare the course summary for Information Retrieval and High Preformance Computing for Big Data",
            "cross_course_comparison",
            (
                "High Preformance Computing for Big Data",
                "Information Retrieval",
            ),
            ALL_INDEXES,
            False,
        ),
        (
            "Find the assignment for Information Retrieval",
            "find_file",
            ("Information Retrieval",),
            ALL_INDEXES,
            True,
        ),
        (
            "Show the assignment code for Information Retrieval",
            "assignment_or_project_lookup",
            ("Information Retrieval",),
            (
                "document_index",
                "slides_index",
                "notebook_index",
                "code_index",
                "data_schema_index",
            ),
            True,
        ),
        (
            "Inspect the code dataset schema in Information Retrieval",
            "code_question",
            ("Information Retrieval",),
            ("code_index", "notebook_index"),
            True,
        ),
        (
            "Inspect the dataset code in Information Retrieval",
            "data_question",
            ("Information Retrieval",),
            ("data_schema_index", "notebook_index", "code_index"),
            True,
        ),
        (
            "Create quiz questions to explain MapReduce in Information Retrieval",
            "study_quiz",
            ("Information Retrieval",),
            ALL_INDEXES,
            False,
        ),
        (
            "Find a portfolio project bullet for Information Retrieval",
            "portfolio_resume",
            ("Information Retrieval",),
            ALL_INDEXES,
            False,
        ),
    ],
    ids=(
        "concept-code",
        "summary-file",
        "comparison-summary",
        "file-assignment",
        "assignment-code",
        "code-data",
        "data-code",
        "study-concept",
        "portfolio-file",
    ),
)
def test_overlapping_routing_matrix_uses_configured_llm_fallback(
    tmp_path: Path,
    query: str,
    query_type: str,
    courses: tuple[str, ...],
    indexes: tuple[str, ...],
    needs_file_inspection: bool,
) -> None:
    """Conflicting cues must invoke the configured fallback, never guess a rule."""
    config = _llm_config(_config_with_courses(tmp_path))
    invocations: list[str] = []
    payload = {
        "query_type": query_type,
        "candidate_courses": list(courses),
        "candidate_indexes": list(indexes),
        "keyword_terms": ["MapReduce"],
        "semantic_queries": ["MapReduce explanation"],
        "needs_keyword_search": True,
        "needs_semantic_search": True,
        "needs_file_inspection": needs_file_inspection,
        "needs_python": False,
        "route_confidence": 0.91,
        "route_reason": "The test-only fallback resolved the conflicting cues.",
    }

    class FakeChat:
        def invoke(self, prompt: str) -> object:
            invocations.append(prompt)
            return type("Response", (), {"content": json.dumps(payload)})()

    output = route_query(config, query, chat_model=FakeChat())

    assert len(invocations) == 1
    assert output.query_type == query_type
    assert output.route_source == "llm"
    assert output.candidate_courses == courses
    assert output.candidate_indexes == indexes
    assert output.needs_keyword_search is True
    assert output.needs_semantic_search is True
    assert output.needs_file_inspection is needs_file_inspection
    assert output.needs_python is False


def test_rule_router_resolves_course_type_indexes_and_terms(tmp_path: Path) -> None:
    config = _config_with_courses(tmp_path)

    output = route_query(
        config,
        "  Explain MapReduce from Information Retrieval  ",
    )

    assert output.query_type == "concept_explanation"
    assert output.candidate_courses == ("Information Retrieval",)
    assert output.candidate_indexes == (
        "document_index",
        "slides_index",
        "notebook_index",
        "code_index",
        "data_schema_index",
        "transcript_index",
    )
    assert output.keyword_terms == ("MapReduce", "Information", "Retrieval")
    assert output.semantic_queries == ("Explain MapReduce from Information Retrieval",)
    assert output.route_source == "rule"
    assert output.needs_python is False


def test_alias_and_extension_routing_preserve_canonical_course_spelling(
    tmp_path: Path,
) -> None:
    config = _config_with_courses(tmp_path)

    hpc = route_query(config, "Find the HPC mapreduce.md")
    code = route_query(config, "Find assignment.py in Information Retrieval")

    assert hpc.candidate_courses == ("High Preformance Computing for Big Data",)
    assert hpc.query_type == "find_file"
    assert hpc.candidate_indexes == ("document_index",)
    assert code.candidate_indexes == ("code_index",)
    assert code.needs_file_inspection is True


def test_unresolved_course_without_llm_returns_unsupported(tmp_path: Path) -> None:
    config = _config_with_courses(tmp_path)

    output = route_query(config, "Explain MapReduce")

    assert output.route_source == "unsupported"
    assert output.query_type == "unknown_or_unsupported"
    assert output.candidate_courses == ()
    assert "LLM fallback unavailable" in output.route_reason


def test_rule_router_does_not_scan_conversation_context(tmp_path: Path) -> None:
    config = _config_with_courses(tmp_path)

    output = route_query(
        config,
        "Explain MapReduce",
        conversation_context=[
            {"role": "user", "content": "Information Retrieval"},
        ],
    )

    assert output.route_source == "unsupported"
    with pytest.raises(RoutingError, match="only role and content"):
        route_query(
            config,
            "Explain MapReduce",
            conversation_context=[
                {"role": "user", "content": "Information Retrieval", "secret": "x"},
            ],
        )


def test_configured_llm_fallback_accepts_and_canonicalizes_json(tmp_path: Path) -> None:
    config = _llm_config(_config_with_courses(tmp_path))

    payload = {
        "query_type": "concept_explanation",
        "candidate_courses": ["Information Retrieval"],
        "candidate_indexes": ["document_index", "notebook_index"],
        "keyword_terms": ["MapReduce"],
        "semantic_queries": ["MapReduce explanation", "MapReduce explanation"],
        "needs_keyword_search": True,
        "needs_semantic_search": True,
        "needs_file_inspection": False,
        "needs_python": False,
        "route_confidence": 0.91,
        "route_reason": "The fallback selected the course and indexes.",
    }

    class FakeChat:
        def invoke(self, prompt: str) -> object:
            assert "secret" not in prompt
            return type("Response", (), {"content": json.dumps(payload)})()

    output = route_query(config, "Explain MapReduce", chat_model=FakeChat())

    assert output.route_source == "llm"
    assert output.candidate_courses == ("Information Retrieval",)
    assert output.semantic_queries == ("MapReduce explanation",)


def test_invalid_llm_json_becomes_unsupported(tmp_path: Path) -> None:
    config = _llm_config(_config_with_courses(tmp_path))

    class FakeChat:
        def invoke(self, _prompt: str) -> object:
            return type("Response", (), {"content": "not json"})()

    output = route_query(config, "Explain MapReduce", chat_model=FakeChat())

    assert output.route_source == "unsupported"
    assert "Rejected LLM router output" in output.route_reason


@pytest.mark.parametrize(
    "invalid_field, invalid_value, reason",
    [
        ("query_type", "made_up_type", "unknown query type"),
        ("candidate_indexes", ["made_up_index"], "unknown logical index"),
        ("needs_keyword_search", "yes", "must be boolean"),
    ],
    ids=("query-type", "index", "flag"),
)
def test_invalid_llm_fields_become_unsupported(
    tmp_path: Path,
    invalid_field: str,
    invalid_value: object,
    reason: str,
) -> None:
    config = _llm_config(_config_with_courses(tmp_path))
    payload = {
        "query_type": "concept_explanation",
        "candidate_courses": ["Information Retrieval"],
        "candidate_indexes": ["document_index"],
        "keyword_terms": ["MapReduce"],
        "semantic_queries": ["MapReduce explanation"],
        "needs_keyword_search": True,
        "needs_semantic_search": True,
        "needs_file_inspection": False,
        "needs_python": False,
        "route_confidence": 0.91,
        "route_reason": "test route",
    }
    payload[invalid_field] = invalid_value

    class FakeChat:
        def invoke(self, _prompt: str) -> object:
            return type("Response", (), {"content": json.dumps(payload)})()

    output = route_query(config, "Explain MapReduce", chat_model=FakeChat())

    assert output.route_source == "unsupported"
    assert reason in output.route_reason


def test_low_confidence_llm_output_becomes_unsupported(tmp_path: Path) -> None:
    config = _llm_config(_config_with_courses(tmp_path))
    payload = {
        "query_type": "concept_explanation",
        "candidate_courses": ["Information Retrieval"],
        "candidate_indexes": ["document_index"],
        "keyword_terms": ["MapReduce"],
        "semantic_queries": ["MapReduce explanation"],
        "needs_keyword_search": True,
        "needs_semantic_search": True,
        "needs_file_inspection": False,
        "needs_python": False,
        "route_confidence": 0.01,
        "route_reason": "test route",
    }

    class FakeChat:
        def invoke(self, _prompt: str) -> object:
            return type("Response", (), {"content": json.dumps(payload)})()

    output = route_query(config, "Explain MapReduce", chat_model=FakeChat())

    assert output.route_source == "unsupported"
    assert "below router_min_confidence" in output.route_reason


def test_llm_invocation_failure_is_fatal(tmp_path: Path) -> None:
    config = _llm_config(_config_with_courses(tmp_path))

    class FailingChat:
        def invoke(self, _prompt: str) -> object:
            raise RuntimeError("provider unavailable")

    with pytest.raises(RoutingError, match="LLM router invocation failed"):
        route_query(config, "Explain MapReduce", chat_model=FailingChat())


@pytest.mark.parametrize("backend", ["metadata", "keyword", "semantic"])
def test_retrieval_backend_failures_are_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    config = _config_with_courses(tmp_path)
    route = _supported_route()

    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.metadata_search",
        (lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("metadata down")))
        if backend == "metadata"
        else lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.keyword_search_terms",
        (lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("keyword down")))
        if backend == "keyword"
        else lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.semantic_search",
        (lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("semantic down")))
        if backend == "semantic"
        else lambda *args, **kwargs: [],
    )

    with pytest.raises(RetrievalError, match="Retrieval backend failed"):
        retrieve(config, "Explain MapReduce", router_output=route, model="BAAI/bge-m3")


def test_zero_hit_retrieval_reports_each_search_weakness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_with_courses(tmp_path)
    route = _supported_route()
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.metadata_search", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.keyword_search_terms",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "uni_rag_agent.retrieval.core.semantic_search", lambda *args, **kwargs: []
    )

    run = retrieve(
        config, "Explain MapReduce", router_output=route, model="BAAI/bge-m3"
    )

    assert run.status == "completed"
    assert run.results == ()
    assert [item.result_set_id for item in run.result_sets] == [
        "metadata",
        "keyword",
        "semantic:1",
    ]
    assert run.weaknesses == (
        "No current metadata files matched the query.",
        "Keyword search returned no hits.",
        "Semantic query returned no hits: MapReduce explanation",
        "All retrieval result lists were empty.",
    )
