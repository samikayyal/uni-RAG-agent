from __future__ import annotations

import json
from pathlib import Path

import pytest

from uni_rag_agent.retrieval import QueryPlanningError, plan_query
from uni_rag_agent.storage import connect_sqlite
from tests.sqlite_helpers import insert_minimal_chunk
from tests.support import make_initialized_config


def _planning_config(tmp_path: Path, *, llm: bool = True):
    config = make_initialized_config(
        tmp_path,
        llm_provider="ollama" if llm else None,
        llm_model="test-planner" if llm else None,
    )
    with connect_sqlite(config) as connection:
        insert_minimal_chunk(
            connection,
            config,
            course_name="Information Retrieval",
            filename="notes.md",
            text="BM25 and query planning",
        )
        insert_minimal_chunk(
            connection,
            config,
            course_name="High Preformance Computing for Big Data",
            filename="mapreduce.md",
            text="MapReduce",
        )
        connection.commit()
    return config


def _supported_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "query_type": "concept_explanation",
        "candidate_courses": ["information retrieval"],
        "candidate_indexes": ["document_index", "slides_index"],
        "keyword_terms": ["BM25", "bm25"],
        "semantic_queries": ["Explain BM25", "explain bm25"],
        "needs_file_inspection": False,
        "needs_python": False,
        "plan_confidence": 0.91,
        "plan_reason": "The query asks for a course-grounded concept explanation.",
    }
    payload.update(changes)
    return payload


class FakeChat:
    def __init__(self, response: object) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> object:
        self.prompts.append(prompt)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_plan_query_validates_and_canonicalizes_supported_plan(tmp_path: Path) -> None:
    config = _planning_config(tmp_path)
    chat = FakeChat(json.dumps(_supported_payload()))

    plan = plan_query(config, "Explain BM25", chat_model=chat)

    assert plan.candidate_courses == ("Information Retrieval",)
    assert plan.candidate_indexes == ("document_index", "slides_index")
    assert plan.keyword_terms == ("BM25",)
    assert plan.semantic_queries == ("Explain BM25",)
    assert plan.plan_confidence == 0.91
    prompt = json.loads(chat.prompts[0])
    assert prompt["canonical_courses"] == [
        "High Preformance Computing for Big Data",
        "Information Retrieval",
    ]
    assert prompt["semantic_query_limit"] == config.semantic_query_limit


def test_plan_query_accepts_json_object_wrapped_in_json_code_fence(
    tmp_path: Path,
) -> None:
    config = _planning_config(tmp_path)
    chat = FakeChat(f"```json\n{json.dumps(_supported_payload())}\n```")

    plan = plan_query(config, "Explain BM25", chat_model=chat)

    assert plan.query_type == "concept_explanation"
    assert plan.candidate_courses == ("Information Retrieval",)


def test_plan_query_truncates_context_and_rejects_unexpected_fields(
    tmp_path: Path,
) -> None:
    config = _planning_config(tmp_path)
    chat = FakeChat(json.dumps(_supported_payload()))
    context = [{"role": "user", "content": f"message {index}"} for index in range(8)]

    plan_query(config, "Explain BM25", context, chat_model=chat)

    assert [
        item["content"] for item in json.loads(chat.prompts[0])["recent_conversation"]
    ] == [f"message {index}" for index in range(2, 8)]
    with pytest.raises(QueryPlanningError, match="only role and content"):
        plan_query(
            config,
            "Explain BM25",
            [{"role": "user", "content": "hello", "extra": "no"}],
            chat_model=chat,
        )


def test_plan_query_accepts_valid_unsupported_plan_without_search_scope(
    tmp_path: Path,
) -> None:
    config = _planning_config(tmp_path)
    chat = FakeChat(
        json.dumps(
            _supported_payload(
                query_type="unknown_or_unsupported",
                candidate_courses=[],
                candidate_indexes=[],
                keyword_terms=[],
                semantic_queries=[],
                plan_reason="This request is outside the course archive domain.",
            )
        )
    )

    plan = plan_query(config, "Write a poem", chat_model=chat)

    assert plan.query_type == "unknown_or_unsupported"
    assert plan.candidate_courses == ()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not json", "must be one JSON object"),
        ("[]", "exactly the QueryPlan fields"),
        (json.dumps(_supported_payload(extra="no")), "exactly the QueryPlan fields"),
        (json.dumps(_supported_payload(query_type="other")), "unknown query type"),
        (
            json.dumps(_supported_payload(candidate_courses=["Unknown"])),
            "unknown course",
        ),
        (
            json.dumps(_supported_payload(candidate_indexes=["other_index"])),
            "unknown logical index",
        ),
        (json.dumps(_supported_payload(keyword_terms=[""])), "must be nonblank"),
        (json.dumps(_supported_payload(needs_python="false")), "must be boolean"),
        (
            json.dumps(_supported_payload(plan_confidence=0.59)),
            "below query_plan_min_confidence",
        ),
        (
            json.dumps(_supported_payload(plan_confidence=1.1)),
            "must be between 0 and 1",
        ),
    ],
)
def test_plan_query_rejects_invalid_structured_output(
    tmp_path: Path, payload: str, message: str
) -> None:
    with pytest.raises(QueryPlanningError, match=message):
        plan_query(
            _planning_config(tmp_path),
            "Explain BM25",
            chat_model=FakeChat(payload),
        )


def test_plan_query_requires_configuration_and_propagates_provider_failure(
    tmp_path: Path,
) -> None:
    with pytest.raises(QueryPlanningError, match="UNI_RAG_LLM_PROVIDER"):
        plan_query(
            _planning_config(tmp_path, llm=False),
            "Explain BM25",
            chat_model=FakeChat(json.dumps(_supported_payload())),
        )

    with pytest.raises(QueryPlanningError, match="invocation failed"):
        plan_query(
            _planning_config(tmp_path / "provider-failure"),
            "Explain BM25",
            chat_model=FakeChat(RuntimeError("provider unavailable")),
        )
