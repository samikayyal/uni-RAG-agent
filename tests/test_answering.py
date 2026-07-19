from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support import make_config, make_initialized_config
from uni_rag_agent.answering import (
    AnswerCitation,
    AnswerGenerationError,
    AnswerParagraph,
    AnswerResult,
    AnswerSession,
    answer_body,
    answer_status,
    generate_answer,
    load_answer,
    store_answer,
    validate_answer_citations,
)
from uni_rag_agent.answering.audit import audit_stored_answer
from uni_rag_agent.retrieval.evidence_models import (
    ANSWER_CONSTRAINTS,
    EvidenceItem,
    EvidenceLocation,
    EvidencePacket,
    RetrievalSettings,
    SearchCoverage,
)
from uni_rag_agent.retrieval.models import QueryPlan
from uni_rag_agent.storage import connect_sqlite


def _packet(tmp_path: Path, *, evidence: bool = True, weaknesses=()) -> EvidencePacket:
    plan = QueryPlan(
        query_type="concept_explanation",
        candidate_courses=("Course A",),
        candidate_indexes=("document_index",),
        keyword_terms=("topic",),
        semantic_queries=("topic",),
        needs_file_inspection=False,
        needs_python=False,
        plan_confidence=0.9,
        plan_reason="fixture",
    )
    settings = RetrievalSettings(
        llm_provider="ollama",
        llm_model="planner",
        embedding_model="BAAI/bge-m3",
        keyword_top_k=20,
        semantic_top_k=20,
        metadata_top_k=20,
        semantic_query_limit=3,
        query_plan_min_confidence=0.6,
        filename_fuzzy_threshold=85,
        path_fuzzy_threshold=90,
        rrf_k=60,
        final_top_k=10,
        evidence_max_tokens=12000,
        conversation_context_message_count=0,
    )
    items = ()
    if evidence:
        items = (
            EvidenceItem(
                course="Course A",
                file_id=1,
                chunk_id=101,
                file=str(tmp_path / "Courses" / "Course A" / "notes.md"),
                source_type="document",
                location=EvidenceLocation("page", "2", "page 2"),
                text="The topic is grounded in this fixture.",
                token_count=7,
                rank=1,
                score=1.0,
                retrieval_method="hybrid",
                contributions=(),
            ),
        )
    coverage = SearchCoverage(
        search_run_id=1,
        status="completed",
        searched_courses=("Course A",),
        searched_indexes=("document_index",),
        keyword_terms=("topic",),
        semantic_queries=("topic",),
        raw_result_count=1 if evidence else 0,
        raw_result_counts_by_method={
            "metadata": 0,
            "keyword": 1 if evidence else 0,
            "semantic": 0,
        },
        fused_candidate_count=1 if evidence else 0,
        selectable_candidate_count=1 if evidence else 0,
        evidence_count=len(items),
        evidence_token_count=sum(item.token_count for item in items),
        courses_with_chunk_hits=("Course A",) if evidence else (),
        indexes_with_chunk_hits=("document_index",) if evidence else (),
        source_types_with_chunk_hits=("document",) if evidence else (),
        courses_without_chunk_hits=() if evidence else ("Course A",),
        indexes_without_chunk_hits=() if evidence else ("document_index",),
        semantic_queries_without_hits=() if evidence else ("topic",),
        missing_capabilities=(),
        file_only_candidate_count=0,
        token_budget_omission_count=0,
        oversized_evidence_omission_count=0,
        unselected_selectable_candidate_count=0,
        weaknesses=tuple(weaknesses),
    )
    return EvidencePacket(
        search_run_id=1,
        query="Explain topic",
        interpreted_intent="concept_explanation",
        query_plan=plan,
        retrieval_settings=settings,
        searched={
            "courses": ("Course A",),
            "indexes": ("document_index",),
            "keyword_terms": ("topic",),
            "semantic_queries": ("topic",),
        },
        coverage=coverage,
        evidence=items,
        weaknesses=tuple(weaknesses),
        answer_constraints=ANSWER_CONSTRAINTS,
    )


class _Chat:
    def __init__(self, *payloads: object, error: Exception | None = None) -> None:
        self.payloads = list(payloads)
        self.prompts: list[str] = []
        self.error = error
        self.model_name = "injected-answer"

    def invoke(self, prompt: str) -> object:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        payload = self.payloads.pop(0)
        return SimpleNamespace(content=json.dumps(payload))


def _valid_payload(*, citation_ids=("E1",), text="Grounded answer", limitations=()):
    return {
        "answer_paragraphs": [{"text": text, "citation_ids": list(citation_ids)}],
        "limitations": list(limitations),
    }


def _persist_packet(config, packet: EvidencePacket) -> int:
    with connect_sqlite(config) as connection:
        search_run_id = connection.execute(
            """INSERT INTO search_runs (query, query_type, query_plan_json,
               retrieval_settings_json, searched_courses_json, searched_indexes_json,
               keyword_terms_json, semantic_queries_json, started_at, finished_at,
               status, weaknesses_json, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                packet.query,
                packet.interpreted_intent,
                json.dumps(packet.query_plan.as_safe_dict()),
                json.dumps(packet.retrieval_settings.as_safe_dict()),
                json.dumps(packet.searched["courses"]),
                json.dumps(packet.searched["indexes"]),
                json.dumps(packet.searched["keyword_terms"]),
                json.dumps(packet.searched["semantic_queries"]),
                "now",
                "now",
                packet.coverage.status,
                json.dumps(packet.weaknesses),
                None,
            ),
        ).lastrowid
        assert int(search_run_id) == packet.search_run_id
        packet_id = connection.execute(
            "INSERT INTO evidence_packets (search_run_id, packet_json, evidence_count, created_at) VALUES (?, ?, ?, ?)",
            (
                search_run_id,
                json.dumps(packet.as_safe_dict()),
                len(packet.evidence),
                "now",
            ),
        ).lastrowid
        connection.commit()
    return int(packet_id)


def test_generate_answer_renders_stable_citation_and_structured_reference(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path, weaknesses=("weak packet",))
    chat = _Chat(_valid_payload())
    result = generate_answer(
        packet,
        conversation_context=[{"role": "user", "content": "secret context"}],
        chat_model=chat,
    )

    assert "Grounded answer [E1]" in result.answer_text
    assert "- [E1] Course A - " in result.answer_text
    assert "page 2" in result.answer_text
    assert result.citations[0].evidence_index == 1
    assert result.limitations == ("weak packet",)
    assert "secret context" not in chat.prompts[0]
    assert validate_answer_citations(result, packet)


def test_chunk_alias_is_accepted_and_canonicalized(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    result = generate_answer(
        packet,
        chat_model=_Chat(_valid_payload(citation_ids=("chunk:101",))),
    )

    assert result.paragraphs[0].citation_ids == ("E1",)
    assert result.citations[0].citation_id == "E1"
    assert "Grounded answer [E1]" in result.answer_text
    assert "chunk:101" not in result.answer_text


def test_ambiguous_or_unknown_citation_ids_are_rejected(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    for unknown_id in ("E101", "1", "101", "E999", "chunk:999"):
        invalid = generate_answer(
            packet,
            chat_model=_Chat(_valid_payload(citation_ids=(unknown_id,))),
            config=replace(make_config(tmp_path), answer_max_retries=0),
        )
        assert not invalid.citations
        assert "validation failed" in invalid.limitations[-1].lower()


@pytest.mark.parametrize(
    "model_text",
    (
        "Claim [E0]",
        "Claim [E01]",
        "Claim [e1]",
        "Claim [ E 1 ]",
        "Claim [1]",
        "Claim [Eabc]",
        "Claim [source](https://example.test)",
        "References: invented source",
        "Limitations: invented limitation",
        "## References:\n- invented source",
        "**References:**\n- invented source",
        "> References:\n> invented source",
    ),
)
def test_model_authored_markers_and_sections_are_rejected(
    tmp_path: Path,
    model_text: str,
) -> None:
    packet = _packet(tmp_path)
    config = replace(make_config(tmp_path), answer_max_retries=0)
    result = generate_answer(
        packet,
        chat_model=_Chat(_valid_payload(text=model_text)),
        config=config,
    )
    assert not result.citations
    assert "validation failed" in result.limitations[-1].lower()


def test_loaded_answer_can_be_revalidated_from_rendered_markers(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    generated = generate_answer(packet, chat_model=_Chat(_valid_payload()))
    reconstructed = replace(generated, paragraphs=())
    validation = validate_answer_citations(reconstructed, packet)
    assert validation.valid
    assert validation.citations[0].citation_id == "E1"


def test_answer_body_and_status_share_the_answering_render_contract(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    answered = generate_answer(packet, chat_model=_Chat(_valid_payload()))
    refused = generate_answer(
        packet,
        chat_model=_Chat(_valid_payload(citation_ids=())),
        config=replace(make_config(tmp_path), answer_max_retries=0),
    )

    assert answer_body(answered.answer_text) == "Grounded answer [E1]"
    assert answer_status(answered) == "answered"
    assert answer_body(refused.answer_text).startswith("I cannot provide")
    assert answer_status(refused) == "validation_failed"


def test_retry_count_and_zero_retry(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    invalid = {
        "answer_paragraphs": [{"text": "missing citation", "citation_ids": []}],
        "limitations": [],
    }
    chat = _Chat(invalid, _valid_payload())
    result = generate_answer(
        packet,
        chat_model=chat,
        config=replace(
            make_config(tmp_path),
            answer_max_retries=1,
            answer_prompt_max_tokens=180,
        ),
    )
    assert len(chat.prompts) == 2
    assert all(len(prompt.split()) <= 180 for prompt in chat.prompts)
    assert [
        [item["citation_id"] for item in json.loads(prompt)["evidence"]]
        for prompt in chat.prompts
    ] == [["E1"], ["E1"]]
    assert result.citations
    zero_chat = _Chat(invalid, _valid_payload())
    result = generate_answer(
        packet,
        chat_model=zero_chat,
        config=replace(make_config(tmp_path), answer_max_retries=0),
    )
    assert len(zero_chat.prompts) == 1
    assert not result.citations


def test_validation_rejections_log_only_bounded_diagnostics(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    packet = _packet(tmp_path)
    invalid = {
        "answer_paragraphs": [{"text": "private raw model prose", "citation_ids": []}],
        "limitations": [],
    }

    with caplog.at_level("WARNING", logger="uni_rag_agent.answering.core"):
        generate_answer(
            packet,
            chat_model=_Chat(invalid),
            config=replace(make_config(tmp_path), answer_max_retries=0),
        )

    assert "paragraph 1 must cite at least one evidence item" in caplog.text
    assert "private raw model prose" not in caplog.text


def test_prompt_budget_omits_complete_items_and_preserves_packet_ids(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    first = replace(packet.evidence[0], text="large " * 400, token_count=400)
    second = replace(
        packet.evidence[0],
        chunk_id=202,
        text="Small supporting evidence.",
        token_count=3,
        rank=2,
    )
    packet = replace(
        packet,
        evidence=(first, second),
        coverage=replace(
            packet.coverage,
            fused_candidate_count=2,
            selectable_candidate_count=2,
            evidence_count=2,
            evidence_token_count=403,
        ),
    )
    config = replace(
        make_config(tmp_path),
        answer_prompt_max_tokens=180,
        answer_max_retries=0,
    )
    chat = _Chat(_valid_payload(citation_ids=("E2",)))

    result = generate_answer(packet, chat_model=chat, config=config)

    prompt = json.loads(chat.prompts[0])
    assert [item["citation_id"] for item in prompt["evidence"]] == ["E2"]
    assert len(chat.prompts[0].split()) <= config.answer_prompt_max_tokens
    assert result.citations[0].citation_id == "E2"
    assert "[E2]" in result.answer_text
    assert "1 evidence item(s) were omitted" in result.limitations[-1]


def test_prompt_budget_accounts_for_overhead_and_can_skip_model(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    chat = _Chat(_valid_payload(), error=RuntimeError("must not call"))
    config = replace(make_config(tmp_path), answer_prompt_max_tokens=80)

    result = generate_answer(packet, chat_model=chat, config=config)

    assert not chat.prompts
    assert not result.citations
    assert "prompt budget" in result.answer_text
    assert "No answer model was invoked" in result.limitations[-1]


def test_packet_near_evidence_budget_fits_default_answer_prompt_budget(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    item = replace(
        packet.evidence[0],
        text="word " * 12_000,
        token_count=12_000,
    )
    packet = replace(
        packet,
        evidence=(item,),
        coverage=replace(packet.coverage, evidence_token_count=12_000),
    )
    config = replace(make_config(tmp_path), answer_max_retries=0)
    chat = _Chat(_valid_payload())

    result = generate_answer(packet, chat_model=chat, config=config)

    assert result.citations
    assert len(chat.prompts[0].split()) <= config.answer_prompt_max_tokens


def test_empty_evidence_never_invokes_model(tmp_path: Path) -> None:
    packet = _packet(tmp_path, evidence=False, weaknesses=("no chunks",))
    chat = _Chat(_valid_payload(), error=RuntimeError("must not call"))
    result = generate_answer(packet, chat_model=chat)
    assert not chat.prompts
    assert not result.citations
    assert "Insufficient evidence" in result.answer_text
    assert "no chunks" in result.limitations


def test_provider_failure_does_not_look_like_a_safe_refusal(tmp_path: Path) -> None:
    with pytest.raises(AnswerGenerationError):
        generate_answer(_packet(tmp_path), chat_model=_Chat(error=RuntimeError("down")))


def test_store_answer_is_append_only_and_round_trips(tmp_path: Path) -> None:
    config = replace(
        make_initialized_config(tmp_path),
        answer_llm_provider="ollama",
        answer_llm_model="test-answer",
    )
    packet = _packet(tmp_path)
    packet_id = _persist_packet(config, packet)
    answer = generate_answer(
        packet,
        chat_model=_Chat(_valid_payload()),
        config=config,
    )
    first = store_answer(packet_id, answer, config=config)
    second = store_answer(config, packet_id, answer)
    assert second > first
    loaded = load_answer(config, first)
    assert loaded.answer_text == answer.answer_text
    assert loaded.model_name == "ollama:test-answer"
    with sqlite3.connect(config.sqlite_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM answers").fetchone()[0] == 2


def test_store_answer_rolls_back_when_commit_guard_cancels(tmp_path: Path) -> None:
    config = replace(
        make_initialized_config(tmp_path),
        answer_llm_provider="ollama",
        answer_llm_model="test-answer",
    )
    packet = _packet(tmp_path)
    packet_id = _persist_packet(config, packet)
    answer = generate_answer(packet, chat_model=_Chat(_valid_payload()), config=config)

    def cancel_commit(commit) -> None:
        del commit
        raise RuntimeError("cancelled before commit")

    with pytest.raises(RuntimeError, match="cancelled before commit"):
        store_answer(
            packet_id,
            answer,
            config=config,
            commit_guard=cancel_commit,
        )

    with sqlite3.connect(config.sqlite_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM answers").fetchone()[0] == 0


def test_store_answer_rejects_packet_mismatches_and_forged_refusal(
    tmp_path: Path,
) -> None:
    config = replace(
        make_initialized_config(tmp_path),
        answer_llm_provider="ollama",
        answer_llm_model="test-answer",
    )
    packet = _packet(tmp_path, weaknesses=("weak packet",))
    packet_id = _persist_packet(config, packet)
    answer = generate_answer(
        packet,
        chat_model=_Chat(_valid_payload()),
        config=config,
    )

    wrong_chunk = replace(
        answer,
        citations=(replace(answer.citations[0], chunk_id=999),),
    )
    with pytest.raises(ValueError, match="packet-authoritative"):
        store_answer(packet_id, wrong_chunk, config=config)

    missing_weakness = replace(answer, limitations=())
    with pytest.raises(ValueError, match="omit packet weaknesses"):
        store_answer(packet_id, missing_weakness, config=config)

    forged_refusal = AnswerResult(
        answer_text="I refuse.",
        limitations=("Answer validation failed after 2 attempt(s).",),
        model_name="ollama:test-answer",
        paragraphs=(AnswerParagraph("I refuse.", ()),),
    )
    with pytest.raises(ValueError, match="deterministic packet-derived"):
        store_answer(packet_id, forged_refusal, config=config)

    forged_section = replace(
        answer,
        paragraphs=(AnswerParagraph("## References:\n- invented source", ("E1",)),),
    )
    with pytest.raises(ValueError, match="rendered section"):
        store_answer(packet_id, forged_section, config=config)

    with sqlite3.connect(config.sqlite_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM answers").fetchone()[0] == 0


def test_empty_evidence_answer_persists_without_answer_provider(
    tmp_path: Path,
) -> None:
    config = make_initialized_config(tmp_path)
    packet = _packet(tmp_path, evidence=False, weaknesses=("no chunks",))
    packet_id = _persist_packet(config, packet)
    answer = generate_answer(packet, chat_model=_Chat(error=RuntimeError("unused")))

    answer_id = store_answer(packet_id, answer, config=config)

    loaded = load_answer(config, answer_id)
    assert loaded.model_name is None
    assert not loaded.citations
    assert "No answer model was invoked" in loaded.limitations[-1]


def test_prompt_budget_insufficient_answer_persists_without_provider(
    tmp_path: Path,
) -> None:
    config = replace(
        make_initialized_config(tmp_path),
        answer_prompt_max_tokens=80,
    )
    packet = _packet(tmp_path)
    packet_id = _persist_packet(config, packet)
    answer = generate_answer(
        packet, config=config, chat_model=_Chat(error=RuntimeError())
    )

    answer_id = store_answer(packet_id, answer, config=config)

    loaded = load_answer(config, answer_id)
    assert loaded.model_name is None
    assert not loaded.citations
    assert "prompt budget" in loaded.answer_text


def test_answer_session_bounds_complete_turns_and_planner_only_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(make_config(tmp_path), answer_session_message_limit=4)
    packet = _packet(tmp_path)
    evidence_result = SimpleNamespace(
        packet=packet, evidence_packet_id=1, search_run_id=1
    )
    planner_contexts: list[tuple[dict[str, str], ...]] = []

    def fake_build(*args, **kwargs):
        planner_contexts.append(tuple(kwargs["conversation_context"]))
        return evidence_result

    answer = AnswerResult(answer_text="ok", paragraphs=(AnswerParagraph("ok", ()),))
    monkeypatch.setattr("uni_rag_agent.answering.session.build_evidence", fake_build)
    monkeypatch.setattr(
        "uni_rag_agent.answering.session.generate_answer",
        lambda *args, **kwargs: answer,
    )
    monkeypatch.setattr(
        "uni_rag_agent.answering.session.store_answer", lambda *args, **kwargs: 1
    )
    session = AnswerSession(config)
    session.ask("one")
    session.ask("two")
    session.ask("three")
    assert [len(context) for context in planner_contexts] == [0, 2, 4]
    assert len(session.conversation_context) == 4
    assert session.conversation_context[0]["content"] == "two"


def test_answer_session_provider_failure_does_not_append_partial_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(make_config(tmp_path), answer_session_message_limit=4)
    evidence_result = SimpleNamespace(
        packet=_packet(tmp_path),
        evidence_packet_id=1,
        search_run_id=1,
    )
    monkeypatch.setattr(
        "uni_rag_agent.answering.session.build_evidence",
        lambda *args, **kwargs: evidence_result,
    )
    monkeypatch.setattr(
        "uni_rag_agent.answering.session.generate_answer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AnswerGenerationError("provider down")
        ),
    )
    session = AnswerSession(config)

    with pytest.raises(AnswerGenerationError, match="provider down"):
        session.ask("failed turn")

    assert session.conversation_context == ()


def test_answer_session_can_record_an_already_persisted_complete_turn(
    tmp_path: Path,
) -> None:
    config = replace(make_config(tmp_path), answer_session_message_limit=2)
    session = AnswerSession(config)

    session.record_complete_turn("first query", "first answer")
    session.record_complete_turn("second query", "second answer")

    assert session.conversation_context == (
        {"role": "user", "content": "second query"},
        {"role": "assistant", "content": "second answer"},
    )


def test_answering_notebook_is_valid_and_read_only() -> None:
    notebook = json.loads(
        Path("notebooks/answering_eda.ipynb").read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "import pandas as pd" in source
    assert "mode=ro" in source
    assert "PRAGMA query_only" in source
    assert "audit_stored_answer" in source
    assert all(
        cell.get("execution_count") is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert all(
        not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_answering_audit_rejects_malformed_or_altered_citations(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    answer = generate_answer(packet, chat_model=_Chat(_valid_payload()))
    packet_json = json.dumps(packet.as_safe_dict())
    citations = [answer.citations[0].as_safe_dict()]

    malformed = audit_stored_answer("{", packet_json, answer.answer_text)
    assert not malformed["valid"]
    assert not malformed["citations_parsed"]

    for field, value in (
        ("chunk_id", 999),
        ("file_path", "forged.md"),
        ("location_label", "page 999"),
    ):
        altered = [dict(citations[0], **{field: value})]
        audit = audit_stored_answer(
            json.dumps(altered),
            packet_json,
            answer.answer_text,
        )
        assert not audit["valid"]
        assert "does not match packet evidence" in audit["diagnostic"]


def test_answering_audit_accepts_canonical_and_legitimate_empty_rows(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    cited = generate_answer(packet, chat_model=_Chat(_valid_payload()))
    cited_audit = audit_stored_answer(
        json.dumps([citation.as_safe_dict() for citation in cited.citations]),
        json.dumps(packet.as_safe_dict()),
        cited.answer_text,
    )
    assert cited_audit["valid"]

    empty_packet = _packet(tmp_path, evidence=False, weaknesses=("no chunks",))
    insufficient = generate_answer(empty_packet)
    empty_audit = audit_stored_answer(
        "[]",
        json.dumps(empty_packet.as_safe_dict()),
        insufficient.answer_text,
    )
    assert empty_audit["valid"]
