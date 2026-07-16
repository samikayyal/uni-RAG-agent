from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.support import make_config
from uni_rag_agent.answering import AnswerCitation, AnswerParagraph, AnswerResult
from uni_rag_agent.evaluation import (
    EvalItem,
    EvalResult,
    EvalSetError,
    EvaluationError,
    default_eval_set_path,
    load_eval_set,
    score_citations,
    score_retrieval,
    validate_fixture_state,
    write_eval_report,
)
from uni_rag_agent.evaluation import core as evaluation_core
from uni_rag_agent.retrieval.evidence_models import (
    ANSWER_CONSTRAINTS,
    EvidenceBuildResult,
    EvidenceItem,
    EvidenceLocation,
    EvidencePacket,
    RetrievalSettings,
    SearchCoverage,
)
from uni_rag_agent.retrieval.models import QueryPlan
from uni_rag_agent.retrieval.evidence_persistence import sanitize_error
from uni_rag_agent.retrieval import EvidenceError, RetrievalError


def _item(**overrides: object) -> EvalItem:
    values: dict[str, object] = {
        "id": "fixture",
        "query": "Explain reciprocal rank fusion",
        "query_type": "concept_explanation",
        "expected_courses": ("Information Retrieval",),
        "expected_files": ("Information Retrieval/hybrid_retrieval.md",),
        "expected_indexes": ("document_index",),
        "must_include_terms": ("reciprocal rank fusion",),
        "expected_weaknesses": (),
        "notes": "fixture",
    }
    values.update(overrides)
    return EvalItem(**values)


def _packet(
    *, evidence: bool = True, weaknesses: tuple[str, ...] = ()
) -> EvidencePacket:
    plan = QueryPlan(
        query_type="concept_explanation",
        candidate_courses=("Information Retrieval",),
        candidate_indexes=("document_index",),
        keyword_terms=("reciprocal rank fusion",),
        semantic_queries=("reciprocal rank fusion",),
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
    items: tuple[EvidenceItem, ...] = ()
    if evidence:
        items = (
            EvidenceItem(
                course="Information Retrieval",
                file_id=1,
                chunk_id=1,
                file="C:/fixture/Information Retrieval/hybrid_retrieval.md",
                source_type="document",
                location=EvidenceLocation("text_section", "1", "text section 1"),
                text="Reciprocal Rank Fusion combines lexical and semantic search.",
                token_count=8,
                rank=1,
                score=1.0,
                retrieval_method="hybrid",
                contributions=(),
            ),
        )
    coverage = SearchCoverage(
        search_run_id=1,
        status="completed",
        searched_courses=("Information Retrieval",),
        searched_indexes=("document_index",),
        keyword_terms=("reciprocal rank fusion",),
        semantic_queries=("reciprocal rank fusion",),
        raw_result_count=len(items),
        raw_result_counts_by_method={
            "metadata": 0,
            "keyword": len(items),
            "semantic": 0,
        },
        fused_candidate_count=len(items),
        selectable_candidate_count=len(items),
        evidence_count=len(items),
        evidence_token_count=sum(item.token_count for item in items),
        courses_with_chunk_hits=("Information Retrieval",) if items else (),
        indexes_with_chunk_hits=("document_index",) if items else (),
        source_types_with_chunk_hits=("document",) if items else (),
        courses_without_chunk_hits=() if items else ("Information Retrieval",),
        indexes_without_chunk_hits=() if items else ("document_index",),
        semantic_queries_without_hits=() if items else ("reciprocal rank fusion",),
        missing_capabilities=(),
        file_only_candidate_count=0,
        token_budget_omission_count=0,
        oversized_evidence_omission_count=0,
        unselected_selectable_candidate_count=0,
        weaknesses=weaknesses,
    )
    return EvidencePacket(
        search_run_id=1,
        query="Explain reciprocal rank fusion",
        interpreted_intent="concept_explanation",
        query_plan=plan,
        retrieval_settings=settings,
        searched={
            "courses": ("Information Retrieval",),
            "indexes": ("document_index",),
            "keyword_terms": ("reciprocal rank fusion",),
            "semantic_queries": ("reciprocal rank fusion",),
        },
        coverage=coverage,
        evidence=items,
        weaknesses=weaknesses,
        answer_constraints=ANSWER_CONSTRAINTS,
    )


def _manifest_snapshot() -> dict[str, object]:
    return {
        "files": 1,
        "chunks": 1,
        "keyword_rows": 1,
        "vector_rows": 1,
        "files_identity": "a" * 64,
        "chunks_identity": "b" * 64,
        "keyword_identity": "c" * 64,
        "vector_identity": "d" * 64,
        "vector_collections": [
            {
                "vector_backend": "chroma",
                "vector_collection": "documents__fixture",
                "embedding_model": "BAAI/bge-m3",
                "embedding_dim": 3,
                "row_count": 1,
            }
        ],
        "chroma_digest": "e" * 64,
    }


def _manifest_payload(config: object, snapshot: dict[str, object]) -> dict[str, object]:
    assert hasattr(config, "embedding_model")
    return {
        "manifest_version": 1,
        "fixture_digest": evaluation_core._sha256_file(default_eval_set_path()),
        "source_digest": evaluation_core._sha256_tree(
            evaluation_core.fixture_source_root()
        ),
        "embedding_model": config.embedding_model,
        **snapshot,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_fake_manifest_state(
    config: object,
    payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    snapshot: dict[str, object],
) -> Path:
    state = evaluation_core.fixture_state_config(config)  # type: ignore[arg-type]
    state.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    state.sqlite_path.touch()
    state.chroma_dir.mkdir(parents=True, exist_ok=True)
    (state.chroma_dir / "profile.bin").write_bytes(b"fixture")
    (state.data_dir / "manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluation_core, "_fixture_state_snapshot", lambda _: snapshot)
    return state.data_dir


def test_committed_fixture_set_is_strict_and_covers_contract() -> None:
    items = load_eval_set(default_eval_set_path())
    assert 15 <= len(items) <= 20
    assert {item.query_type for item in items} == {
        "concept_explanation",
        "course_summary",
        "cross_course_comparison",
        "find_file",
        "assignment_or_project_lookup",
        "code_question",
        "data_question",
        "study_quiz",
        "portfolio_resume",
        "unknown_or_unsupported",
    }
    assert {index for item in items for index in item.expected_indexes} == {
        "document_index",
        "slides_index",
        "notebook_index",
        "code_index",
        "data_schema_index",
        "transcript_index",
    }


def test_evaluation_source_digest_ignores_ipynb_checkpoint_files(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    (source_root / "kept.md").write_text("kept", encoding="utf-8")
    checkpoint_dir = source_root / ".ipynb_checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "kept-checkpoint.ipynb"
    checkpoint.write_text("first", encoding="utf-8")

    first_digest = evaluation_core._sha256_tree(source_root)
    checkpoint.write_text("second", encoding="utf-8")

    assert evaluation_core._sha256_tree(source_root) == first_digest


def test_eval_set_rejects_unknown_fields_and_non_utf8(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    value = json.loads(default_eval_set_path().read_text(encoding="utf-8"))[0]
    value["unknown"] = True
    path.write_text(json.dumps([value]), encoding="utf-8")
    with pytest.raises(EvalSetError, match="unknown"):
        load_eval_set(path)
    value.pop("unknown")
    value["expected_files"] = ["Information Retrieval/.ipynb_checkpoints/notes.ipynb"]
    path.write_text(json.dumps([value]), encoding="utf-8")
    with pytest.raises(EvalSetError, match="checkpoint"):
        load_eval_set(path)
    path.write_bytes(b"\xff")
    with pytest.raises(EvalSetError, match="UTF-8"):
        load_eval_set(path)


def test_retrieval_scoring_requires_exact_sources_and_terms() -> None:
    packet = _packet()
    score = score_retrieval(_item(), packet, courses_root=Path("C:/fixture"))
    assert score.passed
    failed = score_retrieval(
        _item(expected_files=("Information Retrieval/missing.md",)),
        packet,
        courses_root=Path("C:/fixture"),
    )
    assert not failed.passed
    assert failed.missing_files == ("Information Retrieval/missing.md",)


def test_explicit_absence_requires_zero_evidence_and_weakness() -> None:
    packet = _packet(evidence=False, weaknesses=("Unsupported question",))
    item = _item(
        expected_courses=(),
        expected_files=(),
        expected_indexes=(),
        must_include_terms=(),
        expected_weaknesses=("unsupported",),
    )
    score = score_retrieval(item, packet)
    assert score.passed
    assert score.absence_expected


def test_citation_scoring_is_packet_relative_and_checks_terms_and_limitations() -> None:
    packet = _packet(weaknesses=("weak retrieval",))
    item = _item(expected_weaknesses=("weak retrieval",))
    evidence = packet.evidence[0]
    citation = AnswerCitation.from_evidence(1, evidence)
    answer = AnswerResult(
        answer_text=(
            "Reciprocal Rank Fusion is grounded here [E1]\n\n"
            "References:\n"
            "- [E1] Information Retrieval - C:/fixture/Information Retrieval/hybrid_retrieval.md - text section 1\n\n"
            "Limitations:\n"
            "- weak retrieval"
        ),
        citations=(citation,),
        limitations=("weak retrieval",),
        model_name="ollama:model",
        paragraphs=(
            AnswerParagraph("Reciprocal Rank Fusion is grounded here", ("E1",)),
        ),
    )
    score = score_citations(packet, answer, item)
    assert score.passed
    assert score.valid


def test_report_writes_paired_safe_artifacts(tmp_path: Path) -> None:
    result = score_retrieval(_item(), _packet())
    citation = score_citations(
        _packet(),
        AnswerResult("Insufficient evidence", limitations=("weak",)),
    )
    from uni_rag_agent.evaluation import EvalResult

    report = write_eval_report(
        [
            EvalResult(
                item_id="fixture",
                query="secret query is not model output",
                query_type="concept_explanation",
                status="failed",
                retrieval=result,
                citations=citation,
                timings_ms={"evidence_ms": 4.0, "answer_ms": 2.0, "total_ms": 6.0},
                failures=("api_key=secret-value",),
            )
        ],
        tmp_path,
    )
    markdown = report.with_suffix(".md")
    assert report.is_file() and markdown.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["summary"]["timings_ms"]["total_ms"]["p50"] == 6.0
    assert "Reciprocal Rank Fusion combines" not in report.read_text(encoding="utf-8")
    assert "secret query" not in report.read_text(encoding="utf-8")
    assert "secret-value" not in report.read_text(encoding="utf-8")
    assert "raw evidence" in markdown.read_text(encoding="utf-8")


def test_fixture_state_missing_has_setup_guidance(tmp_path: Path) -> None:
    config = make_config(tmp_path, embedding_model="BAAI/bge-m3")
    with pytest.raises(Exception, match="prepare-fixtures"):
        validate_fixture_state(config)


def test_fixture_prepare_and_validate_canonicalizes_gemini_alias_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias = "gemini-embedding-001"
    canonical = "google/gemini-embedding-001"
    config = make_config(tmp_path, embedding_model=alias)
    snapshot = _manifest_snapshot()
    snapshot["vector_collections"] = [
        {
            **snapshot["vector_collections"][0],  # type: ignore[index]
            "embedding_model": canonical,
            "embedding_dim": 3072,
        }
    ]
    vector_calls: list[tuple[str | None, str | None]] = []

    def fake_ensure_data_dirs(state_config: object) -> None:
        state_config.data_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
        state_config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
        state_config.sqlite_path.touch()  # type: ignore[attr-defined]
        state_config.chroma_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]

    monkeypatch.setattr(evaluation_core, "ensure_data_dirs", fake_ensure_data_dirs)
    monkeypatch.setattr(evaluation_core, "inventory_courses", lambda _config: None)
    monkeypatch.setattr(evaluation_core, "extract_pending_files", lambda _config: None)
    monkeypatch.setattr(evaluation_core, "summarize_data_files", lambda _config: None)
    monkeypatch.setattr(
        evaluation_core,
        "sync_keyword_index",
        lambda _config, **_kwargs: None,
    )
    monkeypatch.setattr(
        evaluation_core,
        "sync_vector_index",
        lambda state_config, **kwargs: vector_calls.append(
            (state_config.embedding_model, kwargs.get("model"))
        ),
    )
    monkeypatch.setattr(
        evaluation_core,
        "_fixture_state_snapshot",
        lambda _state: snapshot,
    )

    manifest = evaluation_core.prepare_fixture_state(config)

    assert manifest["embedding_model"] == canonical
    assert vector_calls == [(canonical, canonical)]
    assert validate_fixture_state(config)["embedding_model"] == canonical
    assert (
        validate_fixture_state(replace(config, embedding_model=canonical))[
            "embedding_model"
        ]
        == canonical
    )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("manifest_version", "1"),
        ("fixture_digest", 1),
        ("source_digest", 1),
        ("embedding_model", 1),
        ("files", "1"),
        ("chunks", "1"),
        ("keyword_rows", "1"),
        ("vector_rows", "1"),
        ("files_identity", 1),
        ("chunks_identity", 1),
        ("keyword_identity", 1),
        ("vector_identity", 1),
        ("vector_collections", [{"vector_backend": "chroma"}]),
        ("prepared_at", 123),
    ),
)
def test_manifest_scalar_and_profile_types_are_strict(
    tmp_path: Path,
    field: str,
    wrong_value: object,
) -> None:
    snapshot = _manifest_snapshot()
    payload = _manifest_payload(
        make_config(tmp_path, embedding_model="BAAI/bge-m3"),
        snapshot,
    )
    payload[field] = wrong_value
    assert not evaluation_core._valid_manifest_scalars(payload)


def test_manifest_rejects_unsupported_version_and_same_count_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path, embedding_model="BAAI/bge-m3")
    snapshot = _manifest_snapshot()
    payload = _manifest_payload(config, snapshot)
    _write_fake_manifest_state(config, payload, monkeypatch, snapshot)
    assert validate_fixture_state(config)["manifest_version"] == 1

    unsupported = dict(payload)
    unsupported["manifest_version"] = 2
    state = evaluation_core.fixture_state_config(config)
    (state.data_dir / "manifest.json").write_text(
        json.dumps(unsupported),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="invalid identity"):
        validate_fixture_state(config)


@pytest.mark.parametrize(
    "drift_field",
    ("files_identity", "vector_identity", "chroma_digest", "vector_collections"),
)
def test_manifest_rejects_same_count_identity_and_collection_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_field: str,
) -> None:
    config = make_config(tmp_path, embedding_model="BAAI/bge-m3")
    snapshot = _manifest_snapshot()
    payload = _manifest_payload(config, snapshot)
    _write_fake_manifest_state(config, payload, monkeypatch, snapshot)
    assert validate_fixture_state(config)["files"] == 1

    drifted = dict(snapshot)
    if drift_field == "vector_collections":
        drifted[drift_field] = [
            {
                **snapshot["vector_collections"][0],  # type: ignore[index]
                "vector_collection": "documents__drifted",
            }
        ]
    else:
        drifted[drift_field] = "f" * 64
    monkeypatch.setattr(evaluation_core, "_fixture_state_snapshot", lambda _: drifted)
    with pytest.raises(EvaluationError, match="identity is stale"):
        validate_fixture_state(config)


def test_eval_set_errors_map_to_evaluation_boundary() -> None:
    assert issubclass(EvalSetError, EvaluationError)


def test_retrieval_scoring_uses_configured_root_not_suffix_collision() -> None:
    packet = _packet()
    collision = replace(
        packet,
        evidence=(
            replace(
                packet.evidence[0],
                file="C:/other/Information Retrieval/hybrid_retrieval.md",
            ),
        ),
    )
    score = score_retrieval(
        _item(),
        collision,
        courses_root=Path("C:/fixture"),
    )
    assert not score.passed
    assert score.missing_files == ("Information Retrieval/hybrid_retrieval.md",)


@pytest.mark.parametrize(
    "message",
    (
        'api_key = "quoted secret value"',
        "API KEY : 'spaced secret'",
        "Authorization: Bearer opaque-secret",
        "https://provider.test/?access_token=quoted-secret&keep=1",
        "provider returned sk-live-secret",
    ),
)
def test_eval_failure_sanitization_reuses_retrieval_boundary(message: str) -> None:
    sanitized = sanitize_error(message)
    assert "secret" not in sanitized.casefold()
    assert "opaque" not in sanitized.casefold()
    assert "quoted" not in sanitized.casefold()
    assert len(sanitized) <= 500


@pytest.mark.parametrize(
    ("message", "fragments"),
    (
        ("Authorization: Basic dXNlcjpwYXNz", ("dXNlcjpwYXNz",)),
        (
            'Proxy-Authorization: Digest username="Mufasa", realm="test", nonce="abc"',
            ("Mufasa", "test", "abc"),
        ),
        (
            "Authorization: Custom multi word credential; request_id=keep",
            ("multi word credential",),
        ),
        (
            'Authorization: Custom realm="private", nonce="nonce-leak-value"; request_id=keep',
            ("private", "nonce-leak-value"),
        ),
        (
            "Proxy-Authorization: Fancy first=alpha, second=beta; request_id=keep",
            ("alpha", "beta"),
        ),
        ("password=multi word secret, next=keep", ("multi word secret",)),
        ("https://user:pass@example.test/path", ("user:pass",)),
        (
            "https://provider.test/?token=multi word secret&keep=1",
            ("multi word secret",),
        ),
        ("Bearer multi word bearer-secret", ("multi word bearer-secret",)),
        ("provider returned sk-live-secret", ("sk-live-secret",)),
    ),
)
def test_report_json_and_markdown_redact_complete_credential_values(
    tmp_path: Path,
    message: str,
    fragments: tuple[str, ...],
) -> None:
    report = write_eval_report(
        [
            EvalResult(
                item_id="credential",
                query="private query",
                query_type="concept_explanation",
                status="failed",
                failures=(message,),
            )
        ],
        tmp_path,
    )
    markdown = report.with_suffix(".md")
    for artifact in (report, markdown):
        text = artifact.read_text(encoding="utf-8")
        for fragment in fragments:
            assert fragment not in text
        assert "[redacted]" in text
        assert len(text) < 10_000


def test_run_eval_item_injected_boundaries_preserve_trace_ids_and_timings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="planner",
    )
    packet = _packet()
    evidence_result = EvidenceBuildResult(
        search_run_id=11,
        evidence_packet_id=22,
        retrieval_run=None,  # type: ignore[arg-type]
        coverage=packet.coverage,
        packet=packet,
    )
    monkeypatch.setattr(
        evaluation_core, "build_evidence", lambda *args, **kwargs: evidence_result
    )
    monkeypatch.setattr(
        evaluation_core,
        "generate_answer",
        lambda *args, **kwargs: AnswerResult("grounded answer"),
    )
    monkeypatch.setattr(evaluation_core, "store_answer", lambda *args, **kwargs: 33)
    monkeypatch.setattr(
        evaluation_core,
        "score_retrieval",
        lambda *args, **kwargs: evaluation_core.RetrievalScore(passed=True),
    )
    monkeypatch.setattr(
        evaluation_core,
        "score_citations",
        lambda *args, **kwargs: evaluation_core.CitationScore(
            passed=True,
            valid=True,
        ),
    )

    result = evaluation_core.run_eval_item(_item(), config)

    assert result.passed
    assert result.search_run_id == 11
    assert result.evidence_packet_id == 22
    assert result.answer_id == 33
    assert result.timings_ms and result.timings_ms["total_ms"] >= 0


def test_run_eval_set_continues_after_item_failure_with_real_total_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="planner",
    )
    items = [_item(id="first"), _item(id="second")]
    monkeypatch.setattr(
        evaluation_core, "validate_fixture_state", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(evaluation_core, "load_eval_set", lambda *args, **kwargs: items)

    def fake_run(item: EvalItem, _: object) -> EvalResult:
        if item.id == "first":
            raise RuntimeError("api_key = 'secret'")
        return EvalResult(
            item_id=item.id,
            query=item.query,
            query_type=item.query_type,
            status="passed",
            timings_ms={"evidence_ms": 1.0, "answer_ms": 1.0, "total_ms": 2.0},
        )

    monkeypatch.setattr(evaluation_core, "run_eval_item", fake_run)
    report_path, results = evaluation_core.run_eval_set(config, fixtures=True)

    assert report_path.is_file()
    assert [result.item_id for result in results] == ["first", "second"]
    assert results[0].timings_ms and results[0].timings_ms["total_ms"] > 0
    assert "secret" not in results[0].failures[0]


@pytest.mark.parametrize("error_type", (RetrievalError, EvidenceError))
def test_backend_and_evidence_failure_run_ids_reach_result_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    config = make_config(
        tmp_path,
        embedding_model="BAAI/bge-m3",
        llm_provider="ollama",
        llm_model="planner",
    )
    failure = error_type("provider failed", search_run_id=77)  # type: ignore[call-arg]
    monkeypatch.setattr(
        evaluation_core,
        "build_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(failure),
    )
    with pytest.raises(evaluation_core._EvalItemFailure) as caught:
        evaluation_core.run_eval_item(_item(), config)
    assert caught.value.search_run_id == 77

    monkeypatch.setattr(
        evaluation_core, "validate_fixture_state", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        evaluation_core, "load_eval_set", lambda *args, **kwargs: [_item()]
    )
    monkeypatch.setattr(
        evaluation_core,
        "run_eval_item",
        lambda *args, **kwargs: (_ for _ in ()).throw(caught.value),
    )
    report_path, results = evaluation_core.run_eval_set(config, fixtures=True)
    assert results[0].search_run_id == 77
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["results"][0]["trace_ids"]["search_run_id"] == 77


def test_fixture_preparation_failure_preserves_active_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path, embedding_model="BAAI/bge-m3")
    active = evaluation_core.fixture_state_dir(config)
    active.mkdir(parents=True)
    marker = active / "known-good.marker"
    marker.write_text("known-good", encoding="utf-8")
    monkeypatch.setattr(
        evaluation_core,
        "sync_vector_index",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("vector failed")),
    )

    with pytest.raises(RuntimeError, match="vector failed"):
        evaluation_core.prepare_fixture_state(config)

    assert marker.read_text(encoding="utf-8") == "known-good"
    assert not list(active.parent.glob(".fixture-state-*"))


def test_fixture_activation_move_failure_restores_active_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path, embedding_model="BAAI/bge-m3")
    eval_root = evaluation_core._fixture_eval_root(config)
    active = evaluation_core.fixture_state_dir(config)
    temporary = eval_root / ".fixture-state-test"
    active.mkdir(parents=True)
    temporary.mkdir(parents=True)
    marker = active / "known-good.marker"
    marker.write_text("known-good", encoding="utf-8")

    original_move = evaluation_core.shutil.move

    def fail_activation(source: str, destination: str) -> str:
        if Path(source).resolve() == active.resolve():
            raise OSError("swap failed")
        return original_move(source, destination)

    monkeypatch.setattr(evaluation_core.shutil, "move", fail_activation)
    with pytest.raises(OSError, match="swap failed"):
        evaluation_core._activate_fixture_state(temporary, active, eval_root)

    assert marker.read_text(encoding="utf-8") == "known-good"
    assert temporary.is_dir()


def test_fixture_activation_second_move_failure_restores_active_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(tmp_path, embedding_model="BAAI/bge-m3")
    eval_root = evaluation_core._fixture_eval_root(config)
    active = evaluation_core.fixture_state_dir(config)
    temporary = eval_root / ".fixture-state-second-move-test"
    active.mkdir(parents=True)
    temporary.mkdir(parents=True)
    marker = active / "known-good.marker"
    marker.write_text("known-good", encoding="utf-8")

    original_move = evaluation_core.shutil.move
    calls = 0

    def fail_second_move(source: str, destination: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("temporary activation failed")
        return original_move(source, destination)

    monkeypatch.setattr(evaluation_core.shutil, "move", fail_second_move)
    with pytest.raises(OSError, match="temporary activation failed"):
        evaluation_core._activate_fixture_state(temporary, active, eval_root)

    assert marker.read_text(encoding="utf-8") == "known-good"
    assert temporary.is_dir()
    evaluation_core._safe_remove_fixture_tree(temporary, eval_root)


def test_evaluation_notebook_resolves_root_from_both_working_directories() -> None:
    notebook = json.loads(
        Path("notebooks/evaluation_eda.ipynb").read_text(encoding="utf-8")
    )
    source = "".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "candidate = Path.cwd()" in "".join(cell["source"])
    )
    namespace: dict[str, object] = {}
    original = Path.cwd()
    try:
        for working_directory in (original, original / "notebooks"):
            import os

            os.chdir(working_directory)
            exec(compile(source, "evaluation_eda.ipynb", "exec"), namespace)
            assert Path(namespace["ROOT"]) == original
            assert Path(namespace["REPORT_DIR"]) == original / "data" / "runs" / "eval"
    finally:
        os.chdir(original)
