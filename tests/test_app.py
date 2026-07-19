from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from threading import Event, Lock, Thread
from types import SimpleNamespace

from fastapi.testclient import TestClient

from tests.support import make_config
from uni_rag_agent.answering import AnswerCitation, AnswerGenerationError, AnswerResult
from uni_rag_agent.app import AppServices, create_app
from uni_rag_agent.app.service import ModelRegistry, SessionRegistry
from uni_rag_agent.config import ConfigError
from uni_rag_agent.storage import StorageError


class _Safe:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def as_safe_dict(self) -> dict[str, object]:
        return self.payload


def _coverage() -> _Safe:
    return _Safe(
        {
            "search_run_id": 11,
            "status": "completed",
            "searched_courses": ["Information Retrieval"],
            "searched_indexes": ["documents"],
            "keyword_terms": ["mapreduce"],
            "semantic_queries": ["mapreduce"],
            "raw_result_count": 1,
            "raw_result_counts_by_method": {
                "metadata": 0,
                "keyword": 1,
                "semantic": 0,
            },
            "fused_candidate_count": 1,
            "selectable_candidate_count": 1,
            "evidence_count": 1,
            "evidence_token_count": 5,
            "courses_with_chunk_hits": ["Information Retrieval"],
            "indexes_with_chunk_hits": ["documents"],
            "source_types_with_chunk_hits": ["document"],
            "courses_without_chunk_hits": [],
            "indexes_without_chunk_hits": [],
            "semantic_queries_without_hits": [],
            "missing_capabilities": [],
            "file_only_candidate_count": 0,
            "token_budget_omission_count": 0,
            "oversized_evidence_omission_count": 0,
            "unselected_selectable_candidate_count": 0,
            "weaknesses": [],
        }
    )


def _citation() -> AnswerCitation:
    return AnswerCitation(
        citation_id="E1",
        evidence_index=1,
        course="Information Retrieval",
        file_id=2,
        chunk_id=3,
        file_path="Information Retrieval/lecture.pdf",
        source_type="document",
        location_type="page",
        location_value="4",
        location_label="page 4",
    )


def _answer(*, cited: bool = True) -> AnswerResult:
    return AnswerResult(
        answer_text="MapReduce is a distributed processing model. [E1]"
        if cited
        else "Insufficient evidence was found.",
        citations=(_citation(),) if cited else (),
        limitations=() if cited else ("No evidence chunks were selected.",),
    )


def _services(
    *,
    answer: AnswerResult | None = None,
    store=None,
    generate=None,
) -> AppServices:
    coverage = _coverage()
    packet = SimpleNamespace(
        search_run_id=11,
        coverage=coverage,
        evidence=(object(),),
        as_safe_dict=lambda: {
            "search_run_id": 11,
            "coverage": coverage.as_safe_dict(),
            "evidence": [],
        },
    )
    evidence_result = SimpleNamespace(
        packet=packet,
        evidence_packet_id=22,
        search_run_id=11,
        coverage=coverage,
    )
    resolved_answer = answer or _answer()
    return AppServices(
        build_evidence=lambda *args, **kwargs: evidence_result,
        generate_answer=generate or (lambda *args, **kwargs: resolved_answer),
        store_answer=store or _committed_store,
        load_answer=lambda *args, **kwargs: replace(
            resolved_answer,
            answer_id=33,
            evidence_packet_id=22,
        ),
        load_evidence_packet=lambda *args, **kwargs: packet,
        explain_search_coverage=lambda *args, **kwargs: coverage,
    )


def _committed_store(*args, commit_guard, **kwargs) -> int:
    del args, kwargs
    committed: list[bool] = []
    commit_guard(lambda: committed.append(True))
    assert committed
    return 33


def test_health_is_provider_and_config_independent() -> None:
    def fail_config():
        raise AssertionError("health must not load configuration")

    client = TestClient(create_app(config_loader=fail_config))

    assert client.get("/health").json() == {"status": "ok"}


def test_config_is_operational_and_hides_paths_and_secrets(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        llm_provider="ollama",
        llm_model="planner",
        answer_llm_provider="ollama",
        answer_llm_model="answerer",
    )
    payload = TestClient(create_app(config_loader=lambda: config)).get("/config").json()

    rendered = str(payload)
    assert payload["ask_timeout_seconds"] == 120
    assert payload["paths"]["courses_root_exists"] is True
    assert str(tmp_path) not in rendered
    assert "api_key" not in rendered.lower()


def test_model_registry_builds_each_configured_model_once(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        llm_provider="ollama",
        llm_model="planner",
        answer_llm_provider="ollama",
        answer_llm_model="answerer",
    )
    planner = object()
    answer = object()
    builds: list[str] = []
    registry = ModelRegistry(
        planner_builder=lambda _config: builds.append("planner") or planner,
        answer_builder=lambda _config: builds.append("answer") or answer,
    )

    with TestClient(
        create_app(
            config_loader=lambda: config,
            services=_services(),
            model_registry=registry,
        )
    ):
        pass

    assert registry.planner(config) is planner
    assert registry.planner(config) is planner
    assert registry.answer(config) is answer
    assert registry.answer(config) is answer
    assert builds == ["planner", "answer"]


def test_config_failure_uses_sanitized_503_envelope() -> None:
    def fail_config():
        raise ConfigError("secret path and credential")

    response = TestClient(create_app(config_loader=fail_config)).get("/config")

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "configuration_error",
            "message": "The application configuration is unavailable or invalid.",
        }
    }


def test_framework_http_errors_use_the_stable_error_envelope() -> None:
    client = TestClient(create_app())

    missing = client.get("/api/does-not-exist")
    assert missing.status_code == 404
    assert missing.json() == {
        "error": {
            "code": "not_found",
            "message": "The requested resource does not exist.",
        }
    }

    wrong_method = client.get("/api/ask")
    assert wrong_method.status_code == 405
    assert wrong_method.json() == {
        "error": {
            "code": "method_not_allowed",
            "message": "The requested method is not allowed.",
        }
    }


def test_missing_planner_runtime_configuration_returns_503(tmp_path: Path) -> None:
    response = TestClient(
        create_app(
            config_loader=lambda: make_config(tmp_path),
            services=_services(),
            enforce_model_config=True,
        )
    ).post("/api/ask", json={"query": "configured query"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "configuration_error"


def test_missing_answer_runtime_configuration_returns_503_after_evidence(
    tmp_path: Path,
) -> None:
    config = make_config(
        tmp_path,
        llm_provider="ollama",
        llm_model="planner",
        embedding_model="BAAI/bge-m3",
    )
    response = TestClient(
        create_app(
            config_loader=lambda: config,
            services=_services(),
            enforce_model_config=True,
        )
    ).post("/api/ask", json={"query": "configured query"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "configuration_error"
    assert "evidence packet 22" in response.json()["error"]["message"]


def test_ask_returns_ids_structured_references_and_coverage(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    response = TestClient(
        create_app(config_loader=lambda: config, services=_services())
    ).post("/api/ask", json={"query": "  Explain MapReduce  "})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer_id"] == 33
    assert payload["search_run_id"] == 11
    assert payload["evidence_packet_id"] == 22
    assert payload["citations"][0]["chunk_id"] == 3
    assert payload["references"] == [
        {
            "citation_id": "E1",
            "course": "Information Retrieval",
            "file_path": "Information Retrieval/lecture.pdf",
            "source_type": "document",
            "location_label": "page 4",
        }
    ]
    assert payload["coverage"]["searched_courses"] == ["Information Retrieval"]


def test_answer_projection_separates_body_and_failure_status(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    rendered = AnswerResult(
        answer_text=(
            "First paragraph. [E1]\nSecond paragraph. [E1]\n\n"
            "References:\n- [E1] Information Retrieval - lecture.pdf - page 4\n\n"
            "Limitations:\n- Narrow evidence."
        ),
        citations=(_citation(),),
        limitations=("Narrow evidence.",),
    )
    client = TestClient(
        create_app(config_loader=lambda: config, services=_services(answer=rendered))
    )

    answered = client.post("/api/ask", json={"query": "question"}).json()
    assert answered["answer_body"] == "First paragraph. [E1]\nSecond paragraph. [E1]"
    assert answered["answer_status"] == "answered"

    refused = _answer(cited=False)
    refused = replace(
        refused,
        limitations=(
            "Answer validation failed after 2 attempt(s); "
            "no model answer was accepted.",
        ),
    )
    failure = TestClient(
        create_app(
            config_loader=lambda: config,
            services=_services(answer=refused),
        )
    ).post("/api/ask", json={"query": "question"})
    assert failure.json()["answer_status"] == "validation_failed"


def test_insufficient_evidence_is_a_successful_ask(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    response = TestClient(
        create_app(
            config_loader=lambda: config,
            services=_services(answer=_answer(cited=False)),
        )
    ).post("/api/ask", json={"query": "unknown topic"})

    assert response.status_code == 200
    assert response.json()["citations"] == []
    assert response.json()["references"] == []
    assert response.json()["limitations"] == ["No evidence chunks were selected."]


def test_ask_validation_forbids_blank_invalid_session_and_extra_fields(
    tmp_path: Path,
) -> None:
    client = TestClient(
        create_app(config_loader=lambda: make_config(tmp_path), services=_services())
    )

    for body in (
        {"query": "   "},
        {"query": "ok", "session_id": "contains spaces"},
        {"query": "ok", "model": "forbidden"},
    ):
        response = client.post("/api/ask", json=body)
        assert response.status_code == 422
        assert response.json() == {
            "error": {
                "code": "validation_error",
                "message": "The request is invalid.",
            }
        }


def test_lookup_endpoints_return_locked_public_shapes(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    client = TestClient(create_app(config_loader=lambda: config, services=_services()))

    coverage = client.get("/api/search-runs/11/coverage")
    packet = client.get("/api/evidence-packets/22")
    answer = client.get("/api/answers/33")

    assert coverage.status_code == packet.status_code == answer.status_code == 200
    assert coverage.json()["search_run_id"] == 11
    assert packet.json()["search_run_id"] == 11
    assert answer.json()["answer_id"] == 33
    assert answer.json()["coverage"]["search_run_id"] == 11


def test_missing_lookup_uses_404_envelope(tmp_path: Path) -> None:
    services = replace(
        _services(),
        load_answer=lambda *args, **kwargs: (_ for _ in ()).throw(
            StorageError("Answer 999 does not exist.")
        ),
    )
    response = TestClient(
        create_app(config_loader=lambda: make_config(tmp_path), services=services)
    ).get("/api/answers/999")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "not_found",
            "message": "The requested resource does not exist.",
        }
    }


def test_timeout_returns_504_and_never_persists_late_answer(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), ask_timeout_seconds=0.05)
    stored: list[bool] = []

    def delayed_answer(*args, **kwargs):
        time.sleep(0.15)
        return _answer()

    services = _services(
        generate=delayed_answer,
        store=lambda *args, commit_guard, **kwargs: (
            commit_guard(lambda: stored.append(True)) or 33
        ),
    )
    response = TestClient(
        create_app(config_loader=lambda: config, services=services)
    ).post("/api/ask", json={"query": "slow question"})

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "ask_timeout"
    assert "evidence packet 22" in response.json()["error"]["message"]
    time.sleep(0.2)
    assert stored == []


def test_answer_provider_failure_reports_persisted_packet_without_storing_answer(
    tmp_path: Path,
) -> None:
    stored: list[bool] = []

    def fail_answer(*args, **kwargs):
        raise AnswerGenerationError("provider leaked detail")

    response = TestClient(
        create_app(
            config_loader=lambda: make_config(tmp_path),
            services=_services(
                generate=fail_answer,
                store=lambda *args, **kwargs: stored.append(True) or 33,
            ),
        )
    ).post("/api/ask", json={"query": "provider failure"})

    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "code": "provider_error",
            "message": (
                "A required model service failed after evidence packet 22 was "
                "stored; the packet remains available."
            ),
        }
    }
    assert stored == []


def test_same_session_reuses_only_complete_planner_context(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    contexts: list[tuple[dict[str, str], ...]] = []
    services = _services()

    def capture_build(*args, **kwargs):
        contexts.append(tuple(kwargs["conversation_context"]))
        return services.build_evidence(*args, **kwargs)

    client = TestClient(
        create_app(
            config_loader=lambda: config,
            services=replace(services, build_evidence=capture_build),
        )
    )
    assert (
        client.post(
            "/api/ask", json={"query": "first", "session_id": "study"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/ask", json={"query": "second", "session_id": "study"}
        ).status_code
        == 200
    )

    assert contexts[0] == ()
    assert contexts[1] == (
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": _answer().answer_text},
    )


def test_session_status_distinguishes_live_expired_and_unknown_context(
    tmp_path: Path,
) -> None:
    now = [0.0]
    registry = SessionRegistry(ttl_seconds=10, clock=lambda: now[0])
    client = TestClient(
        create_app(
            config_loader=lambda: make_config(tmp_path),
            services=_services(),
            session_registry=registry,
        )
    )

    assert client.get("/api/sessions/study").json() == {
        "session_id": "study",
        "live": False,
    }
    assert (
        client.post(
            "/api/ask", json={"query": "first", "session_id": "study"}
        ).status_code
        == 200
    )
    now[0] = 9.0
    assert client.get("/api/sessions/study").json()["live"] is True

    now[0] = 11.0
    assert client.get("/api/sessions/study").json()["live"] is False


def test_session_registry_enforces_20_entry_lru_and_two_hour_ttl(
    tmp_path: Path,
) -> None:
    now = [0.0]
    created: list[object] = []

    def factory(config):
        del config
        session = SimpleNamespace(name=len(created))
        created.append(session)
        return session

    registry = SessionRegistry(
        max_sessions=20,
        ttl_seconds=7_200,
        clock=lambda: now[0],
        session_factory=factory,
    )
    config = make_config(tmp_path)
    first = None
    for index in range(20):
        with registry.checkout(f"s{index}", config) as session:
            if index == 0:
                first = session
    with registry.checkout("s20", config):
        pass
    with registry.checkout("s0", config) as recreated:
        assert recreated is not first

    now[0] = 7_201
    with registry.checkout("s20", config) as after_ttl:
        assert after_ttl not in created[:21]


def test_session_registry_serializes_same_id_and_allows_different_ids(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    registry = SessionRegistry()
    entered = Event()
    release = Event()
    state_lock = Lock()
    active = 0
    max_active = 0

    def hold(session_id: str) -> None:
        nonlocal active, max_active
        with registry.checkout(session_id, config):
            with state_lock:
                active += 1
                max_active = max(max_active, active)
                entered.set()
            release.wait(timeout=2)
            with state_lock:
                active -= 1

    first = Thread(target=hold, args=("same",))
    second = Thread(target=hold, args=("same",))
    first.start()
    assert entered.wait(timeout=1)
    second.start()
    time.sleep(0.05)
    assert max_active == 1
    release.set()
    first.join(timeout=1)
    second.join(timeout=1)
    assert not first.is_alive() and not second.is_alive()

    entered.clear()
    release.clear()
    max_active = 0
    one = Thread(target=hold, args=("one",))
    two = Thread(target=hold, args=("two",))
    one.start()
    two.start()
    deadline = time.monotonic() + 1
    while max_active < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    release.set()
    one.join(timeout=1)
    two.join(timeout=1)
    assert max_active == 2


def test_session_registry_never_evicts_active_sessions(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    registry = SessionRegistry(max_sessions=20, ttl_seconds=7_200)
    checkouts = [registry.checkout(f"active-{index}", config) for index in range(20)]
    for checkout in checkouts:
        checkout.__enter__()
    try:
        response = TestClient(
            create_app(
                config_loader=lambda: config,
                services=_services(),
                session_registry=registry,
            )
        ).post(
            "/api/ask",
            json={"query": "capacity", "session_id": "twenty-first"},
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "session_capacity"
    finally:
        for checkout in reversed(checkouts):
            checkout.__exit__(None, None, None)


def test_session_registry_does_not_expire_an_active_session(tmp_path: Path) -> None:
    now = [0.0]
    registry = SessionRegistry(
        max_sessions=2,
        ttl_seconds=10,
        clock=lambda: now[0],
    )
    config = make_config(tmp_path)
    active_checkout = registry.checkout("active", config)
    original = active_checkout.__enter__()
    try:
        now[0] = 20
        with registry.checkout("other", config):
            pass
    finally:
        active_checkout.__exit__(None, None, None)

    with registry.checkout("active", config) as retained:
        assert retained is original


def test_static_ui_loads_as_question_answering_screen() -> None:
    response = TestClient(create_app()).get("/")

    assert response.status_code == 200
    assert "Ask your university materials" in response.text
    assert "ask-form" in response.text
    assert "ingestion" not in response.text.lower()
