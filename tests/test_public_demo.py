from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
import shutil
import subprocess
from threading import Event, Lock, RLock, Thread
from time import monotonic
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tests.support import make_config, make_initialized_config
from tests.test_app import _services
from uni_rag_agent.app import create_app
from uni_rag_agent.app.ask_audit import AskAuditError, InMemoryAskAuditStore
from uni_rag_agent.app.public_demo import (
    AbuseServiceError,
    DemoAuthorizationError,
    DemoQuotaError,
    FirestoreQuotaStore,
    InMemoryQuotaStore,
    SignedDemoTokenManager,
    hash_client_address,
    resolve_client_address,
)
from uni_rag_agent.app.service import EmbeddingRegistry
from uni_rag_agent.config import ConfigError, validate_config
from uni_rag_agent.indexing.embedding_providers.factory import BuiltEmbeddingModel
from uni_rag_agent.indexing import vector as vector_module
from uni_rag_agent.indexing.profiles import EMBEDDING_PROFILES, physical_collection_name
from uni_rag_agent.indexing.vector import semantic_search_many
from uni_rag_agent.search_contracts import LOGICAL_INDEX_TO_SOURCE_TYPE


def _public_config(tmp_path: Path, **changes: object):
    config = make_config(tmp_path)
    values = {
        "public_demo_enabled": True,
        "turnstile_site_key": "site-key",
        "turnstile_secret_key": "turnstile-secret",
        "demo_token_signing_secret": "s" * 40,
        "firestore_project_id": "offline-test-project",
    }
    values.update(changes)
    return replace(config, **values)


def _profile_vector_client(profile):
    logical_index = next(iter(LOGICAL_INDEX_TO_SOURCE_TYPE))
    name = physical_collection_name(
        logical_index,
        provider=profile.provider,
        model_name=profile.model_name,
        dimension=profile.dimension,
        metric=profile.metric,
    )

    class Client:
        def list_collections(self):
            return [type("Collection", (), {"name": name})()]

        def get_collection(self, requested):
            assert requested == name
            return type("Collection", (), {"count": lambda self: 1})()

    return Client()


def _public_client(
    tmp_path: Path,
    *,
    services=None,
    quota_store=None,
    ask_audit_store=None,
    embedding_registry=None,
    raise_server_exceptions: bool = True,
    **config_changes: object,
):
    now = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
    config = _public_config(tmp_path, **config_changes)
    token_manager = SignedDemoTokenManager(
        config.demo_token_signing_secret or "",
        ttl_seconds=config.demo_token_ttl_seconds,
        clock=lambda: now,
    )
    quota = quota_store or InMemoryQuotaStore(config)
    audit = ask_audit_store or InMemoryAskAuditStore()
    client = TestClient(
        create_app(
            config_loader=lambda: config,
            services=services or _services(),
            token_manager=token_manager,
            turnstile_verifier=type(
                "Verifier", (), {"verify": lambda self, value: value == "valid"}
            )(),
            quota_store=quota,
            ask_audit_store=audit,
            embedding_registry=embedding_registry,
            client_resolver=lambda request, hosted_mode: "203.0.113.9",
            now=lambda: now,
        ),
        raise_server_exceptions=raise_server_exceptions,
    )
    return client, config, quota


def _authorize(client: TestClient) -> dict[str, str]:
    response = client.post("/api/demo/session", json={"turnstile_token": "valid"})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['demo_token']}"}


def test_public_settings_are_request_scoped_and_historical_routes_are_hidden(
    tmp_path: Path,
) -> None:
    client, config, _ = _public_client(tmp_path)
    settings = client.get("/api/settings")

    assert settings.status_code == 200
    assert settings.json()["mode"] == "public"
    assert [
        item["model_name"] for item in settings.json()["embedding_model_profiles"]
    ] == list(config.public_embedding_profiles)
    assert client.put("/api/settings", json={"final_top_k": 2}).status_code == 404
    assert client.get("/api/answers/1").status_code == 404
    assert client.get("/api/evidence-packets/1").status_code == 404
    assert client.get("/api/search-runs/1/coverage").status_code == 404
    assert not (config.data_dir / "app_settings.json").exists()


def test_public_config_fails_closed_and_enforces_operator_maxima(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigError, match="TURNSTILE_SITE_KEY"):
        validate_config(
            replace(
                make_config(tmp_path),
                public_demo_enabled=True,
                public_minute_limit=3,
                public_client_daily_limit=10,
                public_global_daily_limit=100,
            )
        )
    with pytest.raises(ConfigError, match="must not exceed 40"):
        validate_config(_public_config(tmp_path, semantic_top_k=41))


def test_public_ask_requires_token_validates_bounds_and_contains_packet(
    tmp_path: Path,
) -> None:
    client, _, quota = _public_client(tmp_path)
    body = {
        "query": "Explain MapReduce",
        "session_id": "session-one",
        "request_id": "request-one",
        "retrieval_settings": {
            "embedding_model": "google/gemini-embedding-001",
            "final_top_k": 15,
            "evidence_max_tokens": 16_000,
        },
    }
    assert client.post("/api/ask", json=body).status_code == 401
    headers = _authorize(client)

    invalid = client.post(
        "/api/ask",
        headers=headers,
        json={
            **body,
            "request_id": "invalid",
            "retrieval_settings": {"final_top_k": 16},
        },
    )
    assert invalid.status_code == 422
    assert quota.remaining("unused", datetime.now(UTC)).client_day == 10

    response = client.post("/api/ask", headers=headers, json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["evidence_packet"]["search_run_id"] == 11
    assert payload["remaining"] == {
        "minute": 2,
        "client_day": 9,
        "global_day": 99,
    }


def test_authenticated_asks_store_full_firestore_audit_traces(
    tmp_path: Path,
) -> None:
    audit = InMemoryAskAuditStore()
    client, _, _ = _public_client(tmp_path, ask_audit_store=audit)
    body = {
        "query": "Explain MapReduce",
        "session_id": "session-one",
        "request_id": "request-one",
        "retrieval_settings": {
            "embedding_model": "google/gemini-embedding-001",
            "final_top_k": 4,
        },
    }

    assert client.post("/api/ask", json=body).status_code == 401
    assert audit.records() == ()

    headers = _authorize(client)
    invalid = client.post(
        "/api/ask",
        headers=headers,
        json={
            **body,
            "request_id": "invalid-settings",
            "retrieval_settings": {"final_top_k": 16},
        },
    )
    completed = client.post("/api/ask", headers=headers, json=body)

    assert invalid.status_code == 422
    assert completed.status_code == 200
    by_request = {record["request_id"]: record for record in audit.records()}
    rejected = by_request["invalid-settings"]
    assert rejected["query"] == "Explain MapReduce"
    assert rejected["status"] == "rejected_validation"
    assert rejected["error"]["code"] == "settings_validation_error"
    assert rejected["client"]["ip_address"] == "203.0.113.9"

    stored = by_request["request-one"]
    assert stored["status"] == "completed"
    assert stored["outcome"] == completed.json()["answer_status"]
    assert stored["models"]["embedding_model"] == "google/gemini-embedding-001"
    assert stored["trace_ids"] == {
        "search_run_id": 11,
        "evidence_packet_id": 22,
        "answer_id": 33,
    }
    assert stored["response"] == completed.json()
    assert stored["response"]["evidence_packet"] == completed.json()["evidence_packet"]
    assert stored["client"]["ip_address"] == "203.0.113.9"
    assert "authorization" not in str(stored).lower()
    assert "timing" in stored


def test_authenticated_replay_and_provider_failure_get_separate_audit_records(
    tmp_path: Path,
) -> None:
    audit = InMemoryAskAuditStore()
    base_services = _services()
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("private provider detail")
        return base_services.build_evidence(*args, **kwargs)

    client, _, _ = _public_client(
        tmp_path,
        ask_audit_store=audit,
        services=replace(base_services, build_evidence=fail_second),
    )
    headers = _authorize(client)
    first_body = {"query": "first", "request_id": "same-id"}

    first = client.post("/api/ask", headers=headers, json=first_body)
    replay = client.post("/api/ask", headers=headers, json=first_body)
    failed = client.post(
        "/api/ask",
        headers=headers,
        json={"query": "second", "request_id": "different-id"},
    )

    assert first.status_code == 200
    assert replay.status_code == 409
    assert failed.status_code == 500
    records = audit.records()
    assert len(records) == 3
    same_id_statuses = {
        record["status"] for record in records if record["request_id"] == "same-id"
    }
    assert same_id_statuses == {"completed", "rejected_replay"}
    failed_record = next(
        record for record in records if record["request_id"] == "different-id"
    )
    assert failed_record["status"] == "failed"
    assert failed_record["error"]["code"] == "internal_error"
    assert "private provider detail" not in str(failed_record)


def test_audit_acceptance_failure_releases_capacity_for_the_next_request(
    tmp_path: Path,
) -> None:
    class FailOnceAuditStore(InMemoryAskAuditStore):
        failed = False

        def update(self, audit_id, fields):
            if fields.get("status") == "accepted" and not self.failed:
                self.failed = True
                raise AskAuditError("offline")
            super().update(audit_id, fields)

    audit = FailOnceAuditStore()
    client, _, _ = _public_client(tmp_path, ask_audit_store=audit)
    headers = _authorize(client)

    failed = client.post(
        "/api/ask",
        headers=headers,
        json={"query": "first", "request_id": "audit-failed"},
    )
    next_request = client.post(
        "/api/ask",
        headers=headers,
        json={"query": "second", "request_id": "audit-recovered"},
    )

    assert failed.status_code == 503
    assert failed.json()["error"]["code"] == "ask_audit_unavailable"
    assert next_request.status_code == 200


def test_completed_request_id_replay_is_rejected_without_new_work_or_quota(
    tmp_path: Path,
) -> None:
    base_services = _services()
    build_calls = 0

    def counted_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return base_services.build_evidence(*args, **kwargs)

    client, config, quota = _public_client(
        tmp_path,
        services=replace(base_services, build_evidence=counted_build),
    )
    headers = _authorize(client)
    body = {
        "query": "Explain MapReduce",
        "request_id": "completed-request",
    }

    first = client.post("/api/ask", headers=headers, json=body)
    after_first = quota.remaining(
        hash_client_address("203.0.113.9", config.demo_token_signing_secret or ""),
        datetime(2026, 7, 22, 12, 0, tzinfo=UTC),
    )
    replay = client.post("/api/ask", headers=headers, json=body)
    after_replay = quota.remaining(
        hash_client_address("203.0.113.9", config.demo_token_signing_secret or ""),
        datetime(2026, 7, 22, 12, 0, tzinfo=UTC),
    )

    assert first.status_code == 200
    assert replay.status_code == 409
    assert replay.json() == {
        "error": {
            "code": "request_already_accepted",
            "message": "This public demo request id was already accepted.",
        }
    }
    assert build_calls == 1
    assert after_first == after_replay


def test_turnstile_rejection_is_fail_closed(tmp_path: Path) -> None:
    client, _, _ = _public_client(tmp_path)
    response = client.post("/api/demo/session", json={"turnstile_token": "invalid"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "demo_authorization_error"


def test_accepted_provider_failure_still_consumes_quota(tmp_path: Path) -> None:
    fail_calls = 0

    def fail(*args, **kwargs):
        nonlocal fail_calls
        fail_calls += 1
        raise RuntimeError("provider failed")

    client, _, quota = _public_client(
        tmp_path,
        services=replace(_services(), build_evidence=fail),
    )
    headers = _authorize(client)
    response = client.post(
        "/api/ask",
        headers=headers,
        json={"query": "q", "request_id": "accepted-failure"},
    )
    replay = client.post(
        "/api/ask",
        headers=headers,
        json={"query": "q", "request_id": "accepted-failure"},
    )
    remaining = quota.remaining(
        hash_client_address("203.0.113.9", "s" * 40),
        datetime(2026, 7, 22, 12, 0, tzinfo=UTC),
    )

    assert response.status_code == 500
    assert replay.status_code == 409
    assert fail_calls == 1
    assert remaining.client_day == 9
    assert remaining.global_day == 99


def test_signed_token_rejects_tampering_expiry_and_client_rebinding() -> None:
    current = [datetime(2026, 7, 22, 12, 0, tzinfo=UTC)]
    manager = SignedDemoTokenManager(
        "a" * 40,
        ttl_seconds=60,
        clock=lambda: current[0],
    )
    token, claims = manager.issue("client-a")
    assert manager.verify(token, "client-a").nonce == claims.nonce
    with pytest.raises(DemoAuthorizationError) as invalid_error:
        manager.verify(token + "x", "client-a")
    assert str(invalid_error.value) == (
        "The demo token is invalid or expired. Please refresh the page."
    )
    with pytest.raises(DemoAuthorizationError):
        manager.verify(token, "client-b")
    current[0] += timedelta(seconds=61)
    with pytest.raises(DemoAuthorizationError):
        manager.verify(token, "client-a")


def test_quota_reservation_is_idempotent_and_rolls_over_utc_day(tmp_path: Path) -> None:
    config = _public_config(tmp_path)
    store = InMemoryQuotaStore(config)
    now = datetime(2026, 7, 22, 23, 59, 30, tzinfo=UTC)
    assert store.lookup("reservation") is None
    first = store.reserve("client", "reservation", now)
    repeated = store.reserve("client", "reservation", now)
    assert first.created is True
    assert repeated.created is False
    assert first.remaining == repeated.remaining
    store.reserve("client", "second", now)
    store.reserve("client", "third", now)
    with pytest.raises(DemoQuotaError):
        store.reserve("client", "fourth", now)
    next_day = now + timedelta(seconds=61)
    assert store.reserve("client", "next-day", next_day).remaining.client_day == 9


def test_quota_enforces_client_and_global_daily_limits(tmp_path: Path) -> None:
    config = _public_config(tmp_path)
    client_store = InMemoryQuotaStore(config)
    start = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
    reservation = 0
    for minute, count in ((0, 3), (1, 3), (2, 3), (3, 1)):
        for _ in range(count):
            reservation += 1
            client_store.reserve(
                "one-client",
                f"client-{reservation}",
                start + timedelta(seconds=61 * minute),
            )
    with pytest.raises(DemoQuotaError):
        client_store.reserve(
            "one-client", "client-over", start + timedelta(seconds=61 * 4)
        )

    global_store = InMemoryQuotaStore(config)
    for index in range(100):
        client = f"client-{index // 3}"
        global_store.reserve(client, f"global-{index}", start)
    with pytest.raises(DemoQuotaError):
        global_store.reserve("fresh-client", "global-over", start)


def test_busy_third_ask_does_not_consume_quota(tmp_path: Path) -> None:
    now = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
    config = _public_config(
        tmp_path,
        public_ask_capacity=2,
        public_capacity_wait_seconds=1,
    )
    entered = Event()
    release = Event()
    count_lock = Lock()
    entered_count = 0
    base_services = _services()

    def blocking_build(*args, **kwargs):
        nonlocal entered_count
        with count_lock:
            entered_count += 1
            if entered_count == 2:
                entered.set()
        assert release.wait(5)
        return base_services.build_evidence(*args, **kwargs)

    services = replace(base_services, build_evidence=blocking_build)
    token_manager = SignedDemoTokenManager(
        config.demo_token_signing_secret or "",
        ttl_seconds=config.demo_token_ttl_seconds,
        clock=lambda: now,
    )
    quota = InMemoryQuotaStore(config)
    audit = InMemoryAskAuditStore()
    client = TestClient(
        create_app(
            config_loader=lambda: config,
            services=services,
            token_manager=token_manager,
            turnstile_verifier=type(
                "Verifier", (), {"verify": lambda self, value: value == "valid"}
            )(),
            quota_store=quota,
            ask_audit_store=audit,
            client_resolver=lambda request, hosted_mode: "203.0.113.9",
            now=lambda: now,
        )
    )
    headers = _authorize(client)
    responses: list[int] = []

    def ask(request_id: str) -> None:
        response = client.post(
            "/api/ask",
            headers=headers,
            json={"query": "q", "request_id": request_id},
        )
        responses.append(response.status_code)

    workers = [Thread(target=ask, args=(f"accepted-{index}",)) for index in range(2)]
    for worker in workers:
        worker.start()
    assert entered.wait(5)
    before = quota.remaining(hash_client_address("203.0.113.9", "s" * 40), now)
    busy = client.post(
        "/api/ask",
        headers=headers,
        json={"query": "q", "request_id": "busy-third"},
    )
    after = quota.remaining(hash_client_address("203.0.113.9", "s" * 40), now)
    release.set()
    for worker in workers:
        worker.join(5)

    assert busy.status_code == 503
    assert busy.json()["error"]["code"] == "ask_capacity_busy"
    assert before == after
    assert sorted(responses) == [200, 200]


def test_completed_replay_bypasses_saturated_capacity_without_new_work_or_quota(
    tmp_path: Path,
) -> None:
    entered = Event()
    release = Event()
    call_lock = Lock()
    build_calls = 0
    blocked_calls = 0
    base_services = _services()

    def block_after_first(*args, **kwargs):
        nonlocal build_calls, blocked_calls
        with call_lock:
            build_calls += 1
            should_block = build_calls > 1
            if should_block:
                blocked_calls += 1
                if blocked_calls == 2:
                    entered.set()
        if should_block:
            assert release.wait(10)
        return base_services.build_evidence(*args, **kwargs)

    client, config, quota = _public_client(
        tmp_path,
        services=replace(base_services, build_evidence=block_after_first),
        public_ask_capacity=2,
        public_capacity_wait_seconds=5,
    )
    headers = _authorize(client)
    completed_body = {"query": "q", "request_id": "completed-before-busy"}
    assert (
        client.post("/api/ask", headers=headers, json=completed_body).status_code == 200
    )

    statuses: list[int] = []

    def ask(request_id: str) -> None:
        statuses.append(
            client.post(
                "/api/ask",
                headers=headers,
                json={"query": "q", "request_id": request_id},
            ).status_code
        )

    workers = [Thread(target=ask, args=(f"capacity-{index}",)) for index in range(2)]
    for worker in workers:
        worker.start()
    assert entered.wait(5)
    client_hash = hash_client_address(
        "203.0.113.9", config.demo_token_signing_secret or ""
    )
    before = quota.remaining(client_hash, datetime(2026, 7, 22, 12, 0, tzinfo=UTC))
    started = monotonic()
    try:
        replay = client.post("/api/ask", headers=headers, json=completed_body)
        elapsed = monotonic() - started
    finally:
        release.set()
        for worker in workers:
            worker.join(10)
    after = quota.remaining(client_hash, datetime(2026, 7, 22, 12, 0, tzinfo=UTC))

    assert replay.status_code == 409
    assert replay.json()["error"]["code"] == "request_already_accepted"
    assert elapsed < 1
    assert before == after
    assert build_calls == 3
    assert sorted(statuses) == [200, 200]


def test_only_owning_demo_token_can_cancel_an_active_ask(tmp_path: Path) -> None:
    entered = Event()
    release = Event()
    base_services = _services()

    def blocking_build(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return base_services.build_evidence(*args, **kwargs)

    client, _, _ = _public_client(
        tmp_path,
        services=replace(base_services, build_evidence=blocking_build),
    )
    owner_headers = _authorize(client)
    other_headers = _authorize(client)
    result: list[int] = []

    worker = Thread(
        target=lambda: result.append(
            client.post(
                "/api/ask",
                headers=owner_headers,
                json={"query": "q", "request_id": "owned-request"},
            ).status_code
        )
    )
    worker.start()
    assert entered.wait(5)
    denied = client.post("/api/asks/owned-request/cancel", headers=other_headers)
    accepted = client.post("/api/asks/owned-request/cancel", headers=owner_headers)
    release.set()
    worker.join(5)

    assert denied.status_code == 404
    assert accepted.status_code == 200
    assert accepted.json()["cancelled"] is True
    assert result == [499]


class _FakeSnapshot:
    def __init__(self, reference, data):
        self.reference = reference
        self._data = data
        self.exists = data is not None

    def to_dict(self):
        return dict(self._data) if self._data is not None else None


class _FakeReference:
    def __init__(self, client, path: str):
        self.client = client
        self.path = path

    def get(self):
        return _FakeSnapshot(self, self.client.data.get(self.path))


class _FakeCollection:
    def __init__(self, client, name: str):
        self.client = client
        self.name = name

    def document(self, document_id: str):
        return _FakeReference(self.client, f"{self.name}/{document_id}")


class _FakeTransaction:
    def __init__(self, client):
        self.client = client

    def get_all(self, references):
        return [reference.get() for reference in references]

    def set(self, reference, value):
        self.client.data[reference.path] = dict(value)


class _FakeFirestoreClient:
    def __init__(self):
        self.data: dict[str, dict[str, object]] = {}
        self.lock = Lock()

    def collection(self, name: str):
        return _FakeCollection(self, name)

    def transaction(self):
        return _FakeTransaction(self)


class _FailingFirestoreClient:
    def collection(self, name: str):
        del name
        return self

    def document(self, document_id: str):
        del document_id
        return self

    def get(self):
        raise RuntimeError("firestore unavailable")


def _firestore_store(client, config):
    store = object.__new__(FirestoreQuotaStore)
    store._client = client
    store._minute_limit = config.public_minute_limit
    store._client_limit = config.public_client_daily_limit
    store._global_limit = config.public_global_daily_limit
    return store


def test_firestore_remaining_failure_uses_exact_api_503_envelope(
    tmp_path: Path,
) -> None:
    config = _public_config(tmp_path)
    store = _firestore_store(_FailingFirestoreClient(), config)
    with pytest.raises(AbuseServiceError, match="quota service"):
        store.remaining("client", datetime(2026, 7, 22, 12, 0, tzinfo=UTC))

    client, _, _ = _public_client(
        tmp_path,
        quota_store=store,
        raise_server_exceptions=False,
    )
    response = client.post("/api/demo/session", json={"turnstile_token": "valid"})

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "abuse_service_unavailable",
            "message": "The public quota service is temporarily unavailable.",
        }
    }

    token_manager = SignedDemoTokenManager(
        config.demo_token_signing_secret or "",
        ttl_seconds=config.demo_token_ttl_seconds,
        clock=lambda: datetime(2026, 7, 22, 12, 0, tzinfo=UTC),
    )
    client_hash = hash_client_address(
        "203.0.113.9", config.demo_token_signing_secret or ""
    )
    token, _ = token_manager.issue(client_hash)
    ask = client.post(
        "/api/ask",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "q", "request_id": "lookup-failure"},
    )
    assert ask.status_code == 503
    assert ask.json() == response.json()


def test_firestore_transaction_race_is_idempotent_and_never_exceeds_limit() -> None:
    client = _FakeFirestoreClient()

    def transactional(function):
        def run(transaction):
            with transaction.client.lock:
                return function(transaction)

        return run

    store = object.__new__(FirestoreQuotaStore)
    store._firestore = SimpleNamespace(transactional=transactional)
    store._client = client
    store._minute_limit = 3
    store._client_limit = 10
    store._global_limit = 100
    now = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)

    repeated: list[object] = []
    workers = [
        Thread(target=lambda: repeated.append(store.reserve("client", "same", now)))
        for _ in range(8)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
    assert len(repeated) == 8
    assert sum(item.created for item in repeated) == 1
    assert {item.remaining.minute for item in repeated} == {2}
    assert store.lookup("same") == repeated[0].remaining

    outcomes: list[str] = []

    def reserve_unique(index: int) -> None:
        try:
            store.reserve("client", f"unique-{index}", now)
            outcomes.append("accepted")
        except DemoQuotaError:
            outcomes.append("limited")

    workers = [Thread(target=reserve_unique, args=(index,)) for index in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)

    assert outcomes.count("accepted") == 2
    assert outcomes.count("limited") == 6
    assert client.data["demo_quota_global/2026-07-22"]["count"] == 3


def test_hosted_client_resolution_uses_platform_appended_last_hop() -> None:
    request = type(
        "Request",
        (),
        {
            "headers": {"x-forwarded-for": "spoofed, 198.51.100.7"},
            "client": type("Client", (), {"host": "127.0.0.1"})(),
        },
    )()
    assert resolve_client_address(request, hosted_mode=True) == "198.51.100.7"
    assert resolve_client_address(request, hosted_mode=False) == "127.0.0.1"
    assert hash_client_address("198.51.100.7", "secret") != "198.51.100.7"


def test_embedding_registry_reuses_models_client_and_only_locks_local_profiles(
    tmp_path: Path,
) -> None:
    builds: list[str] = []
    clients: list[Path] = []

    def build(config, model, *, error):
        del config, error
        builds.append(model)
        profile = EMBEDDING_PROFILES[model]
        return BuiltEmbeddingModel(object(), profile, profile.dimension)

    def build_client(config, *, error):
        del error
        clients.append(config.chroma_dir)
        return object()

    registry = EmbeddingRegistry(
        embedding_builder=build,
        chroma_builder=build_client,
    )
    local_config = replace(
        make_config(tmp_path), embedding_model="google/embeddinggemma-300m"
    )
    hosted_config = replace(local_config, embedding_model="google/gemini-embedding-001")

    local_one = registry.runtime(local_config)
    local_two = registry.runtime(local_config)
    hosted = registry.runtime(hosted_config)

    assert local_one.built is local_two.built
    assert local_one.chroma_client is hosted.chroma_client
    assert local_one.encoding_lock is not None
    assert hosted.encoding_lock is None
    assert builds == ["google/embeddinggemma-300m", "google/gemini-embedding-001"]
    assert clients == [local_config.chroma_dir]


def test_embedding_registry_reports_lazy_local_load_once_and_recovers_cached_failure(
    tmp_path: Path,
) -> None:
    config = replace(
        make_config(tmp_path), embedding_model="google/embeddinggemma-300m"
    )
    profile = EMBEDDING_PROFILES["google/embeddinggemma-300m"]
    attempts = 0
    fail = True

    def build(config, model, *, error):
        nonlocal attempts
        del config, model, error
        attempts += 1
        if fail:
            raise RuntimeError("model unavailable")
        return BuiltEmbeddingModel(object(), profile, profile.dimension)

    registry = EmbeddingRegistry(
        embedding_builder=build,
        chroma_builder=lambda config, *, error: object(),
    )
    phases: list[str] = []

    with pytest.raises(RuntimeError, match="model unavailable"):
        registry.runtime(config, progress_callback=phases.append)
    with pytest.raises(RuntimeError, match="model unavailable"):
        registry.runtime(config, progress_callback=phases.append)
    assert attempts == 1
    assert phases == ["loading_embedding_model"]

    fail = False
    registry.clear_failure(profile.model_name)
    first = registry.runtime(config, progress_callback=phases.append)
    second = registry.runtime(config, progress_callback=phases.append)

    assert attempts == 2
    assert first.built is second.built
    assert phases == [
        "loading_embedding_model",
        "loading_embedding_model",
    ]


def test_embedding_registry_prepares_one_local_profile_without_blocking_hosted(
    tmp_path: Path,
) -> None:
    local = "google/embeddinggemma-300m"
    hosted = "Qwen/Qwen3-Embedding-8B"
    entered = Event()
    release = Event()
    builds: list[str] = []
    errors: list[Exception] = []

    def build(_config, model, *, error):
        del error
        builds.append(model)
        if model == local:
            entered.set()
            assert release.wait(5)
        profile = EMBEDDING_PROFILES[model]
        return BuiltEmbeddingModel(object(), profile, profile.dimension)

    registry = EmbeddingRegistry(
        embedding_builder=build,
        chroma_builder=lambda _config, *, error: object(),
    )
    local_config = replace(make_config(tmp_path), embedding_model=local)
    hosted_config = replace(local_config, embedding_model=hosted)

    def prepare() -> None:
        try:
            registry.prepare(local_config, local)
        except Exception as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    first = Thread(target=prepare)
    duplicate = Thread(target=prepare)
    first.start()
    assert entered.wait(2)
    duplicate.start()
    hosted_finished = Event()

    def acquire_hosted() -> None:
        try:
            registry.runtime(hosted_config)
            hosted_finished.set()
        except Exception as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    hosted_thread = Thread(target=acquire_hosted)
    hosted_thread.start()
    assert hosted_finished.wait(2)
    release.set()
    first.join(5)
    duplicate.join(5)
    hosted_thread.join(5)

    assert not errors
    assert builds.count(local) == 1
    assert builds.count(hosted) == 1


def test_semantic_encoding_lock_serializes_local_and_allows_hosted_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(make_config(tmp_path), embedding_model="BAAI/bge-m3")
    profile = EMBEDDING_PROFILES["BAAI/bge-m3"]

    class ConcurrentEmbeddings:
        def __init__(self) -> None:
            self.active = 0
            self.maximum = 0
            self.guard = Lock()
            self.entered = Event()
            self.overlapped = Event()
            self.release = Event()

        def embed_queries(self, queries):
            with self.guard:
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                self.entered.set()
                if self.active == 2:
                    self.overlapped.set()
            assert self.release.wait(5)
            with self.guard:
                self.active -= 1
            return [[1.0] for _ in queries]

    def exercise(lock, *, expect_overlap: bool) -> int:
        embeddings = ConcurrentEmbeddings()
        context = SimpleNamespace(
            built=SimpleNamespace(
                embeddings=embeddings,
                profile=profile,
                dimension=1,
            ),
            collections={},
            counts={},
        )
        monkeypatch.setattr(
            vector_module,
            "_build_semantic_context",
            lambda *args, **kwargs: context,
        )
        errors: list[Exception] = []

        def search() -> None:
            try:
                semantic_search_many(
                    config,
                    ["query"],
                    indexes=["document_index"],
                    encoding_lock=lock,
                )
            except Exception as exc:  # pragma: no cover - assertion aid
                errors.append(exc)

        first = Thread(target=search)
        second = Thread(target=search)
        first.start()
        assert embeddings.entered.wait(2)
        second.start()
        if expect_overlap:
            assert embeddings.overlapped.wait(2)
        else:
            assert not embeddings.overlapped.wait(0.2)
        embeddings.release.set()
        first.join(5)
        second.join(5)
        assert not errors
        return embeddings.maximum

    assert exercise(RLock(), expect_overlap=False) == 1
    assert exercise(None, expect_overlap=True) == 2


def test_hosted_registry_readiness_checks_only_default_vector_space_without_models(
    tmp_path: Path,
) -> None:
    logical_index = next(iter(LOGICAL_INDEX_TO_SOURCE_TYPE))
    default = "Qwen/Qwen3-Embedding-8B"
    names = {
        physical_collection_name(
            logical_index,
            provider=EMBEDDING_PROFILES[default].provider,
            model_name=default,
            dimension=EMBEDDING_PROFILES[default].dimension,
            metric=EMBEDDING_PROFILES[default].metric,
        )
    }

    class Client:
        def list_collections(self):
            return [type("Collection", (), {"name": name})() for name in names]

        def get_collection(self, name):
            assert name in names
            return type("Collection", (), {"count": lambda self: 1})()

    builds: list[str] = []

    def build(config, model, *, error):
        del config, error
        builds.append(model)
        profile = EMBEDDING_PROFILES[model]
        return BuiltEmbeddingModel(object(), profile, profile.dimension)

    registry = EmbeddingRegistry(
        embedding_builder=build,
        chroma_builder=lambda config, *, error: Client(),
    )
    config = _public_config(
        tmp_path,
        hosted_mode=True,
        public_default_embedding_model=default,
    )

    assert not registry.hosted_ready()
    registry.warm_hosted(config)
    assert registry.hosted_ready()
    assert builds == []

    registry.runtime(replace(config, embedding_model="google/gemini-embedding-001"))
    assert builds == ["google/gemini-embedding-001"]


def test_hosted_readiness_returns_the_default_vector_space_contract(
    tmp_path: Path,
) -> None:
    default = "Qwen/Qwen3-Embedding-8B"
    offline_model = tmp_path / "offline-embeddinggemma"
    offline_model.mkdir()
    config = make_initialized_config(
        tmp_path,
        hosted_mode=True,
        public_demo_enabled=True,
        turnstile_site_key="site-key",
        turnstile_secret_key="turnstile-secret",
        demo_token_signing_secret="s" * 40,
        firestore_project_id="offline-test-project",
        public_default_embedding_model=default,
        embeddinggemma_model_path=offline_model,
    )
    logical_index = next(iter(LOGICAL_INDEX_TO_SOURCE_TYPE))
    name = physical_collection_name(
        logical_index,
        provider=EMBEDDING_PROFILES[default].provider,
        model_name=default,
        dimension=EMBEDDING_PROFILES[default].dimension,
        metric=EMBEDDING_PROFILES[default].metric,
    )

    class Client:
        def list_collections(self):
            return [type("Collection", (), {"name": name})()]

        def get_collection(self, requested):
            assert requested == name
            return type("Collection", (), {"count": lambda self: 1})()

    builds: list[str] = []
    registry = EmbeddingRegistry(
        embedding_builder=lambda _config, model, *, error: builds.append(model),
        chroma_builder=lambda _config, *, error: Client(),
    )
    with TestClient(
        create_app(
            config_loader=lambda: config,
            services=_services(),
            embedding_registry=registry,
        )
    ) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "storage_ready": True,
        "default_vector_index_ready": True,
    }
    assert builds == []


def test_hosted_readiness_returns_503_when_the_default_vector_space_is_missing(
    tmp_path: Path,
) -> None:
    offline_model = tmp_path / "offline-embeddinggemma"
    offline_model.mkdir()
    config = make_initialized_config(
        tmp_path,
        hosted_mode=True,
        public_demo_enabled=True,
        turnstile_site_key="site-key",
        turnstile_secret_key="turnstile-secret",
        demo_token_signing_secret="s" * 40,
        firestore_project_id="offline-test-project",
        public_default_embedding_model="Qwen/Qwen3-Embedding-8B",
        embeddinggemma_model_path=offline_model,
    )

    class EmptyClient:
        def list_collections(self):
            return []

    registry = EmbeddingRegistry(
        chroma_builder=lambda _config, *, error: EmptyClient(),
    )
    with TestClient(
        create_app(
            config_loader=lambda: config,
            services=_services(),
            embedding_registry=registry,
        )
    ) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "storage_ready": True,
        "default_vector_index_ready": False,
    }


def test_public_embedding_profile_preparation_is_authenticated_and_quota_free(
    tmp_path: Path,
) -> None:
    profile = EMBEDDING_PROFILES["google/embeddinggemma-300m"]
    builds: list[str] = []
    registry = EmbeddingRegistry(
        embedding_builder=lambda _config, model, *, error: (
            builds.append(model)
            or BuiltEmbeddingModel(object(), profile, profile.dimension)
        ),
        chroma_builder=lambda _config, *, error: _profile_vector_client(profile),
    )
    client, _, quota = _public_client(tmp_path, embedding_registry=registry)
    body = {"embedding_model": profile.model_name}

    assert client.post("/api/embedding-profiles/prepare", json=body).status_code == 401
    headers = _authorize(client)
    response = client.post(
        "/api/embedding-profiles/prepare", headers=headers, json=body
    )

    assert response.status_code == 200
    assert response.json() == {"embedding_model": profile.model_name, "status": "ready"}
    assert (
        client.post(
            "/api/embedding-profiles/prepare", headers=headers, json=body
        ).status_code
        == 200
    )
    assert builds == [profile.model_name]
    assert quota.remaining("unused", datetime.now(UTC)).client_day == 10
    invalid = client.post(
        "/api/embedding-profiles/prepare",
        headers=headers,
        json={"embedding_model": "Qwen/Qwen3-Embedding-8B"},
    )
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "embedding_profile_invalid"


def test_embedding_profile_preparation_requires_a_usable_vector_index(
    tmp_path: Path,
) -> None:
    profile = EMBEDDING_PROFILES["google/embeddinggemma-300m"]
    builds: list[str] = []

    class EmptyClient:
        def list_collections(self):
            return []

    registry = EmbeddingRegistry(
        embedding_builder=lambda _config, model, *, error: (
            builds.append(model)
            or BuiltEmbeddingModel(object(), profile, profile.dimension)
        ),
        chroma_builder=lambda _config, *, error: EmptyClient(),
    )
    client, _, _ = _public_client(tmp_path, embedding_registry=registry)
    response = client.post(
        "/api/embedding-profiles/prepare",
        headers=_authorize(client),
        json={"embedding_model": profile.model_name},
    )

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "embedding_profile_index_unavailable",
        "message": "The embedding profile does not have a usable vector index.",
    }
    assert builds == []


def test_explicit_embedding_profile_preparation_retries_cached_failure(
    tmp_path: Path,
) -> None:
    profile = EMBEDDING_PROFILES["google/embeddinggemma-300m"]
    failed = True
    attempts = 0

    def build(_config, _model, *, error):
        nonlocal attempts
        attempts += 1
        if failed:
            raise RuntimeError("private model path")
        return BuiltEmbeddingModel(object(), profile, profile.dimension)

    registry = EmbeddingRegistry(
        embedding_builder=build,
        chroma_builder=lambda _config, *, error: _profile_vector_client(profile),
    )
    client, _, _ = _public_client(tmp_path, embedding_registry=registry)
    headers = _authorize(client)
    body = {"embedding_model": profile.model_name}

    failed_response = client.post(
        "/api/embedding-profiles/prepare", headers=headers, json=body
    )
    assert failed_response.status_code == 502
    assert failed_response.json() == {
        "error": {
            "code": "embedding_profile_unavailable",
            "message": "The embedding profile is currently unavailable.",
        }
    }
    failed = False
    recovered = client.post(
        "/api/embedding-profiles/prepare", headers=headers, json=body
    )
    assert recovered.status_code == 200
    assert attempts == 2


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_public_frontend_state_and_script_execute_under_node() -> None:
    repo_root = Path(__file__).parents[1]
    script = repo_root / "src" / "uni_rag_agent" / "app" / "static" / "app.js"
    syntax = subprocess.run(
        ["node", "--check", str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    behavior = subprocess.run(
        ["node", "--test", str(repo_root / "tests" / "browser_state.test.js")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    assert behavior.returncode == 0, behavior.stdout + behavior.stderr
