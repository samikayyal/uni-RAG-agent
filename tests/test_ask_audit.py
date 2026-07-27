from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from tests.support import make_config
from uni_rag_agent.app.ask_audit import (
    ASK_AUDIT_COLLECTION,
    FirestoreAskAuditStore,
)


class _Snapshot:
    def __init__(self, document_id: str, data: dict[str, object] | None) -> None:
        self.id = document_id
        self.exists = data is not None
        self._data = data

    def to_dict(self):
        return dict(self._data or {})


class _Document:
    def __init__(self, collection: "_Collection", document_id: str) -> None:
        self.collection = collection
        self.id = document_id

    def set(self, fields, merge=False):
        if merge:
            self.collection.data.setdefault(self.id, {}).update(fields)
        else:
            self.collection.data[self.id] = dict(fields)

    def get(self):
        return _Snapshot(self.id, self.collection.data.get(self.id))


class _Query:
    def __init__(self, collection: "_Collection") -> None:
        self.collection = collection
        self.before = None
        self.size = None

    def where(self, field, operator, value):
        assert (field, operator) == ("received_at", "<")
        self.before = value
        return self

    def limit(self, size):
        self.size = size
        return self

    def stream(self):
        records = sorted(
            self.collection.data.items(),
            key=lambda item: item[1]["received_at"],
            reverse=True,
        )
        if self.before is not None:
            records = [item for item in records if item[1]["received_at"] < self.before]
        return [_Snapshot(key, value) for key, value in records[: self.size]]


class _Collection:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, object]] = {}

    def document(self, document_id):
        return _Document(self, document_id)

    def order_by(self, field, direction):
        assert field == "received_at"
        assert direction is not None
        return _Query(self)


class _Client:
    def __init__(self) -> None:
        self.asks = _Collection()

    def collection(self, name):
        assert name == ASK_AUDIT_COLLECTION
        return self.asks


def test_firestore_ask_audit_store_writes_updates_and_pages(tmp_path: Path) -> None:
    config = replace(
        make_config(tmp_path),
        firestore_project_id="offline-project",
    )
    client = _Client()
    store = FirestoreAskAuditStore(config, client=client)
    first_time = datetime(2026, 7, 27, 12, 0, tzinfo=UTC)
    second_time = datetime(2026, 7, 27, 12, 1, tzinfo=UTC)

    first = store.start(
        {"received_at": first_time, "query": "first", "status": "received"}
    )
    second = store.start(
        {"received_at": second_time, "query": "second", "status": "received"}
    )
    store.update(first, {"status": "completed", "response": {"answer": "ok"}})

    assert store.get(first)["response"] == {"answer": "ok"}
    page = store.list_recent(limit=1)
    assert [record["audit_id"] for record in page.records] == [second]
    assert page.next_before == second_time
    older = store.list_recent(limit=10, before=page.next_before)
    assert [record["audit_id"] for record in older.records] == [first]
