"""Durable Firestore audit records for authenticated public-demo asks."""

from __future__ import annotations

import secrets
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Protocol

from ..config import Config

ASK_AUDIT_COLLECTION = "demo_asks"


class AskAuditError(RuntimeError):
    """Raised when the durable ask ledger cannot be read or written."""


class AskAuditStore(Protocol):
    def start(self, record: dict[str, object]) -> str: ...

    def update(self, audit_id: str, fields: dict[str, object]) -> None: ...


@dataclass(frozen=True)
class AskAuditPage:
    records: tuple[dict[str, object], ...]
    next_before: datetime | None


class InMemoryAskAuditStore:
    """Deterministic audit store for focused application tests."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._records: dict[str, dict[str, object]] = {}

    def start(self, record: dict[str, object]) -> str:
        audit_id = secrets.token_hex(16)
        with self._lock:
            self._records[audit_id] = {"audit_id": audit_id, **deepcopy(record)}
        return audit_id

    def update(self, audit_id: str, fields: dict[str, object]) -> None:
        with self._lock:
            if audit_id not in self._records:
                raise AskAuditError(f"Ask audit record {audit_id} does not exist.")
            self._records[audit_id].update(deepcopy(fields))

    def records(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(deepcopy(list(self._records.values())))


class FirestoreAskAuditStore:
    """Store complete safe ask traces in one Firestore document per attempt."""

    def __init__(self, config: Config, *, client: object | None = None) -> None:
        try:
            from google.cloud import firestore
        except ImportError as exc:  # pragma: no cover - optional deployment extra
            raise AskAuditError(
                "Firestore ask-audit support is unavailable in this environment."
            ) from exc
        self._firestore = firestore
        try:
            self._client = client or firestore.Client(
                project=config.firestore_project_id,
                database=config.firestore_database,
            )
        except Exception as exc:
            raise AskAuditError(
                "The Firestore ask-audit service could not be initialized."
            ) from exc

    def start(self, record: dict[str, object]) -> str:
        audit_id = secrets.token_hex(16)
        try:
            (
                self._client.collection(ASK_AUDIT_COLLECTION)
                .document(audit_id)
                .set({"audit_id": audit_id, **record})
            )
        except Exception as exc:
            raise AskAuditError("The authenticated ask could not be recorded.") from exc
        return audit_id

    def update(self, audit_id: str, fields: dict[str, object]) -> None:
        try:
            (
                self._client.collection(ASK_AUDIT_COLLECTION)
                .document(audit_id)
                .set(fields, merge=True)
            )
        except Exception as exc:
            raise AskAuditError("The ask audit record could not be completed.") from exc

    def get(self, audit_id: str) -> dict[str, object] | None:
        try:
            snapshot = (
                self._client.collection(ASK_AUDIT_COLLECTION).document(audit_id).get()
            )
        except Exception as exc:
            raise AskAuditError("The ask audit record could not be read.") from exc
        if not snapshot.exists:
            return None
        return {"audit_id": snapshot.id, **(snapshot.to_dict() or {})}

    def list_recent(
        self,
        *,
        limit: int = 100,
        before: datetime | None = None,
    ) -> AskAuditPage:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        try:
            query = self._client.collection(ASK_AUDIT_COLLECTION).order_by(
                "received_at",
                direction=self._firestore.Query.DESCENDING,
            )
            if before is not None:
                query = query.where("received_at", "<", before)
            snapshots = list(query.limit(limit + 1).stream())
        except Exception as exc:
            raise AskAuditError("Ask audit records could not be listed.") from exc
        has_more = len(snapshots) > limit
        selected = snapshots[:limit]
        records = tuple(
            {"audit_id": snapshot.id, **(snapshot.to_dict() or {})}
            for snapshot in selected
        )
        next_before = None
        if has_more and selected:
            value = (selected[-1].to_dict() or {}).get("received_at")
            if isinstance(value, datetime):
                next_before = value.astimezone(UTC)
        return AskAuditPage(records=records, next_before=next_before)


__all__ = [
    "ASK_AUDIT_COLLECTION",
    "AskAuditError",
    "AskAuditPage",
    "AskAuditStore",
    "FirestoreAskAuditStore",
    "InMemoryAskAuditStore",
]
