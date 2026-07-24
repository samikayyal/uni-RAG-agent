"""Public-demo authentication, client privacy, and transactional quotas."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Protocol

from ..config import Config

TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


class DemoAuthorizationError(ValueError):
    """Raised when a public demo token is missing, invalid, or expired."""


class DemoQuotaError(RuntimeError):
    """Raised when a valid public client has exhausted an ask allowance."""

    def __init__(self, message: str, *, remaining: QuotaRemaining) -> None:
        super().__init__(message)
        self.remaining = remaining


class AbuseServiceError(RuntimeError):
    """Raised when CAPTCHA or quota infrastructure cannot fail safely."""


@dataclass(frozen=True)
class DemoTokenClaims:
    nonce: str
    client_hash: str
    issued_at: int
    expires_at: int


@dataclass(frozen=True)
class QuotaRemaining:
    minute: int
    client_day: int
    global_day: int

    def as_dict(self) -> dict[str, int]:
        return {
            "minute": max(0, self.minute),
            "client_day": max(0, self.client_day),
            "global_day": max(0, self.global_day),
        }


@dataclass(frozen=True)
class QuotaReservation:
    """Result of an idempotent quota reservation attempt."""

    remaining: QuotaRemaining
    created: bool


class QuotaStore(Protocol):
    def remaining(self, client_hash: str, now: datetime) -> QuotaRemaining: ...

    def lookup(self, reservation_id: str) -> QuotaRemaining | None: ...

    def reserve(
        self,
        client_hash: str,
        reservation_id: str,
        now: datetime,
    ) -> QuotaReservation: ...


class TurnstileVerifier(Protocol):
    def verify(self, response_token: str) -> bool: ...


class SignedDemoTokenManager:
    """Issue and verify signed, short-lived tokens with explicit claim checks."""

    def __init__(
        self,
        secret: str,
        *,
        ttl_seconds: int = 30 * 60,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        try:
            from itsdangerous import URLSafeSerializer
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise AbuseServiceError(
                "Signed-token support is unavailable in this deployment."
            ) from exc
        self._serializer = URLSafeSerializer(
            secret_key=secret,
            salt="uni-rag-public-demo-v1",
            signer_kwargs={"digest_method": hashlib.sha256},
        )
        self._ttl_seconds = ttl_seconds
        self._clock = clock or (lambda: datetime.now(UTC))

    def issue(self, client_hash: str) -> tuple[str, DemoTokenClaims]:
        now = int(self._clock().timestamp())
        claims = DemoTokenClaims(
            nonce=secrets.token_urlsafe(24),
            client_hash=client_hash,
            issued_at=now,
            expires_at=now + self._ttl_seconds,
        )
        token = self._serializer.dumps(
            {
                "v": 1,
                "nonce": claims.nonce,
                "client": claims.client_hash,
                "iat": claims.issued_at,
                "exp": claims.expires_at,
            }
        )
        return token, claims

    def verify(self, token: str, client_hash: str) -> DemoTokenClaims:
        try:
            value = self._serializer.loads(token)
        except Exception as exc:
            raise DemoAuthorizationError(
                "The demo token is invalid or expired."
            ) from exc
        if not isinstance(value, dict) or value.get("v") != 1:
            raise DemoAuthorizationError("The demo token is invalid or expired.")
        try:
            claims = DemoTokenClaims(
                nonce=str(value["nonce"]),
                client_hash=str(value["client"]),
                issued_at=int(value["iat"]),
                expires_at=int(value["exp"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DemoAuthorizationError(
                "The demo token is invalid or expired."
            ) from exc
        now = int(self._clock().timestamp())
        if (
            not claims.nonce
            or claims.issued_at > now + 30
            or claims.expires_at <= now
            or claims.expires_at - claims.issued_at > self._ttl_seconds
            or not hmac.compare_digest(claims.client_hash, client_hash)
        ):
            raise DemoAuthorizationError("The demo token is invalid or expired.")
        return claims


class CloudflareTurnstileVerifier:
    """Verify a single-use Turnstile response without logging response content."""

    def __init__(self, secret_key: str, *, timeout_seconds: float = 5.0) -> None:
        self._secret_key = secret_key
        self._timeout_seconds = timeout_seconds

    def verify(self, response_token: str) -> bool:
        if not response_token.strip():
            return False
        body = urllib.parse.urlencode(
            {
                "secret": self._secret_key,
                "response": response_token,
            }
        ).encode("ascii")
        request = urllib.request.Request(
            TURNSTILE_VERIFY_URL,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout_seconds
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise AbuseServiceError(
                "The abuse-prevention service is temporarily unavailable."
            ) from exc
        return isinstance(payload, dict) and payload.get("success") is True


def resolve_client_address(request: Any, *, hosted_mode: bool) -> str:
    """Resolve the platform-appended address while ignoring spoofable prefixes."""
    if hosted_mode:
        forwarded = request.headers.get("x-forwarded-for", "")
        values = [item.strip() for item in forwarded.split(",") if item.strip()]
        if not values:
            raise AbuseServiceError("The public client address could not be verified.")
        return values[-1]
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    if not isinstance(host, str) or not host:
        raise AbuseServiceError("The client address could not be verified.")
    return host


def hash_client_address(address: str, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"), address.encode("utf-8"), hashlib.sha256
    ).hexdigest()


class InMemoryQuotaStore:
    """Transaction-equivalent quota store for deterministic offline tests."""

    def __init__(self, config: Config) -> None:
        self._minute_limit = config.public_minute_limit
        self._client_limit = config.public_client_daily_limit
        self._global_limit = config.public_global_daily_limit
        self._lock = Lock()
        self._global: dict[str, int] = {}
        self._client: dict[tuple[str, str], int] = {}
        self._minute: dict[str, list[float]] = {}
        self._reservations: dict[str, QuotaRemaining] = {}

    def remaining(self, client_hash: str, now: datetime) -> QuotaRemaining:
        with self._lock:
            return self._remaining_locked(client_hash, now)

    def lookup(self, reservation_id: str) -> QuotaRemaining | None:
        with self._lock:
            return self._reservations.get(reservation_id)

    def reserve(
        self,
        client_hash: str,
        reservation_id: str,
        now: datetime,
    ) -> QuotaReservation:
        with self._lock:
            previous = self._reservations.get(reservation_id)
            if previous is not None:
                return QuotaReservation(remaining=previous, created=False)
            current = self._remaining_locked(client_hash, now)
            if min(current.minute, current.client_day, current.global_day) <= 0:
                raise DemoQuotaError(
                    "The public demo ask limit has been reached.",
                    remaining=current,
                )
            day = now.astimezone(UTC).strftime("%Y-%m-%d")
            self._global[day] = self._global.get(day, 0) + 1
            client_key = (client_hash, day)
            self._client[client_key] = self._client.get(client_key, 0) + 1
            cutoff = now.timestamp() - 60.0
            timestamps = [
                item for item in self._minute.get(client_hash, []) if item > cutoff
            ]
            timestamps.append(now.timestamp())
            self._minute[client_hash] = timestamps
            result = self._remaining_locked(client_hash, now)
            self._reservations[reservation_id] = result
            return QuotaReservation(remaining=result, created=True)

    def _remaining_locked(self, client_hash: str, now: datetime) -> QuotaRemaining:
        day = now.astimezone(UTC).strftime("%Y-%m-%d")
        cutoff = now.timestamp() - 60.0
        recent = [item for item in self._minute.get(client_hash, []) if item > cutoff]
        self._minute[client_hash] = recent
        return QuotaRemaining(
            minute=self._minute_limit - len(recent),
            client_day=self._client_limit - self._client.get((client_hash, day), 0),
            global_day=self._global_limit - self._global.get(day, 0),
        )


class FirestoreQuotaStore:
    """Reserve all public quotas and request idempotency in one transaction."""

    def __init__(self, config: Config, *, client: object | None = None) -> None:
        try:
            from google.cloud import firestore
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise AbuseServiceError(
                "Firestore quota support is unavailable in this deployment."
            ) from exc
        self._firestore = firestore
        self._client = client or firestore.Client(
            project=config.firestore_project_id,
            database=config.firestore_database,
        )
        self._minute_limit = config.public_minute_limit
        self._client_limit = config.public_client_daily_limit
        self._global_limit = config.public_global_daily_limit

    def remaining(self, client_hash: str, now: datetime) -> QuotaRemaining:
        try:
            day = now.astimezone(UTC).strftime("%Y-%m-%d")
            global_snapshot = (
                self._client.collection("demo_quota_global").document(day).get()
            )
            client_snapshot = (
                self._client.collection("demo_quota_clients")
                .document(f"{day}_{client_hash}")
                .get()
            )
            minute_snapshot = (
                self._client.collection("demo_quota_minutes")
                .document(client_hash)
                .get()
            )
            recent = _recent_timestamps(
                (minute_snapshot.to_dict() or {}).get("timestamps", [])
                if minute_snapshot.exists
                else [],
                now,
            )
            return QuotaRemaining(
                minute=self._minute_limit - len(recent),
                client_day=self._client_limit - _snapshot_count(client_snapshot),
                global_day=self._global_limit - _snapshot_count(global_snapshot),
            )
        except (DemoQuotaError, AbuseServiceError):
            raise
        except Exception as exc:
            raise AbuseServiceError(
                "The public quota service is temporarily unavailable."
            ) from exc

    def lookup(self, reservation_id: str) -> QuotaRemaining | None:
        try:
            snapshot = (
                self._client.collection("demo_quota_reservations")
                .document(hashlib.sha256(reservation_id.encode("utf-8")).hexdigest())
                .get()
            )
            if not snapshot.exists:
                return None
            data = snapshot.to_dict() or {}
            return QuotaRemaining(
                minute=int(data["minute"]),
                client_day=int(data["client_day"]),
                global_day=int(data["global_day"]),
            )
        except Exception as exc:
            raise AbuseServiceError(
                "The public quota service is temporarily unavailable."
            ) from exc

    def reserve(
        self,
        client_hash: str,
        reservation_id: str,
        now: datetime,
    ) -> QuotaReservation:
        firestore = self._firestore
        transaction = self._client.transaction()
        day = now.astimezone(UTC).strftime("%Y-%m-%d")
        global_ref = self._client.collection("demo_quota_global").document(day)
        client_ref = self._client.collection("demo_quota_clients").document(
            f"{day}_{client_hash}"
        )
        minute_ref = self._client.collection("demo_quota_minutes").document(client_hash)
        reservation_ref = self._client.collection("demo_quota_reservations").document(
            hashlib.sha256(reservation_id.encode("utf-8")).hexdigest()
        )

        @firestore.transactional
        def _reserve(current_transaction: object) -> QuotaReservation:
            snapshots = {
                snapshot.reference.path: snapshot
                for snapshot in current_transaction.get_all(  # type: ignore[attr-defined]
                    [reservation_ref, global_ref, client_ref, minute_ref]
                )
            }
            reservation_snapshot = snapshots[reservation_ref.path]
            global_snapshot = snapshots[global_ref.path]
            client_snapshot = snapshots[client_ref.path]
            minute_snapshot = snapshots[minute_ref.path]
            if reservation_snapshot.exists:
                data = reservation_snapshot.to_dict() or {}
                return QuotaReservation(
                    remaining=QuotaRemaining(
                        minute=int(data["minute"]),
                        client_day=int(data["client_day"]),
                        global_day=int(data["global_day"]),
                    ),
                    created=False,
                )
            global_count = _snapshot_count(global_snapshot)
            client_count = _snapshot_count(client_snapshot)
            minute_data = minute_snapshot.to_dict() or {}
            recent = _recent_timestamps(minute_data.get("timestamps", []), now)
            current = QuotaRemaining(
                minute=self._minute_limit - len(recent),
                client_day=self._client_limit - client_count,
                global_day=self._global_limit - global_count,
            )
            if min(current.minute, current.client_day, current.global_day) <= 0:
                raise DemoQuotaError(
                    "The public demo ask limit has been reached.",
                    remaining=current,
                )
            recent.append(now.timestamp())
            remaining = QuotaRemaining(
                minute=current.minute - 1,
                client_day=current.client_day - 1,
                global_day=current.global_day - 1,
            )
            current_transaction.set(  # type: ignore[attr-defined]
                global_ref, {"count": global_count + 1, "day": day}
            )
            current_transaction.set(  # type: ignore[attr-defined]
                client_ref,
                {"count": client_count + 1, "day": day, "client": client_hash},
            )
            current_transaction.set(  # type: ignore[attr-defined]
                minute_ref, {"timestamps": recent, "client": client_hash}
            )
            current_transaction.set(  # type: ignore[attr-defined]
                reservation_ref,
                {
                    **remaining.as_dict(),
                    "client": client_hash,
                    "created_at": now.timestamp(),
                },
            )
            return QuotaReservation(remaining=remaining, created=True)

        try:
            return _reserve(transaction)
        except DemoQuotaError:
            raise
        except Exception as exc:
            raise AbuseServiceError(
                "The public quota service is temporarily unavailable."
            ) from exc


def _snapshot_count(snapshot: object) -> int:
    if not getattr(snapshot, "exists", False):
        return 0
    data = snapshot.to_dict() or {}  # type: ignore[attr-defined]
    value = data.get("count", 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _recent_timestamps(values: object, now: datetime) -> list[float]:
    if not isinstance(values, list):
        return []
    cutoff = now.timestamp() - 60.0
    return [
        float(item)
        for item in values
        if isinstance(item, (int, float)) and float(item) > cutoff
    ]
