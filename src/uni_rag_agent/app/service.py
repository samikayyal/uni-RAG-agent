"""Thread-safe application orchestration and bounded session memory."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from threading import Lock, RLock
from time import monotonic
from typing import Any

from ..answering import AnswerResult, AnswerSession
from ..answering.providers import build_answer_chat_model
from ..config import Config, ConfigError
from ..retrieval.planner import build_chat_model


class AskCancelled(RuntimeError):
    """Raised inside a worker when its HTTP request has timed out."""


class SessionCapacityError(RuntimeError):
    """Raised when all bounded session slots are currently active."""


class ModelRegistry:
    """Cache the configured planner and answer models for one app process."""

    def __init__(
        self,
        *,
        planner_builder: Callable[[Config], object] = build_chat_model,
        answer_builder: Callable[[Config], object] = build_answer_chat_model,
    ) -> None:
        self._planner_builder = planner_builder
        self._answer_builder = answer_builder
        self._lock = RLock()
        self._identity: tuple[object, ...] | None = None
        self._planner: object | None = None
        self._answer: object | None = None
        self._planner_error: Exception | None = None
        self._answer_error: Exception | None = None

    def warm(self, config: Config) -> None:
        """Construct each configured model at most once per configuration."""
        with self._lock:
            identity = (
                config.llm_provider,
                config.llm_model,
                config.answer_llm_provider,
                config.answer_llm_model,
            )
            if identity == self._identity:
                return
            self._identity = identity
            self._planner = None
            self._answer = None
            self._planner_error = None
            self._answer_error = None
            if config.llm_provider and config.llm_model:
                try:
                    self._planner = self._planner_builder(config)
                except Exception as exc:  # noqa: BLE001 - report on request
                    self._planner_error = exc
            if config.answer_llm_provider and config.answer_llm_model:
                try:
                    self._answer = self._answer_builder(config)
                except Exception as exc:  # noqa: BLE001 - report on request
                    self._answer_error = exc

    def planner(self, config: Config) -> object | None:
        self.warm(config)
        with self._lock:
            if self._planner_error is not None:
                raise self._planner_error
            return self._planner

    def answer(self, config: Config) -> object | None:
        self.warm(config)
        with self._lock:
            if self._answer_error is not None:
                raise self._answer_error
            return self._answer


class PersistenceGate:
    """Make timeout cancellation atomic with answer persistence."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._cancelled = False
        self._committed = False
        self._search_run_id: int | None = None
        self._evidence_packet_id: int | None = None

    @property
    def trace_ids(self) -> tuple[int | None, int | None]:
        with self._lock:
            return self._search_run_id, self._evidence_packet_id

    def record_evidence(self, search_run_id: int, evidence_packet_id: int) -> None:
        with self._lock:
            self._search_run_id = search_run_id
            self._evidence_packet_id = evidence_packet_id

    def cancel(self) -> bool:
        """Cancel before commit, returning false when commit already completed."""
        with self._lock:
            if self._committed:
                return False
            self._cancelled = True
            return True

    def commit(self, action: Callable[[], None]) -> None:
        """Commit the transaction only when the response is still live."""
        with self._lock:
            if self._cancelled:
                raise AskCancelled("ask request timed out before answer persistence")
            action()
            self._committed = True


@dataclass
class _SessionEntry:
    session: AnswerSession
    request_lock: RLock
    last_used: float
    active: int = 0


class SessionRegistry:
    """LRU session registry with inactivity expiry and active-entry safety."""

    def __init__(
        self,
        *,
        max_sessions: int = 20,
        ttl_seconds: float = 7_200,
        clock: Callable[[], float] = monotonic,
        session_factory: Callable[[Config], AnswerSession] = AnswerSession,
    ) -> None:
        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._session_factory = session_factory
        self._lock = RLock()
        self._entries: OrderedDict[str, _SessionEntry] = OrderedDict()

    @contextmanager
    def checkout(self, session_id: str, config: Config) -> Iterator[AnswerSession]:
        entry = self._acquire_entry(session_id, config)
        try:
            with entry.request_lock:
                yield entry.session
        finally:
            with self._lock:
                current = self._entries.get(session_id)
                if current is entry:
                    entry.active -= 1
                    entry.last_used = self._clock()
                    self._entries.move_to_end(session_id)

    def _acquire_entry(self, session_id: str, config: Config) -> _SessionEntry:
        with self._lock:
            now = self._clock()
            self._purge_expired(now)
            entry = self._entries.get(session_id)
            if entry is None:
                self._make_room()
                entry = _SessionEntry(
                    session=self._session_factory(config),
                    request_lock=RLock(),
                    last_used=now,
                )
                self._entries[session_id] = entry
            entry.active += 1
            entry.last_used = now
            self._entries.move_to_end(session_id)
            return entry

    def _purge_expired(self, now: float) -> None:
        expired = [
            session_id
            for session_id, entry in self._entries.items()
            if entry.active == 0 and now - entry.last_used >= self._ttl_seconds
        ]
        for session_id in expired:
            del self._entries[session_id]

    def _make_room(self) -> None:
        if len(self._entries) < self._max_sessions:
            return
        for session_id, entry in self._entries.items():
            if entry.active == 0:
                del self._entries[session_id]
                return
        raise SessionCapacityError("all in-process session slots are active")


class AskOrchestrator:
    """Run the persisted ask phases without duplicating retrieval internals."""

    def __init__(
        self,
        *,
        build_evidence: Callable[..., Any],
        generate_answer: Callable[..., AnswerResult],
        store_answer: Callable[..., int],
        registry: SessionRegistry,
        session_factory: Callable[[Config], AnswerSession] = AnswerSession,
        enforce_model_config: bool = False,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self._build_evidence = build_evidence
        self._generate_answer = generate_answer
        self._store_answer = store_answer
        self._registry = registry
        self._session_factory = session_factory
        self._enforce_model_config = enforce_model_config
        self._model_registry = model_registry

    def ask(
        self,
        config: Config,
        query: str,
        session_id: str | None,
        gate: PersistenceGate,
    ) -> tuple[AnswerResult, Any]:
        if self._enforce_model_config and (
            config.llm_provider is None
            or config.llm_model is None
            or config.embedding_model is None
        ):
            raise ConfigError(
                "HTTP ask requires planner provider/model and an embedding model"
            )
        if session_id is None:
            session = self._session_factory(config)
            return self._ask_with_session(config, query, session, gate)
        with self._registry.checkout(session_id, config) as session:
            return self._ask_with_session(config, query, session, gate)

    def _ask_with_session(
        self,
        config: Config,
        query: str,
        session: AnswerSession,
        gate: PersistenceGate,
    ) -> tuple[AnswerResult, Any]:
        context = session.conversation_context
        planner_model = (
            self._model_registry.planner(config)
            if self._model_registry is not None
            else None
        )
        evidence_result = self._build_evidence(
            config,
            query,
            conversation_context=context,
            chat_model=planner_model,
        )
        gate.record_evidence(
            evidence_result.search_run_id,
            evidence_result.evidence_packet_id,
        )
        if (
            self._enforce_model_config
            and evidence_result.packet.evidence
            and (config.answer_llm_provider is None or config.answer_llm_model is None)
        ):
            raise ConfigError(
                "Non-empty HTTP evidence requires answer provider/model configuration"
            )
        answer_model = (
            self._model_registry.answer(config)
            if self._model_registry is not None and evidence_result.packet.evidence
            else None
        )
        answer = self._generate_answer(
            evidence_result.packet,
            conversation_context=context,
            config=config,
            chat_model=answer_model,
        )

        answer_id = self._store_answer(
            evidence_result.evidence_packet_id,
            answer,
            config=config,
            commit_guard=gate.commit,
        )
        session.record_complete_turn(query, answer.answer_text)
        completed = replace(
            answer,
            answer_id=answer_id,
            evidence_packet_id=evidence_result.evidence_packet_id,
            search_run_id=evidence_result.search_run_id,
        )
        return completed, evidence_result.coverage
