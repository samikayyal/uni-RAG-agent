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
from ..indexing import BuiltEmbeddingModel, SemanticSearchError, build_embedding_model
from ..indexing.profiles import physical_collection_name, resolve_embedding_profile
from ..indexing.vector import build_chroma_client
from ..retrieval.planner import build_chat_model
from ..search_contracts import LOGICAL_INDEX_TO_SOURCE_TYPE


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


@dataclass(frozen=True)
class EmbeddingRuntime:
    built: BuiltEmbeddingModel
    chroma_client: object
    encoding_lock: object | None


class EmbeddingRegistry:
    """Reuse embedding providers and one Chroma client for the web process."""

    def __init__(
        self,
        *,
        embedding_builder: Callable[..., BuiltEmbeddingModel] = build_embedding_model,
        chroma_builder: Callable[..., object] = build_chroma_client,
    ) -> None:
        self._embedding_builder = embedding_builder
        self._chroma_builder = chroma_builder
        # State changes are deliberately tiny critical sections.  Building a
        # local transformer can take minutes and must not make an unrelated
        # hosted profile wait behind it.
        self._state_lock = RLock()
        self._models: dict[str, BuiltEmbeddingModel] = {}
        self._errors: dict[str, Exception] = {}
        self._profile_locks: dict[str, RLock] = {}
        self._profile_build_locks: dict[str, RLock] = {}
        self._chroma_build_lock = RLock()
        self._chroma_client: object | None = None
        self._chroma_path: object | None = None
        self._chroma_error: Exception | None = None
        self._default_vector_index_ready = False

    def runtime(
        self,
        config: Config,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> EmbeddingRuntime:
        profile = resolve_embedding_profile(
            config, config.embedding_model, error=SemanticSearchError
        )
        with self._state_lock:
            loading = (
                progress_callback is not None
                and profile.provider == "huggingface"
                and profile.model_name not in self._models
                and profile.model_name not in self._errors
            )
        if loading:
            progress_callback("loading_embedding_model")
        built = self._model(config, profile.model_name)
        client = self._client(config)
        encoding_lock = self._encoding_lock(profile.model_name, profile.provider)
        return EmbeddingRuntime(
            built=built,
            chroma_client=client,
            encoding_lock=encoding_lock,
        )

    def warm_hosted(self, config: Config) -> None:
        """Build Chroma and check only the hosted serving default's vectors."""
        model_name = (
            config.public_default_embedding_model
            if config.public_demo_enabled
            else config.embedding_model
        )
        if not model_name:
            with self._state_lock:
                self._default_vector_index_ready = False
            return
        try:
            profile = resolve_embedding_profile(
                config, model_name, error=SemanticSearchError
            )
            client = self._client(config)
            ready = self._expected_indexes_ready(client, (profile.model_name,))
        except Exception:
            ready = False
        with self._state_lock:
            self._default_vector_index_ready = ready

    def hosted_ready(self) -> bool:
        with self._state_lock:
            return (
                self._chroma_client is not None
                and self._default_vector_index_ready
                and self._chroma_error is None
            )

    def prepare(self, config: Config, model_name: str, *, retry: bool = False) -> None:
        """Construct one explicitly selected local profile without touching Chroma."""
        profile = resolve_embedding_profile(
            config, model_name, error=SemanticSearchError
        )
        if profile.provider != "huggingface":
            raise SemanticSearchError("Only local embedding profiles can be prepared.")
        if retry:
            self.clear_failure(profile.model_name)
        self._model(config, profile.model_name)

    def profile_vector_index_ready(self, config: Config, model_name: str) -> bool:
        """Return whether a selected profile has at least one usable vector collection."""
        try:
            profile = resolve_embedding_profile(
                config, model_name, error=SemanticSearchError
            )
            return self._expected_indexes_ready(
                self._client(config), (profile.model_name,)
            )
        except Exception:
            return False

    def clear_failure(self, model_name: str | None = None) -> None:
        """Allow an operator/test to retry a corrected runtime dependency."""
        with self._state_lock:
            if model_name is None:
                self._errors.clear()
                self._chroma_error = None
                self._default_vector_index_ready = False
                return
            self._errors.pop(model_name, None)

    def _model(self, config: Config, model_name: str) -> BuiltEmbeddingModel:
        build_lock = self._profile_build_lock(model_name)
        with build_lock:
            with self._state_lock:
                previous_error = self._errors.get(model_name)
                if previous_error is not None:
                    raise previous_error
                existing = self._models.get(model_name)
                if existing is not None:
                    return existing
            try:
                built = self._embedding_builder(
                    config, model_name, error=SemanticSearchError
                )
            except Exception as exc:  # noqa: BLE001 - cached/surfaced on request
                with self._state_lock:
                    self._errors[model_name] = exc
                raise
            with self._state_lock:
                self._models[model_name] = built
            return built

    def _client(self, config: Config) -> object:
        path_identity = config.chroma_dir.resolve()
        with self._chroma_build_lock:
            with self._state_lock:
                if self._chroma_path != path_identity:
                    self._chroma_client = None
                    self._chroma_error = None
                    self._default_vector_index_ready = False
                    self._chroma_path = path_identity
                if self._chroma_error is not None:
                    raise self._chroma_error
                existing = self._chroma_client
            if existing is not None:
                return existing
            try:
                built = self._chroma_builder(config, error=SemanticSearchError)
            except Exception as exc:  # noqa: BLE001 - cached/surfaced on request
                with self._state_lock:
                    self._chroma_error = exc
                raise
            with self._state_lock:
                self._chroma_client = built
            return built

    def _profile_build_lock(self, model_name: str) -> RLock:
        with self._state_lock:
            return self._profile_build_locks.setdefault(model_name, RLock())

    def _encoding_lock(self, model_name: str, provider: str) -> object | None:
        if provider != "huggingface":
            return None
        with self._state_lock:
            return self._profile_locks.setdefault(model_name, RLock())

    @staticmethod
    def _expected_indexes_ready(client: object, model_names: tuple[str, ...]) -> bool:
        """Check local collection metadata only; never construct hosted providers."""
        try:
            existing = {
                getattr(item, "name", str(item))
                for item in client.list_collections()  # type: ignore[attr-defined]
            }
            for model_name in model_names:
                profile = resolve_embedding_profile(
                    None, model_name, error=SemanticSearchError
                )
                names = (
                    physical_collection_name(
                        logical_index,
                        provider=profile.provider,
                        model_name=profile.model_name,
                        dimension=profile.dimension,
                        metric=profile.metric,
                    )
                    for logical_index in LOGICAL_INDEX_TO_SOURCE_TYPE
                )
                if not any(
                    name in existing and int(client.get_collection(name).count()) > 0  # type: ignore[attr-defined]
                    for name in names
                ):
                    return False
        except Exception:
            return False
        return True


class PersistenceGate:
    """Make request cancellation atomic with answer persistence."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._cancelled = False
        self._cancel_reason: str | None = None
        self._committed = False
        self._search_run_id: int | None = None
        self._evidence_packet_id: int | None = None
        self._phase: str | None = None
        self._started_at = monotonic()
        self._phase_started_at: float | None = None
        self._completed_phases: list[dict[str, object]] = []

    @property
    def trace_ids(self) -> tuple[int | None, int | None]:
        with self._lock:
            return self._search_run_id, self._evidence_packet_id

    @property
    def cancel_reason(self) -> str | None:
        with self._lock:
            return self._cancel_reason

    def progress(self) -> dict[str, object]:
        """Return non-sensitive in-memory request progress for the web UI."""
        with self._lock:
            return {
                "phase": self._phase,
                "elapsed_seconds": monotonic() - self._started_at,
                "cancelled": self._cancelled,
            }

    def set_phase(self, phase: str) -> None:
        with self._lock:
            if not self._cancelled:
                now = monotonic()
                if (
                    self._phase is not None
                    and self._phase_started_at is not None
                    and phase != self._phase
                ):
                    self._completed_phases.append(
                        {
                            "phase": self._phase,
                            "duration_seconds": now - self._phase_started_at,
                        }
                    )
                if phase != self._phase:
                    self._phase_started_at = now
                self._phase = phase

    def audit_timing(self) -> dict[str, object]:
        """Return the complete in-process phase timeline for durable auditing."""
        with self._lock:
            now = monotonic()
            phases = [dict(item) for item in self._completed_phases]
            if self._phase is not None and self._phase_started_at is not None:
                phases.append(
                    {
                        "phase": self._phase,
                        "duration_seconds": now - self._phase_started_at,
                    }
                )
            return {
                "elapsed_seconds": now - self._started_at,
                "phases": phases,
            }

    def record_evidence(self, search_run_id: int, evidence_packet_id: int) -> None:
        with self._lock:
            self._search_run_id = search_run_id
            self._evidence_packet_id = evidence_packet_id

    def cancel(self, reason: str = "cancelled") -> bool:
        """Cancel before commit, returning false when commit already completed."""
        with self._lock:
            if self._committed:
                return False
            self._cancelled = True
            self._cancel_reason = reason
            return True

    def commit(self, action: Callable[[], None]) -> None:
        """Commit the transaction only when the response is still live."""
        with self._lock:
            if self._cancelled:
                raise AskCancelled("ask request timed out before answer persistence")
            action()
            self._committed = True


class ActiveAskRegistry:
    """Expose active request progress and cancellation without retaining history."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._gates: dict[str, tuple[PersistenceGate, str | None]] = {}

    def register(
        self, request_id: str, gate: PersistenceGate, owner: str | None = None
    ) -> bool:
        with self._lock:
            if request_id in self._gates:
                return False
            self._gates[request_id] = (gate, owner)
            return True

    def complete(self, request_id: str, gate: PersistenceGate) -> None:
        with self._lock:
            entry = self._gates.get(request_id)
            if entry is not None and entry[0] is gate:
                del self._gates[request_id]

    def progress(
        self, request_id: str, owner: str | None = None
    ) -> dict[str, object] | None:
        with self._lock:
            entry = self._gates.get(request_id)
        if entry is None or entry[1] != owner:
            return None
        return entry[0].progress()

    def cancel(self, request_id: str, owner: str | None = None) -> bool | None:
        with self._lock:
            entry = self._gates.get(request_id)
        if entry is None or entry[1] != owner:
            return None
        return entry[0].cancel()


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

    def has_live_session(self, session_id: str) -> bool:
        """Return whether planner context still exists without extending its TTL."""
        with self._lock:
            self._purge_expired(self._clock())
            return session_id in self._entries

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
        embedding_registry: EmbeddingRegistry | None = None,
    ) -> None:
        self._build_evidence = build_evidence
        self._generate_answer = generate_answer
        self._store_answer = store_answer
        self._registry = registry
        self._session_factory = session_factory
        self._enforce_model_config = enforce_model_config
        self._model_registry = model_registry
        self._embedding_registry = embedding_registry

    def ask(
        self,
        config: Config,
        query: str,
        session_id: str | None,
        gate: PersistenceGate,
    ) -> tuple[AnswerResult, Any, Any]:
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
    ) -> tuple[AnswerResult, Any, Any]:
        context = session.conversation_context
        planner_model = (
            self._model_registry.planner(config)
            if self._model_registry is not None
            else None
        )
        embedding_kwargs: dict[str, object] = {}
        if self._embedding_registry is not None:
            runtime = self._embedding_registry.runtime(
                config,
                progress_callback=gate.set_phase,
            )
            embedding_kwargs = {
                "built_embedding": runtime.built,
                "chroma_client": runtime.chroma_client,
            }
            if runtime.encoding_lock is not None:
                embedding_kwargs["encoding_lock"] = runtime.encoding_lock
        evidence_result = self._build_evidence(
            config,
            query,
            conversation_context=context,
            chat_model=planner_model,
            progress_callback=gate.set_phase,
            **embedding_kwargs,
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
        gate.set_phase("answer_generation")
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
        return completed, evidence_result.coverage, evidence_result.packet
