"""FastAPI routes and safe public response projections."""

from __future__ import annotations

import asyncio
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Annotated, Any

from fastapi import FastAPI, Request
from fastapi import Path as ApiPath
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..answering import (
    AnswerGenerationError,
    AnswerResult,
    AnswerSession,
    answer_body,
    answer_status,
    generate_answer,
    load_answer,
    store_answer,
)
from ..config import Config, ConfigError, load_config, validate_config
from ..indexing.profiles import resolve_embedding_profile
from ..retrieval import (
    EvidenceError,
    QueryPlanningError,
    RetrievalError,
    build_evidence,
    explain_search_coverage,
    load_evidence_packet,
)
from ..storage import StorageError, check_storage
from .public_demo import (
    AbuseServiceError,
    CloudflareTurnstileVerifier,
    DemoAuthorizationError,
    DemoQuotaError,
    FirestoreQuotaStore,
    QuotaRemaining,
    SignedDemoTokenManager,
    hash_client_address,
    resolve_client_address,
)
from .service import (
    ActiveAskRegistry,
    AskCancelled,
    AskOrchestrator,
    EmbeddingRegistry,
    ModelRegistry,
    PersistenceGate,
    SessionCapacityError,
    SessionRegistry,
    load_index_status,
)
from .settings import (
    SettingsError,
    WebSettingsStore,
    apply_public_settings,
    describe_public_settings,
    describe_settings,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
PositiveId = Annotated[int, ApiPath(gt=0)]
SessionId = Annotated[str, ApiPath(pattern=r"^[A-Za-z0-9_-]{1,128}$")]
RequestId = Annotated[str, ApiPath(pattern=r"^[A-Za-z0-9_-]{1,128}$")]


@dataclass(frozen=True)
class AppServices:
    build_evidence: Any = build_evidence
    generate_answer: Any = generate_answer
    store_answer: Any = store_answer
    load_answer: Any = load_answer
    load_evidence_packet: Any = load_evidence_packet
    explain_search_coverage: Any = explain_search_coverage
    session_factory: Any = AnswerSession


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=10_000)
    session_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{1,128}$",
    )
    request_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{1,128}$",
    )
    retrieval_settings: "PublicRetrievalSettings | None" = None


class PublicRetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_model: str | None = None
    keyword_top_k: int | None = None
    semantic_top_k: int | None = None
    metadata_top_k: int | None = None
    final_top_k: int | None = None
    rrf_k: int | None = None
    semantic_query_limit: int | None = None
    filename_fuzzy_threshold: int | None = None
    path_fuzzy_threshold: int | None = None
    evidence_max_tokens: int | None = None
    query_plan_min_confidence: float | None = None


class DemoSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    turnstile_token: str = Field(min_length=1, max_length=4_096)


class SettingsUpdateRequest(BaseModel):
    """Web-adjustable settings only; sensitive fields are rejected as extras.

    A field set to ``null`` clears its override so the environment default
    applies again; omitted fields are left unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    embedding_model: str | None = None
    keyword_top_k: int | None = None
    semantic_top_k: int | None = None
    metadata_top_k: int | None = None
    final_top_k: int | None = None
    rrf_k: int | None = None
    semantic_query_limit: int | None = None
    filename_fuzzy_threshold: int | None = None
    path_fuzzy_threshold: int | None = None
    evidence_max_tokens: int | None = None
    query_plan_min_confidence: float | None = None


class EmbeddingProfilePrepareRequest(BaseModel):
    """An explicit browser request to load a selected local profile."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    embedding_model: str = Field(min_length=1, max_length=256)


class ApiError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def create_app(
    *,
    config_loader: Any = load_config,
    services: AppServices | None = None,
    clock: Any = None,
    session_registry: SessionRegistry | None = None,
    enforce_model_config: bool | None = None,
    model_registry: ModelRegistry | None = None,
    ask_registry: ActiveAskRegistry | None = None,
    warm_models: bool = True,
    settings_store: WebSettingsStore | None = None,
    embedding_registry: EmbeddingRegistry | None = None,
    token_manager: Any = None,
    turnstile_verifier: Any = None,
    quota_store: Any = None,
    client_resolver: Any = resolve_client_address,
    now: Any = None,
) -> FastAPI:
    """Create an app with injectable services and cached model instances."""
    resolved = services or AppServices()
    web_settings = settings_store or WebSettingsStore()
    resolved_model_registry = (
        model_registry
        if model_registry is not None
        else ModelRegistry()
        if services is None
        else None
    )
    resolved_embedding_registry = (
        embedding_registry
        if embedding_registry is not None
        else EmbeddingRegistry()
        if services is None
        else None
    )
    registry_kwargs: dict[str, Any] = {"session_factory": resolved.session_factory}
    if clock is not None:
        registry_kwargs["clock"] = clock
    registry = session_registry or SessionRegistry(
        max_sessions=20,
        ttl_seconds=7_200,
        **registry_kwargs,
    )
    orchestrator = AskOrchestrator(
        build_evidence=resolved.build_evidence,
        generate_answer=resolved.generate_answer,
        store_answer=resolved.store_answer,
        registry=registry,
        session_factory=resolved.session_factory,
        enforce_model_config=(services is None)
        if enforce_model_config is None
        else enforce_model_config,
        model_registry=resolved_model_registry,
        embedding_registry=resolved_embedding_registry,
    )
    active_asks = ask_registry or ActiveAskRegistry()
    runtime_lock = Lock()
    runtime: dict[str, Any] = {}
    clock = now or (lambda: datetime.now(UTC))

    def _demo_runtime(config: Config) -> dict[str, Any]:
        with runtime_lock:
            if "token_manager" not in runtime:
                runtime["token_manager"] = token_manager or SignedDemoTokenManager(
                    config.demo_token_signing_secret or "",
                    ttl_seconds=config.demo_token_ttl_seconds,
                    clock=clock,
                )
                runtime["turnstile_verifier"] = (
                    turnstile_verifier
                    or CloudflareTurnstileVerifier(config.turnstile_secret_key or "")
                )
                runtime["quota_store"] = quota_store or FirestoreQuotaStore(config)
                runtime["semaphore"] = asyncio.Semaphore(config.public_ask_capacity)
                runtime["session_owners"] = {}
            return runtime

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        config = config_loader()
        validate_config(config)
        if config.public_demo_enabled:
            _demo_runtime(config)
        if config.hosted_mode and resolved_embedding_registry is not None:
            try:
                resolved_embedding_registry.warm_hosted(config)
            except Exception:
                # The readiness route stays closed and exposes no provider detail.
                pass
        if warm_models and resolved_model_registry is not None:
            try:
                resolved_model_registry.warm(config)
            except Exception:
                # Keep health and diagnostics available if startup config or
                # an optional provider is unavailable.
                pass
        yield

    app = FastAPI(
        title="Uni RAG Agent",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.exception_handler(ApiError)
    async def handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _: Request, __: RequestValidationError
    ) -> JSONResponse:
        return _error_response(422, "validation_error", "The request is invalid.")

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        _: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        if exc.status_code == 404:
            return _error_response(
                404,
                "not_found",
                "The requested resource does not exist.",
            )
        if exc.status_code == 405:
            return _error_response(
                405,
                "method_not_allowed",
                "The requested method is not allowed.",
            )
        return _error_response(
            exc.status_code,
            "http_error",
            "The request could not be completed.",
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, __: Exception) -> JSONResponse:
        return _error_response(500, "internal_error", "An internal error occurred.")

    def _base_config() -> Config:
        return _load_config(config_loader)

    def _effective_config() -> Config:
        """Load env configuration with stored web settings layered on top.

        Reads the merged .env and the settings-override file, so async
        handlers always call it through ``asyncio.to_thread``.
        """
        return web_settings.apply(_base_config())

    def _request_config(
        base: Config, retrieval_settings: PublicRetrievalSettings | None
    ) -> Config:
        if not base.public_demo_enabled:
            if retrieval_settings is not None:
                raise ApiError(
                    422,
                    "settings_validation_error",
                    "Request-scoped retrieval settings are available only in public mode.",
                )
            return web_settings.apply(base)
        values = (
            retrieval_settings.model_dump(exclude_unset=True)
            if retrieval_settings is not None
            else {}
        )
        try:
            return apply_public_settings(base, values)
        except SettingsError as exc:
            raise ApiError(422, "settings_validation_error", str(exc)) from exc

    def _client_hash(request: Request, config: Config) -> str:
        try:
            address = client_resolver(request, hosted_mode=config.hosted_mode)
        except AbuseServiceError:
            raise
        except Exception as exc:
            raise AbuseServiceError(
                "The public client address could not be verified."
            ) from exc
        return hash_client_address(
            address,
            config.demo_token_signing_secret or "",
        )

    def _authorize(request: Request, config: Config) -> tuple[Any, str]:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.casefold() != "bearer" or not token.strip():
            raise DemoAuthorizationError("A valid demo token is required.")
        client_hash = _client_hash(request, config)
        claims = _demo_runtime(config)["token_manager"].verify(
            token.strip(), client_hash
        )
        return claims, client_hash

    def _claim_public_owner(session_id: str, claims: Any) -> tuple[bool, bool]:
        with runtime_lock:
            owners = runtime.setdefault("session_owners", {})
            current_time = int(clock().timestamp())
            for expired_id, (_, expires_at) in tuple(owners.items()):
                if expires_at <= current_time:
                    del owners[expired_id]
            owner = owners.get(session_id)
            if owner is None:
                owners[session_id] = (claims.nonce, claims.expires_at)
                return True, True
            return owner[0] == claims.nonce, False

    def _release_public_owner_claim(session_id: str | None, claims: Any) -> None:
        if session_id is None:
            return
        with runtime_lock:
            owners = runtime.get("session_owners", {})
            if owners.get(session_id) == (claims.nonce, claims.expires_at):
                del owners[session_id]

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        config = await asyncio.to_thread(_base_config)
        storage = await asyncio.to_thread(check_storage, config)
        default_vector_index_ready = not config.hosted_mode or (
            resolved_embedding_registry is not None
            and resolved_embedding_registry.hosted_ready()
        )
        is_ready = storage.ok and default_vector_index_ready
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={
                "status": "ready" if is_ready else "not_ready",
                "storage_ready": storage.ok,
                "default_vector_index_ready": default_vector_index_ready,
            },
        )

    @app.get("/config")
    async def config_view() -> dict[str, object]:
        config = await asyncio.to_thread(_base_config)
        effective = (
            apply_public_settings(config, {})
            if config.public_demo_enabled
            else await asyncio.to_thread(web_settings.apply, config)
        )
        return _public_config(effective)

    @app.get("/api/index-status")
    async def index_status() -> dict[str, object]:
        """Report indexed course/file/chunk counts for the browser index panel."""
        config = await asyncio.to_thread(_base_config)
        if config.public_demo_enabled:
            raise ApiError(404, "not_found", "The requested resource does not exist.")
        return await asyncio.to_thread(load_index_status, config)

    @app.get("/api/settings")
    async def settings_view() -> dict[str, object]:
        config = await asyncio.to_thread(_base_config)
        if config.public_demo_enabled:
            return describe_public_settings(config)
        overrides = await asyncio.to_thread(web_settings.load_overrides, config)
        return describe_settings(config, overrides)

    @app.put("/api/settings")
    async def settings_update(payload: SettingsUpdateRequest) -> dict[str, object]:
        config = await asyncio.to_thread(_base_config)
        if config.public_demo_enabled:
            raise ApiError(404, "not_found", "The requested resource does not exist.")
        changes = payload.model_dump(exclude_unset=True)
        try:
            overrides = await asyncio.to_thread(web_settings.update, config, changes)
        except SettingsError as exc:
            raise ApiError(422, "settings_validation_error", str(exc)) from exc
        return describe_settings(config, overrides)

    @app.post("/api/embedding-profiles/prepare")
    async def prepare_embedding_profile(
        payload: EmbeddingProfilePrepareRequest, request: Request
    ) -> dict[str, str]:
        """Load one selected local profile before browser settings are persisted."""
        config = await asyncio.to_thread(_base_config)
        if config.public_demo_enabled:
            try:
                _authorize(request, config)
            except DemoAuthorizationError as exc:
                raise ApiError(401, "demo_authorization_error", str(exc)) from exc
            except AbuseServiceError as exc:
                raise ApiError(503, "abuse_service_unavailable", str(exc)) from exc
        try:
            profile = resolve_embedding_profile(
                config, payload.embedding_model, error=SettingsError
            )
        except SettingsError as exc:
            raise ApiError(
                422,
                "embedding_profile_invalid",
                "The embedding profile cannot be prepared.",
            ) from exc
        if profile.provider != "huggingface" or (
            config.public_demo_enabled
            and profile.model_name not in config.public_embedding_profiles
        ):
            raise ApiError(
                422,
                "embedding_profile_invalid",
                "The embedding profile cannot be prepared.",
            )
        if resolved_embedding_registry is None:
            raise ApiError(
                502,
                "embedding_profile_unavailable",
                "The embedding profile is currently unavailable.",
            )
        try:
            await asyncio.to_thread(
                resolved_embedding_registry.prepare,
                config,
                profile.model_name,
                retry=True,
            )
        except Exception as exc:  # noqa: BLE001 - never disclose model/runtime details
            raise ApiError(
                502,
                "embedding_profile_unavailable",
                "The embedding profile is currently unavailable.",
            ) from exc
        return {"embedding_model": profile.model_name, "status": "ready"}

    @app.post("/api/demo/session")
    async def demo_session(
        payload: DemoSessionRequest, request: Request
    ) -> dict[str, object]:
        config = await asyncio.to_thread(_base_config)
        if not config.public_demo_enabled:
            raise ApiError(404, "not_found", "The requested resource does not exist.")
        try:
            client_hash = _client_hash(request, config)
            components = _demo_runtime(config)
            verified = await asyncio.to_thread(
                components["turnstile_verifier"].verify,
                payload.turnstile_token,
            )
            if not verified:
                raise DemoAuthorizationError("Turnstile verification failed.")
            token, claims = components["token_manager"].issue(client_hash)
            remaining = await asyncio.to_thread(
                components["quota_store"].remaining,
                client_hash,
                clock(),
            )
        except DemoAuthorizationError as exc:
            raise ApiError(401, "demo_authorization_error", str(exc)) from exc
        except AbuseServiceError as exc:
            raise ApiError(503, "abuse_service_unavailable", str(exc)) from exc
        return {
            "demo_token": token,
            "expires_at": claims.expires_at,
            "remaining": remaining.as_dict(),
        }

    @app.post("/api/ask")
    async def ask(payload: AskRequest, request: Request) -> dict[str, object]:
        base_config = await asyncio.to_thread(_base_config)
        config = await asyncio.to_thread(
            _request_config, base_config, payload.retrieval_settings
        )
        public = config.public_demo_enabled
        claims: Any = None
        client_hash: str | None = None
        remaining: QuotaRemaining | None = None
        components: dict[str, Any] | None = None
        request_id = payload.request_id or (
            secrets.token_urlsafe(18) if public else None
        )
        reservation_id: str | None = None
        if public:
            if len(payload.query) > config.public_query_max_chars:
                raise ApiError(
                    422,
                    "settings_validation_error",
                    f"query must contain at most {config.public_query_max_chars} characters.",
                )
            try:
                claims, client_hash = _authorize(request, config)
            except DemoAuthorizationError as exc:
                raise ApiError(401, "demo_authorization_error", str(exc)) from exc
            except AbuseServiceError as exc:
                raise ApiError(503, "abuse_service_unavailable", str(exc)) from exc
            components = _demo_runtime(config)
            reservation_id = f"{claims.nonce}:{request_id}"
            try:
                existing_reservation = await asyncio.to_thread(
                    components["quota_store"].lookup,
                    reservation_id,
                )
            except AbuseServiceError as exc:
                raise ApiError(503, "abuse_service_unavailable", str(exc)) from exc
            if existing_reservation is not None:
                raise ApiError(
                    409,
                    "request_already_accepted",
                    "This public demo request id was already accepted.",
                )
            try:
                await asyncio.wait_for(
                    components["semaphore"].acquire(),
                    timeout=config.public_capacity_wait_seconds,
                )
            except TimeoutError as exc:
                raise ApiError(
                    503,
                    "ask_capacity_busy",
                    "The public demo is busy. Please try again shortly.",
                ) from exc

        owner_claimed = False
        if public and payload.session_id is not None:
            owner_allowed, owner_claimed = _claim_public_owner(
                payload.session_id, claims
            )
            if not owner_allowed:
                components["semaphore"].release()
                raise ApiError(
                    404, "not_found", "The requested resource does not exist."
                )

        gate = PersistenceGate()
        owner = claims.nonce if public else None
        if request_id is not None and not active_asks.register(request_id, gate, owner):
            if components is not None:
                components["semaphore"].release()
                if owner_claimed:
                    _release_public_owner_claim(payload.session_id, claims)
            raise ApiError(
                409,
                "request_in_progress",
                "An ask request with this request id is already active.",
            )
        if public:
            try:
                reservation = await asyncio.to_thread(
                    components["quota_store"].reserve,
                    client_hash,
                    reservation_id,
                    clock(),
                )
                remaining = reservation.remaining
                if not reservation.created:
                    if request_id is not None:
                        active_asks.complete(request_id, gate)
                    components["semaphore"].release()
                    if owner_claimed:
                        _release_public_owner_claim(payload.session_id, claims)
                    raise ApiError(
                        409,
                        "request_already_accepted",
                        "This public demo request id was already accepted.",
                    )
            except DemoQuotaError as exc:
                if request_id is not None:
                    active_asks.complete(request_id, gate)
                components["semaphore"].release()
                if owner_claimed:
                    _release_public_owner_claim(payload.session_id, claims)
                raise ApiError(
                    429,
                    "demo_quota_exhausted",
                    "The public demo ask limit has been reached.",
                ) from exc
            except AbuseServiceError as exc:
                if request_id is not None:
                    active_asks.complete(request_id, gate)
                components["semaphore"].release()
                if owner_claimed:
                    _release_public_owner_claim(payload.session_id, claims)
                raise ApiError(503, "abuse_service_unavailable", str(exc)) from exc
        task = asyncio.create_task(
            asyncio.to_thread(
                orchestrator.ask,
                config,
                payload.query,
                payload.session_id,
                gate,
            )
        )
        if components is not None:
            task.add_done_callback(lambda _: components["semaphore"].release())
        if request_id is not None:
            task.add_done_callback(
                lambda _: active_asks.complete(request_id or "", gate)
            )
        try:
            answer, coverage, packet = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=config.ask_timeout_seconds,
            )
        except TimeoutError:
            if await asyncio.to_thread(gate.cancel, "timeout"):
                task.add_done_callback(_consume_task_exception)
                _, packet_id = gate.trace_ids
                message = "The ask request exceeded its configured timeout."
                if packet_id is not None:
                    message = (
                        f"The ask request timed out after evidence packet {packet_id} "
                        "was stored; the packet remains available."
                    )
                raise ApiError(
                    504,
                    "ask_timeout",
                    message,
                )
            answer, coverage, packet = await task
        except AskCancelled:
            if gate.cancel_reason == "cancelled":
                raise ApiError(499, "ask_cancelled", "The ask request was cancelled.")
            raise ApiError(504, "ask_timeout", "The ask request timed out.")
        except SessionCapacityError:
            raise ApiError(
                503,
                "session_capacity",
                "All in-process session slots are currently active.",
            )
        except Exception as exc:
            raise _domain_error(exc, lookup=False, trace_ids=gate.trace_ids) from exc
        response = _public_answer(
            answer,
            coverage,
            packet=packet if public else None,
        )
        if public:
            response["request_id"] = request_id
            response["remaining"] = remaining.as_dict() if remaining else {}
        return response

    @app.get("/api/asks/{request_id}/progress")
    async def ask_progress(
        request_id: RequestId, request: Request
    ) -> dict[str, object]:
        config = await asyncio.to_thread(_base_config)
        owner = None
        if config.public_demo_enabled:
            try:
                owner = _authorize(request, config)[0].nonce
            except DemoAuthorizationError as exc:
                raise ApiError(401, "demo_authorization_error", str(exc)) from exc
            except AbuseServiceError as exc:
                raise ApiError(503, "abuse_service_unavailable", str(exc)) from exc
        progress = active_asks.progress(request_id, owner)
        if progress is None:
            raise ApiError(404, "not_found", "The requested resource does not exist.")
        return {"request_id": request_id, **progress}

    @app.post("/api/asks/{request_id}/cancel")
    async def cancel_ask(request_id: RequestId, request: Request) -> dict[str, object]:
        config = await asyncio.to_thread(_base_config)
        owner = None
        if config.public_demo_enabled:
            try:
                owner = _authorize(request, config)[0].nonce
            except DemoAuthorizationError as exc:
                raise ApiError(401, "demo_authorization_error", str(exc)) from exc
            except AbuseServiceError as exc:
                raise ApiError(503, "abuse_service_unavailable", str(exc)) from exc
        cancelled = active_asks.cancel(request_id, owner)
        if cancelled is None:
            raise ApiError(404, "not_found", "The requested resource does not exist.")
        return {"request_id": request_id, "cancelled": cancelled}

    @app.get("/api/sessions/{session_id}")
    async def session_status(
        session_id: SessionId, request: Request
    ) -> dict[str, object]:
        config = await asyncio.to_thread(_base_config)
        if config.public_demo_enabled:
            try:
                claims, _ = _authorize(request, config)
            except DemoAuthorizationError as exc:
                raise ApiError(401, "demo_authorization_error", str(exc)) from exc
            except AbuseServiceError as exc:
                raise ApiError(503, "abuse_service_unavailable", str(exc)) from exc
            with runtime_lock:
                owner = runtime.get("session_owners", {}).get(session_id)
                if owner is None or owner[0] != claims.nonce:
                    return {"session_id": session_id, "live": False}
        return {
            "session_id": session_id,
            "live": await asyncio.to_thread(registry.has_live_session, session_id),
        }

    @app.get("/api/search-runs/{search_run_id}/coverage")
    async def coverage(search_run_id: PositiveId) -> dict[str, object]:
        base = await asyncio.to_thread(_base_config)
        if base.public_demo_enabled:
            raise ApiError(404, "not_found", "The requested resource does not exist.")
        config = await asyncio.to_thread(web_settings.apply, base)
        try:
            result = await asyncio.to_thread(
                resolved.explain_search_coverage,
                config,
                search_run_id,
            )
        except Exception as exc:
            raise _domain_error(exc, lookup=True) from exc
        return result.as_safe_dict()

    @app.get("/api/evidence-packets/{evidence_packet_id}")
    async def evidence_packet(evidence_packet_id: PositiveId) -> dict[str, object]:
        base = await asyncio.to_thread(_base_config)
        if base.public_demo_enabled:
            raise ApiError(404, "not_found", "The requested resource does not exist.")
        config = await asyncio.to_thread(web_settings.apply, base)
        try:
            packet = await asyncio.to_thread(
                resolved.load_evidence_packet,
                config,
                evidence_packet_id,
            )
        except Exception as exc:
            raise _domain_error(exc, lookup=True) from exc
        return packet.as_safe_dict()

    @app.get("/api/answers/{answer_id}")
    async def answer(answer_id: PositiveId) -> dict[str, object]:
        base = await asyncio.to_thread(_base_config)
        if base.public_demo_enabled:
            raise ApiError(404, "not_found", "The requested resource does not exist.")
        config = await asyncio.to_thread(web_settings.apply, base)
        try:
            loaded = await asyncio.to_thread(resolved.load_answer, config, answer_id)
            packet = await asyncio.to_thread(
                resolved.load_evidence_packet,
                config,
                loaded.evidence_packet_id,
            )
        except Exception as exc:
            raise _domain_error(exc, lookup=True) from exc
        completed = AnswerResult(
            answer_text=loaded.answer_text,
            citations=loaded.citations,
            limitations=loaded.limitations,
            model_name=loaded.model_name,
            paragraphs=loaded.paragraphs,
            answer_id=loaded.answer_id,
            evidence_packet_id=loaded.evidence_packet_id,
            search_run_id=packet.search_run_id,
        )
        return _public_answer(completed, packet.coverage)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _load_config(config_loader: Any) -> Config:
    try:
        return config_loader()
    except ConfigError as exc:
        raise ApiError(
            503,
            "configuration_error",
            "The application configuration is unavailable or invalid.",
        ) from exc


def _public_config(config: Config) -> dict[str, object]:
    return {
        "mode": "public" if config.public_demo_enabled else "local",
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
        "embedding_model": config.embedding_model,
        "answer_llm_provider": config.answer_llm_provider,
        "answer_llm_model": config.answer_llm_model,
        "ocr_enabled": config.ocr_enabled,
        "keyword_top_k": config.keyword_top_k,
        "semantic_top_k": config.semantic_top_k,
        "metadata_top_k": config.metadata_top_k,
        "final_top_k": config.final_top_k,
        "evidence_max_tokens": config.evidence_max_tokens,
        "answer_prompt_max_tokens": config.answer_prompt_max_tokens,
        "answer_session_message_limit": config.answer_session_message_limit,
        "ask_timeout_seconds": config.ask_timeout_seconds,
        "paths": {
            "courses_root_exists": config.courses_root.is_dir(),
            "data_dir_exists": config.data_dir.is_dir(),
            "sqlite_exists": config.sqlite_path.is_file(),
            "chroma_dir_exists": config.chroma_dir.is_dir(),
            "runs_dir_exists": config.runs_dir.is_dir(),
        },
    }


def _public_answer(
    answer: AnswerResult,
    coverage: Any,
    *,
    packet: Any = None,
) -> dict[str, object]:
    references: list[dict[str, object]] = []
    seen: set[str] = set()
    for citation in answer.citations:
        if citation.citation_id in seen:
            continue
        seen.add(citation.citation_id)
        references.append(
            {
                "citation_id": citation.citation_id,
                "course": citation.course,
                "file_path": citation.file_path,
                "source_type": citation.source_type,
                "location_label": citation.location_label,
            }
        )
    result = {
        "answer_id": answer.answer_id,
        "search_run_id": answer.search_run_id,
        "evidence_packet_id": answer.evidence_packet_id,
        "answer_text": answer.answer_text,
        "answer_body": answer_body(answer.answer_text),
        "answer_status": answer_status(answer),
        "citations": [item.as_safe_dict() for item in answer.citations],
        "references": references,
        "limitations": list(answer.limitations),
        "coverage": coverage.as_safe_dict(),
    }
    if packet is not None:
        result["evidence_packet"] = packet.as_safe_dict()
    return result


def _domain_error(
    exc: Exception,
    *,
    lookup: bool,
    trace_ids: tuple[int | None, int | None] = (None, None),
) -> ApiError:
    missing = "does not exist" in str(exc).lower()
    if lookup and missing:
        return ApiError(404, "not_found", "The requested resource does not exist.")
    if isinstance(exc, ConfigError):
        _, packet_id = trace_ids
        message = "Configuration is invalid or incomplete."
        if packet_id is not None:
            message = (
                f"Configuration is incomplete after evidence packet {packet_id} "
                "was stored; the packet remains available."
            )
        return ApiError(503, "configuration_error", message)
    if isinstance(exc, (QueryPlanningError, RetrievalError, AnswerGenerationError)):
        _, packet_id = trace_ids
        message = "A required model service failed."
        if packet_id is not None:
            message = (
                f"A required model service failed after evidence packet {packet_id} "
                "was stored; the packet remains available."
            )
        return ApiError(502, "provider_error", message)
    if isinstance(exc, EvidenceError):
        status = 500 if lookup else 502
        code = "stored_resource_error" if lookup else "retrieval_error"
        return ApiError(status, code, "Evidence processing failed.")
    if isinstance(exc, StorageError):
        return ApiError(500, "storage_error", "Stored application data is unavailable.")
    return ApiError(500, "internal_error", "An internal error occurred.")


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    try:
        task.exception()
    except (AskCancelled, asyncio.CancelledError):
        pass
