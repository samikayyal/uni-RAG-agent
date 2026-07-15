"""FastAPI routes and safe public response projections."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Annotated

from fastapi import FastAPI, Path as ApiPath, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..answering import (
    AnswerGenerationError,
    AnswerResult,
    AnswerSession,
    generate_answer,
    load_answer,
    store_answer,
)
from ..config import Config, ConfigError, load_config
from ..retrieval import (
    EvidenceError,
    QueryPlanningError,
    RetrievalError,
    build_evidence,
    explain_search_coverage,
    load_evidence_packet,
)
from ..storage import StorageError
from .service import (
    AskCancelled,
    AskOrchestrator,
    PersistenceGate,
    SessionCapacityError,
    SessionRegistry,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
PositiveId = Annotated[int, ApiPath(gt=0)]


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
) -> FastAPI:
    """Create a provider-lazy app with injectable service boundaries."""
    resolved = services or AppServices()
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
    )
    app = FastAPI(title="Uni RAG Agent", docs_url=None, redoc_url=None)

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

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/config")
    async def config_view() -> dict[str, object]:
        config = _load_config(config_loader)
        return _public_config(config)

    @app.post("/api/ask")
    async def ask(payload: AskRequest) -> dict[str, object]:
        config = _load_config(config_loader)
        gate = PersistenceGate()
        task = asyncio.create_task(
            asyncio.to_thread(
                orchestrator.ask,
                config,
                payload.query,
                payload.session_id,
                gate,
            )
        )
        try:
            answer, coverage = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=config.ask_timeout_seconds,
            )
        except TimeoutError:
            if await asyncio.to_thread(gate.cancel):
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
            answer, coverage = await task
        except AskCancelled:
            raise ApiError(504, "ask_timeout", "The ask request timed out.")
        except SessionCapacityError:
            raise ApiError(
                503,
                "session_capacity",
                "All in-process session slots are currently active.",
            )
        except Exception as exc:
            raise _domain_error(exc, lookup=False, trace_ids=gate.trace_ids) from exc
        return _public_answer(answer, coverage)

    @app.get("/api/search-runs/{search_run_id}/coverage")
    async def coverage(search_run_id: PositiveId) -> dict[str, object]:
        config = _load_config(config_loader)
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
        config = _load_config(config_loader)
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
        config = _load_config(config_loader)
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


def _public_answer(answer: AnswerResult, coverage: Any) -> dict[str, object]:
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
    return {
        "answer_id": answer.answer_id,
        "search_run_id": answer.search_run_id,
        "evidence_packet_id": answer.evidence_packet_id,
        "answer_text": answer.answer_text,
        "citations": [item.as_safe_dict() for item in answer.citations],
        "references": references,
        "limitations": list(answer.limitations),
        "coverage": coverage.as_safe_dict(),
    }


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
