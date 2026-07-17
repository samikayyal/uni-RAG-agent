"""Shared CLI logging adapters and long-running command orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from logging import Logger
import sys
from typing import Protocol

from ..config import Config, ConfigError
from .constants import CONFIG_ERROR, SUCCESS
from ..indexing import resolve_embedding_profile
from ..logging_config import build_run_log_path, configure_logging
from ..retrieval import EvidenceBuildResult
from ..retrieval.models import RetrievalRun
from ..storage import StorageError


class LoggedRunResult(Protocol):
    run_id: int
    status: str
    files_seen: int


def command_logger(config: Config, command_name: str) -> Logger:
    return configure_logging(
        level=config.log_level,
        jsonl_path=build_run_log_path(config.runs_dir, command_name),
        console=False,
    )


def embedding_model_log_label(config: Config, explicit_model: str | None) -> str:
    """Return the normalized model selection represented in command telemetry."""

    explicit = explicit_model.strip() if explicit_model else ""
    selected = explicit or (config.embedding_model or "").strip()
    if not selected:
        return "(unset)"
    try:
        return resolve_embedding_profile(config, selected).model_name
    except Exception:
        # Keep the original selection visible for the subsequent sanitized
        # construction error when the model is unknown or unsupported.
        return selected


def run_logged_command(
    *,
    command_name: str,
    event_prefix: str,
    error_label: str,
    domain_error: type[Exception],
    error_code: int,
    completed_message: str,
    load_config: Callable[[], Config],
    validate_config: Callable[[Config], None],
    logger_factory: Callable[[Config, str], Logger],
    run: Callable[[Config], LoggedRunResult],
    print_result: Callable[[LoggedRunResult], None],
    extra: Mapping[str, object] | None = None,
) -> int:
    """Run an ingestion command with one shared config/logging skeleton."""

    base_extra = dict(extra or {})
    logger: Logger | None = None
    try:
        config = load_config()
        validate_config(config)
        logger = logger_factory(config, command_name)
        logger.info(
            f"{command_name} started",
            extra={
                "event": f"{event_prefix}_started",
                "command": command_name,
                **base_extra,
            },
        )
        result = run(config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except (StorageError, domain_error) as exc:
        if logger is not None:
            logger.exception(
                f"{command_name} failed",
                extra={
                    "event": f"{event_prefix}_failed",
                    "command": command_name,
                    "status": "failed",
                    **base_extra,
                },
            )
        print(f"{error_label}: {exc}", file=sys.stderr)
        return error_code

    logger.info(
        f"{command_name} completed",
        extra={
            "event": f"{event_prefix}_completed",
            "command": command_name,
            "run_id": result.run_id,
            "status": result.status,
            "count": result.files_seen,
            **base_extra,
        },
    )
    print(completed_message)
    print_result(result)
    return SUCCESS


def log_answer_event(
    logger: Logger | None,
    answer: object,
    command_name: str,
) -> None:
    if logger is None:
        return
    logger.info(
        "answer completed",
        extra={
            "event": "answer_completed",
            "command": command_name,
            "status": "completed",
            "answer_id": answer.answer_id,
            "evidence_packet_id": answer.evidence_packet_id,
            "citation_count": len(answer.citations),
            "limitation_count": len(answer.limitations),
            "answer_model": answer.model_name,
        },
    )


def log_retrieval_events(
    logger: Logger | None,
    run: RetrievalRun,
    model: str,
    config: Config,
) -> None:
    if logger is None:
        return
    common = {
        "command": "retrieve",
        "model": model,
        "courses": run.searched_courses,
        "indexes": run.searched_indexes,
        "query_type": run.query_plan.query_type,
        "plan_confidence": run.query_plan.plan_confidence,
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
        "semantic_query_count": len(run.semantic_queries),
    }
    logger.info(
        "query planning completed",
        extra={
            "event": "query_planning_unsupported"
            if run.status == "unsupported"
            else "query_planning_completed",
            "status": run.status,
            **common,
        },
    )
    for result_set in run.result_sets:
        logger.info(
            f"{result_set.retrieval_method} search completed",
            extra={
                "event": f"{result_set.retrieval_method}_search_completed",
                "status": "completed",
                "count": len(result_set.results),
                "result_count": len(result_set.results),
                **common,
            },
        )
    logger.info(
        "retrieval completed",
        extra={
            "event": "retrieval_completed",
            "status": run.status,
            "count": len(run.results),
            "final_count": len(run.results),
            "weakness_count": len(run.weaknesses),
            **common,
        },
    )


def log_evidence_events(
    logger: Logger | None,
    result: EvidenceBuildResult,
    model: str,
) -> None:
    if logger is None:
        return
    common = {
        "command": "evidence build",
        "model": model,
        "search_run_id": result.search_run_id,
        "evidence_packet_id": result.evidence_packet_id,
    }
    logger.info(
        "search run created",
        extra={
            "event": "search_run_created",
            "status": "running",
            **common,
        },
    )
    for result_set in result.retrieval_run.result_sets:
        logger.info(
            "search result set recorded",
            extra={
                "event": "search_result_set_recorded",
                "status": "completed",
                "result_set_id": result_set.result_set_id,
                "result_count": len(result_set.results),
                **common,
            },
        )
    logger.info(
        "search results fused",
        extra={
            "event": "search_results_fused",
            "status": "completed",
            "fused_candidate_count": result.coverage.fused_candidate_count,
            **common,
        },
    )
    omitted_count = (
        result.coverage.token_budget_omission_count
        + result.coverage.oversized_evidence_omission_count
    )
    logger.info(
        "evidence packet created",
        extra={
            "event": "evidence_packet_created",
            "status": result.coverage.status,
            "evidence_count": result.coverage.evidence_count,
            "omitted_count": omitted_count,
            **common,
        },
    )
    logger.info(
        "evidence build completed",
        extra={
            "event": "evidence_build_completed",
            "status": result.coverage.status,
            "evidence_count": result.coverage.evidence_count,
            "fused_candidate_count": result.coverage.fused_candidate_count,
            "omitted_count": omitted_count,
            **common,
        },
    )


__all__ = [
    "LoggedRunResult",
    "command_logger",
    "embedding_model_log_label",
    "log_answer_event",
    "log_evidence_events",
    "log_retrieval_events",
    "run_logged_command",
]
