"""Persisted evidence, answer, and one-shot ask CLI commands."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from ..answering import AnswerGenerationError, AnswerModelError
from ..cli_support.constants import (
    ANSWER_ERROR,
    CONFIG_ERROR,
    EVIDENCE_ERROR,
    SEARCH_ERROR,
    STORAGE_ERROR,
    SUCCESS,
)
from ..config import ConfigError
from ..retrieval import EvidenceError, QueryPlanningError, RetrievalError
from ..storage import StorageError


@dataclass(frozen=True)
class EvidenceAnsweringServices:
    load_config: Callable[[], Any]
    validate_config: Callable[[Any], None]
    build_evidence: Callable[..., Any]
    load_evidence_packet: Callable[..., Any]
    generate_answer: Callable[..., Any]
    store_answer: Callable[..., int]
    command_logger: Callable[[Any, str], Any]
    embedding_model_log_label: Callable[[Any, str | None], str]
    log_evidence_events: Callable[..., None]
    log_answer_event: Callable[..., None]
    print_evidence_build_result: Callable[..., None]
    print_evidence_packet: Callable[[Any], None]
    print_answer_result: Callable[..., None]


def register_commands(
    subparsers: argparse._SubParsersAction,
    *,
    evidence_build_handler: Callable[[argparse.Namespace], int],
    evidence_show_handler: Callable[[argparse.Namespace], int],
    answer_handler: Callable[[argparse.Namespace], int],
    ask_handler: Callable[[argparse.Namespace], int],
) -> None:
    evidence_parser = subparsers.add_parser(
        "evidence",
        help="Persisted evidence-packet and search-coverage commands.",
    )
    evidence_subparsers = evidence_parser.add_subparsers(
        dest="evidence_command",
        metavar="subcommand",
    )
    evidence_subparsers.required = True

    build_parser = evidence_subparsers.add_parser(
        "build",
        help="Run retrieval and persist one immutable evidence packet.",
    )
    build_parser.add_argument("query", nargs="+", help="Query text.")
    build_parser.add_argument(
        "--model",
        help="Supported reviewed local or hosted embedding profile; overrides UNI_RAG_EMBEDDING_MODEL.",
    )
    build_parser.add_argument(
        "--debug",
        action="store_true",
        help="Print persisted raw result sets and full fused contribution diagnostics.",
    )
    build_parser.add_argument(
        "--json",
        action="store_true",
        help="Print one complete safe EvidenceBuildResult JSON object.",
    )
    build_parser.set_defaults(handler=evidence_build_handler)

    show_parser = evidence_subparsers.add_parser(
        "show",
        help="Load an existing evidence packet without rebuilding it.",
    )
    show_parser.add_argument(
        "--search-run-id",
        type=int,
        required=True,
        help="Unique persisted search-run identifier.",
    )
    show_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the exact parsed stored packet as JSON.",
    )
    show_parser.set_defaults(handler=evidence_show_handler)

    answer_parser = subparsers.add_parser(
        "answer",
        help="Generate and persist an answer from an existing evidence packet.",
    )
    answer_parser.add_argument(
        "--evidence-packet-id",
        type=int,
        required=True,
        help="Evidence packet identifier to answer.",
    )
    answer_parser.add_argument(
        "--json",
        action="store_true",
        help="Print one complete answer result JSON object.",
    )
    answer_parser.set_defaults(handler=answer_handler)

    ask_parser = subparsers.add_parser(
        "ask",
        help="Build an evidence packet and answer it in one shot.",
    )
    ask_parser.add_argument("query", nargs="+", help="Query text.")
    ask_parser.add_argument(
        "--model",
        help="Supported reviewed local or hosted embedding profile; overrides UNI_RAG_EMBEDDING_MODEL.",
    )
    ask_parser.add_argument(
        "--json",
        action="store_true",
        help="Print one complete answer result JSON object.",
    )
    ask_parser.set_defaults(handler=ask_handler)


def handle_evidence_build(
    args: argparse.Namespace,
    *,
    services: EvidenceAnsweringServices,
) -> int:
    command_name = "evidence build"
    query_text = " ".join(args.query)
    logger = None
    model_label = "(unset)"
    try:
        config = services.load_config()
        services.validate_config(config)
        model_label = services.embedding_model_log_label(config, args.model)
        logger = services.command_logger(config, command_name)
        logger.info(
            "evidence build started",
            extra={
                "event": "evidence_build_started",
                "command": command_name,
                "status": "started",
                "model": model_label,
            },
        )
        result = services.build_evidence(config, query_text, model=args.model)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except StorageError as exc:
        if logger is not None:
            logger.exception(
                "evidence build failed",
                extra={
                    "event": "evidence_build_failed",
                    "command": command_name,
                    "status": "failed",
                    "model": model_label,
                },
            )
        print(f"Storage error: {exc}", file=sys.stderr)
        return STORAGE_ERROR
    except (RetrievalError, QueryPlanningError) as exc:
        if logger is not None:
            logger.exception(
                "evidence build failed",
                extra={
                    "event": "evidence_build_failed",
                    "command": command_name,
                    "status": "failed",
                    "model": model_label,
                },
            )
        print(f"Retrieval error: {exc}", file=sys.stderr)
        return SEARCH_ERROR
    except EvidenceError as exc:
        if logger is not None:
            logger.exception(
                "evidence build failed",
                extra={
                    "event": "evidence_build_failed",
                    "command": command_name,
                    "status": "failed",
                    "model": model_label,
                },
            )
        print(f"Evidence error: {exc}", file=sys.stderr)
        return EVIDENCE_ERROR

    services.log_evidence_events(logger, result, model_label)
    if args.json:
        print(json.dumps(result.as_safe_dict(), indent=2, sort_keys=True))
    else:
        services.print_evidence_build_result(
            result,
            debug=args.debug,
            config=config,
        )
    return SUCCESS


def handle_evidence_show(
    args: argparse.Namespace,
    *,
    services: EvidenceAnsweringServices,
) -> int:
    command_name = "evidence show"
    logger = None
    try:
        config = services.load_config()
        services.validate_config(config)
        logger = services.command_logger(config, command_name)
        packet = services.load_evidence_packet(
            config,
            search_run_id=args.search_run_id,
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except StorageError as exc:
        print(f"Storage error: {exc}", file=sys.stderr)
        return STORAGE_ERROR
    except EvidenceError as exc:
        if logger is not None:
            logger.exception(
                "evidence packet load failed",
                extra={
                    "event": "evidence_packet_load_failed",
                    "command": command_name,
                    "status": "failed",
                    "search_run_id": args.search_run_id,
                },
            )
        print(f"Evidence error: {exc}", file=sys.stderr)
        return EVIDENCE_ERROR

    if logger is not None:
        logger.info(
            "evidence packet loaded",
            extra={
                "event": "evidence_packet_loaded",
                "command": command_name,
                "status": "completed",
                "search_run_id": packet.search_run_id,
            },
        )
    if args.json:
        print(json.dumps(packet.as_safe_dict(), indent=2, sort_keys=True))
    else:
        services.print_evidence_packet(packet)
    return SUCCESS


def handle_answer(
    args: argparse.Namespace,
    *,
    services: EvidenceAnsweringServices,
) -> int:
    command_name = "answer"
    logger = None
    try:
        config = services.load_config()
        services.validate_config(config)
        logger = services.command_logger(config, command_name)
        packet = services.load_evidence_packet(
            config,
            evidence_packet_id=args.evidence_packet_id,
        )
        logger.info(
            "answer generation started",
            extra={
                "event": "answer_generation_started",
                "command": command_name,
                "status": "started",
                "evidence_packet_id": args.evidence_packet_id,
            },
        )
        answer = services.generate_answer(packet, config=config)
        answer_id = services.store_answer(
            args.evidence_packet_id,
            answer,
            config=config,
        )
        result = answer_with_ids(
            answer,
            answer_id=answer_id,
            evidence_packet_id=args.evidence_packet_id,
            search_run_id=packet.search_run_id,
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except StorageError as exc:
        print(f"Storage error: {exc}", file=sys.stderr)
        return STORAGE_ERROR
    except EvidenceError as exc:
        print(f"Evidence error: {exc}", file=sys.stderr)
        return EVIDENCE_ERROR
    except (AnswerGenerationError, AnswerModelError) as exc:
        if logger is not None:
            logger.error(
                "answer generation failed",
                extra={
                    "event": "answer_generation_failed",
                    "command": command_name,
                    "status": "failed",
                    "answer_error": type(exc).__name__,
                },
            )
        print(f"Answer error: {exc}", file=sys.stderr)
        return ANSWER_ERROR

    services.log_answer_event(logger, result, command_name)
    services.print_answer_result(result, json_output=args.json)
    return SUCCESS


def handle_ask(
    args: argparse.Namespace,
    *,
    services: EvidenceAnsweringServices,
) -> int:
    command_name = "ask"
    query_text = " ".join(args.query)
    logger = None
    model_label = "(unset)"
    packet_id: int | None = None
    try:
        config = services.load_config()
        services.validate_config(config)
        model_label = services.embedding_model_log_label(config, args.model)
        logger = services.command_logger(config, command_name)
        result = services.build_evidence(config, query_text, model=args.model)
        packet_id = result.evidence_packet_id
        answer = services.generate_answer(result.packet, config=config)
        answer_id = services.store_answer(packet_id, answer, config=config)
        answer_result = answer_with_ids(
            answer,
            answer_id=answer_id,
            evidence_packet_id=packet_id,
            search_run_id=result.search_run_id,
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except StorageError as exc:
        if logger is not None:
            logger.error(
                "ask failed",
                extra={
                    "event": "ask_failed",
                    "command": command_name,
                    "status": "failed",
                    "model": model_label,
                    "answer_error": "StorageError",
                },
            )
        print(f"Storage error: {exc}", file=sys.stderr)
        return STORAGE_ERROR
    except (RetrievalError, QueryPlanningError) as exc:
        print(f"Retrieval error: {exc}", file=sys.stderr)
        return SEARCH_ERROR
    except EvidenceError as exc:
        print(f"Evidence error: {exc}", file=sys.stderr)
        return EVIDENCE_ERROR
    except (AnswerGenerationError, AnswerModelError) as exc:
        # Evidence was built before answer generation; leave its packet and
        # search trace intact, but report the independent answer failure.
        if logger is not None:
            logger.error(
                "ask answer generation failed",
                extra={
                    "event": "ask_answer_failed",
                    "command": command_name,
                    "status": "failed",
                    "model": model_label,
                    "evidence_packet_id": packet_id,
                },
            )
        print(f"Answer error: {exc}", file=sys.stderr)
        return ANSWER_ERROR

    services.log_answer_event(logger, answer_result, command_name)
    services.print_answer_result(answer_result, json_output=args.json)
    return SUCCESS


def answer_with_ids(
    answer: Any,
    *,
    answer_id: int,
    evidence_packet_id: int,
    search_run_id: int,
) -> Any:
    return replace(
        answer,
        answer_id=answer_id,
        evidence_packet_id=evidence_packet_id,
        search_run_id=search_run_id,
    )


__all__ = [
    "EvidenceAnsweringServices",
    "answer_with_ids",
    "handle_answer",
    "handle_ask",
    "handle_evidence_build",
    "handle_evidence_show",
    "register_commands",
]
