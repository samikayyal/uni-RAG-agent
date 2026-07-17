"""Planner-backed retrieval CLI command."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..cli_support.constants import CONFIG_ERROR, SEARCH_ERROR, STORAGE_ERROR, SUCCESS
from ..config import ConfigError
from ..retrieval import QueryPlanningError, RetrievalError
from ..storage import StorageError


@dataclass(frozen=True)
class RetrievalServices:
    load_config: Callable[[], Any]
    validate_config: Callable[[Any], None]
    retrieve: Callable[..., Any]
    command_logger: Callable[[Any, str], Any]
    embedding_model_log_label: Callable[[Any, str | None], str]
    log_retrieval_events: Callable[..., None]
    print_retrieval_run: Callable[..., None]


def register_command(
    subparsers: argparse._SubParsersAction,
    *,
    handler: Callable[[argparse.Namespace], int],
) -> None:
    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Retrieve source-grounded evidence for a query.",
    )
    retrieve_parser.add_argument("query", nargs="+", help="Query text.")
    retrieve_parser.add_argument(
        "--model",
        help="Supported reviewed local or hosted embedding profile; overrides UNI_RAG_EMBEDDING_MODEL.",
    )
    retrieve_parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw result sets and full RRF contribution diagnostics.",
    )
    retrieve_parser.add_argument(
        "--json",
        action="store_true",
        help="Print one complete safe RetrievalRun JSON object.",
    )
    retrieve_parser.set_defaults(handler=handler)


def handle_retrieve(
    args: argparse.Namespace,
    *,
    services: RetrievalServices,
) -> int:
    command_name = "retrieve"
    query_text = " ".join(args.query)
    logger = None
    model_label = "(unset)"
    try:
        config = services.load_config()
        services.validate_config(config)
        model_label = services.embedding_model_log_label(config, args.model)
        logger = services.command_logger(config, command_name)
        logger.info(
            "retrieval started",
            extra={
                "event": "retrieval_started",
                "command": command_name,
                "status": "started",
                "model": model_label,
            },
        )
        run = services.retrieve(config, query_text, model=args.model)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except StorageError as exc:
        if logger is not None:
            logger.exception(
                "retrieval failed",
                extra={
                    "event": "retrieval_failed",
                    "command": command_name,
                    "status": "failed",
                },
            )
        print(f"Storage error: {exc}", file=sys.stderr)
        return STORAGE_ERROR
    except (RetrievalError, QueryPlanningError) as exc:
        if logger is not None:
            logger.exception(
                "retrieval failed",
                extra={
                    "event": "retrieval_failed",
                    "command": command_name,
                    "status": "failed",
                    "model": model_label,
                },
            )
        print(f"Retrieval error: {exc}", file=sys.stderr)
        return SEARCH_ERROR

    services.log_retrieval_events(logger, run, model_label, config)
    if args.json:
        print(json.dumps(run.as_safe_dict(), indent=2, sort_keys=True))
    else:
        services.print_retrieval_run(run, debug=args.debug)
    return SUCCESS


__all__ = [
    "RetrievalServices",
    "handle_retrieve",
    "register_command",
]
