"""Evaluation command family."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..cli_support.constants import CONFIG_ERROR, EVALUATION_ERROR, SUCCESS
from ..config import ConfigError
from ..evaluation import EvaluationError
from ..indexing import KeywordIndexError, VectorIndexError
from ..inventory import InventoryError
from ..extraction import ExtractionError
from ..storage import StorageError


@dataclass(frozen=True)
class EvaluationServices:
    load_config: Callable[[], Any]
    validate_config: Callable[[Any], None]
    prepare_fixture_state: Callable[[Any], Any]
    run_eval_set: Callable[..., Any]
    sanitize_error: Callable[[object], str]


def register_commands(
    subparsers: argparse._SubParsersAction,
    *,
    prepare_handler: Callable[[argparse.Namespace], int],
    run_handler: Callable[[argparse.Namespace], int],
) -> None:
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluation and hardening commands.",
    )
    eval_subparsers = eval_parser.add_subparsers(
        dest="eval_command",
        metavar="subcommand",
    )
    eval_subparsers.required = True

    prepare_parser = eval_subparsers.add_parser(
        "prepare-fixtures",
        help=(
            "Build isolated fixture inventory, extraction, keyword, and vector "
            "state using configured production providers."
        ),
    )
    prepare_parser.set_defaults(handler=prepare_handler)

    run_parser = eval_subparsers.add_parser(
        "run",
        help="Run the committed fixture eval set (the default mode).",
    )
    mode = run_parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fixtures",
        action="store_true",
        help="Run the committed fixture set (equivalent to bare eval run).",
    )
    mode.add_argument(
        "--smoke-real-archive",
        action="store_true",
        help=(
            "Explicitly run data/runs/eval/real-archive.json against the normal "
            "configured archive state."
        ),
    )
    run_parser.set_defaults(handler=run_handler)


def handle_prepare_fixtures(
    _: argparse.Namespace,
    *,
    services: EvaluationServices,
) -> int:
    try:
        config = services.load_config()
        services.validate_config(config)
        manifest = services.prepare_fixture_state(config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except (
        EvaluationError,
        StorageError,
        InventoryError,
        ExtractionError,
        KeywordIndexError,
        VectorIndexError,
    ) as exc:
        print(f"Evaluation error: {services.sanitize_error(exc)}", file=sys.stderr)
        return EVALUATION_ERROR

    print("Fixture evaluation state prepared")
    print(f"manifest_version: {manifest['manifest_version']}")
    print(f"embedding_model: {manifest['embedding_model']}")
    print(f"files: {manifest['files']}")
    print(f"chunks: {manifest['chunks']}")
    print(f"keyword_rows: {manifest['keyword_rows']}")
    print(f"vector_rows: {manifest['vector_rows']}")
    return SUCCESS


def handle_run(
    args: argparse.Namespace,
    *,
    services: EvaluationServices,
) -> int:
    smoke = bool(args.smoke_real_archive)
    try:
        config = services.load_config()
        services.validate_config(config)
        report_path, results = services.run_eval_set(
            config,
            fixtures=not smoke,
            smoke_real_archive=smoke,
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except (EvaluationError, StorageError) as exc:
        print(f"Evaluation error: {services.sanitize_error(exc)}", file=sys.stderr)
        return EVALUATION_ERROR

    failed = sum(not result.passed for result in results)
    print("Evaluation run completed")
    print(f"mode: {'real-archive' if smoke else 'fixtures'}")
    print(f"items: {len(results)}")
    print(f"passed: {len(results) - failed}")
    print(f"failed: {failed}")
    print(f"json_report: {report_path}")
    print(f"markdown_report: {report_path.with_suffix('.md')}")
    return SUCCESS if failed == 0 else EVALUATION_ERROR


__all__ = [
    "EvaluationServices",
    "handle_prepare_fixtures",
    "handle_run",
    "register_commands",
]
