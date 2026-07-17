"""Inventory and extraction command family."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..cli_support.constants import (
    CONFIG_ERROR,
    EXTRACTION_ERROR,
    INVENTORY_ERROR,
    SUCCESS,
)
from ..cli_support.telemetry import LoggedRunResult
from ..config import ConfigError
from ..extraction import ExtractionError
from ..inventory import InventoryError
from ..storage import StorageError


@dataclass(frozen=True)
class IngestionServices:
    load_config: Callable[[], Any]
    inventory_courses: Callable[[Any], LoggedRunResult]
    load_inventory_summary: Callable[[Any], Any]
    extract_pending_files: Callable[..., LoggedRunResult]
    load_extraction_status: Callable[[Any], Any]
    summarize_data_files: Callable[..., LoggedRunResult]
    run_logged_command: Callable[..., int]
    print_inventory_run_result: Callable[[Any], None]
    print_inventory_summary: Callable[[Any], None]
    print_extraction_run_result: Callable[[Any], None]
    print_extraction_status: Callable[[Any], None]
    print_data_summary_run_result: Callable[[Any], None]


def register_commands(
    subparsers: argparse._SubParsersAction,
    *,
    inventory_run_handler: Callable[[argparse.Namespace], int],
    inventory_summary_handler: Callable[[argparse.Namespace], int],
    extract_run_handler: Callable[[argparse.Namespace], int],
    extract_status_handler: Callable[[argparse.Namespace], int],
    extract_data_summaries_handler: Callable[[argparse.Namespace], int],
) -> None:
    inventory_parser = subparsers.add_parser(
        "inventory",
        help="Course archive inventory commands.",
    )
    inventory_subparsers = inventory_parser.add_subparsers(
        dest="inventory_command",
        metavar="subcommand",
    )
    inventory_subparsers.required = True

    run_parser = inventory_subparsers.add_parser(
        "run",
        help="Inventory Courses and classify files without extracting content.",
    )
    run_parser.set_defaults(handler=inventory_run_handler)

    summary_parser = inventory_subparsers.add_parser(
        "summary",
        help="Print aggregate inventory counts from SQLite.",
    )
    summary_parser.set_defaults(handler=inventory_summary_handler)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Text extraction and chunking commands.",
    )
    extract_subparsers = extract_parser.add_subparsers(
        dest="extract_command",
        metavar="subcommand",
    )
    extract_subparsers.required = True

    run_parser = extract_subparsers.add_parser(
        "run",
        help="Extract pending text-like files into retrieval chunks.",
    )
    run_parser.add_argument(
        "--category",
        choices=("document", "slides", "notebook", "code", "transcript"),
        help="Limit extraction to one handled inventory category.",
    )
    run_parser.set_defaults(handler=extract_run_handler)

    status_parser = extract_subparsers.add_parser(
        "status",
        help="Print extraction and chunk coverage from SQLite.",
    )
    status_parser.set_defaults(handler=extract_status_handler)

    data_summary_parser = extract_subparsers.add_parser(
        "data-summaries",
        help="Summarize pending CSV/XLSX/JSON/JSONL/SQLite data files.",
    )
    data_summary_parser.add_argument(
        "--file-id",
        type=int,
        help="Limit data-summary extraction to one pending data_schema file id.",
    )
    data_summary_parser.set_defaults(handler=extract_data_summaries_handler)


def handle_inventory_run(
    _: argparse.Namespace,
    *,
    services: IngestionServices,
) -> int:
    return services.run_logged_command(
        command_name="inventory run",
        event_prefix="inventory",
        error_label="Inventory error",
        domain_error=InventoryError,
        error_code=INVENTORY_ERROR,
        completed_message="Inventory run completed",
        run=services.inventory_courses,
        print_result=services.print_inventory_run_result,
    )


def handle_inventory_summary(
    _: argparse.Namespace,
    *,
    services: IngestionServices,
) -> int:
    try:
        config = services.load_config()
        summary = services.load_inventory_summary(config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except (StorageError, InventoryError) as exc:
        print(f"Inventory summary error: {exc}", file=sys.stderr)
        return INVENTORY_ERROR

    print("Inventory summary")
    services.print_inventory_summary(summary)
    return SUCCESS


def handle_extract_run(
    args: argparse.Namespace,
    *,
    services: IngestionServices,
) -> int:
    return services.run_logged_command(
        command_name="extract run",
        event_prefix="extraction",
        error_label="Extraction error",
        domain_error=ExtractionError,
        error_code=EXTRACTION_ERROR,
        completed_message="Extraction run completed",
        run=lambda config: services.extract_pending_files(
            config,
            category=args.category,
        ),
        print_result=services.print_extraction_run_result,
    )


def handle_extract_status(
    _: argparse.Namespace,
    *,
    services: IngestionServices,
) -> int:
    try:
        config = services.load_config()
        status = services.load_extraction_status(config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except (StorageError, ExtractionError) as exc:
        print(f"Extraction status error: {exc}", file=sys.stderr)
        return EXTRACTION_ERROR

    print("Extraction status")
    services.print_extraction_status(status)
    return SUCCESS


def handle_extract_data_summaries(
    args: argparse.Namespace,
    *,
    services: IngestionServices,
) -> int:
    return services.run_logged_command(
        command_name="extract data-summaries",
        event_prefix="data_summary",
        error_label="Data summary error",
        domain_error=ExtractionError,
        error_code=EXTRACTION_ERROR,
        completed_message="Data summary run completed",
        run=lambda config: services.summarize_data_files(
            config,
            file_id=args.file_id,
        ),
        print_result=services.print_data_summary_run_result,
        extra={"file_id": args.file_id},
    )


__all__ = [
    "IngestionServices",
    "handle_extract_data_summaries",
    "handle_extract_run",
    "handle_extract_status",
    "handle_inventory_run",
    "handle_inventory_summary",
    "register_commands",
]
