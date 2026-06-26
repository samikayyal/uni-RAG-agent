"""Command line dispatcher for Uni RAG Agent."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from typing import Callable

from . import __version__
from .config import ConfigError, load_config, validate_config
from .extraction import (
    ExtractionError,
    ExtractionRunResult,
    ExtractionStatus,
    extract_pending_files,
    load_extraction_status,
)
from .inventory import (
    InventoryError,
    InventoryRunResult,
    InventorySummary,
    inventory_courses,
    load_inventory_summary,
)
from .storage import (
    StorageCheckResult,
    StorageError,
    check_storage,
    connect_sqlite,
    ensure_data_dirs,
    initialize_schema,
)

SUCCESS = 0
NOT_IMPLEMENTED = 1
CONFIG_ERROR = 2
STORAGE_ERROR = 3
INVENTORY_ERROR = 4
EXTRACTION_ERROR = 5

CommandHandler = Callable[[argparse.Namespace], int]

COMMAND_EXAMPLES = """\
Available command shapes:
  uv run -m uni_rag_agent config check
  uv run -m uni_rag_agent storage init
  uv run -m uni_rag_agent storage check
  uv run -m uni_rag_agent inventory run
  uv run -m uni_rag_agent extract run
  uv run -m uni_rag_agent index keyword
  uv run -m uni_rag_agent index vector
  uv run -m uni_rag_agent retrieve "query text"
  uv run -m uni_rag_agent eval run
  uv run -m uni_rag_agent app serve
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: CommandHandler | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help(sys.stderr)
        return CONFIG_ERROR
    return handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uni_rag_agent",
        description="Local course archive intelligence system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=COMMAND_EXAMPLES,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="command")
    subparsers.required = True

    _add_config_commands(subparsers)
    _add_storage_commands(subparsers)
    _add_inventory_commands(subparsers)
    _add_extract_commands(subparsers)
    _add_stub_group(
        subparsers,
        "index",
        "Index maintenance commands.",
        {
            "keyword": ("index keyword", "Feature Spec 06: Keyword Indexing"),
            "vector": ("index vector", "Feature Spec 07: Vector Indexing"),
        },
    )
    _add_retrieve_command(subparsers)
    _add_stub_group(
        subparsers,
        "eval",
        "Evaluation commands.",
        {"run": ("eval run", "Feature Spec 12: Evaluation and Hardening")},
    )
    _add_stub_group(
        subparsers,
        "app",
        "Application server commands.",
        {"serve": ("app serve", "Feature Spec 11: FastAPI HTML UI")},
    )

    return parser


def _add_config_commands(subparsers: argparse._SubParsersAction) -> None:
    config_parser = subparsers.add_parser("config", help="Configuration commands.")
    config_subparsers = config_parser.add_subparsers(
        dest="config_command",
        metavar="subcommand",
    )
    config_subparsers.required = True

    check_parser = config_subparsers.add_parser(
        "check",
        help="Load configuration and print non-secret resolved values.",
    )
    check_parser.set_defaults(handler=_handle_config_check)


def _add_storage_commands(subparsers: argparse._SubParsersAction) -> None:
    storage_parser = subparsers.add_parser(
        "storage",
        help="Storage and schema commands.",
    )
    storage_subparsers = storage_parser.add_subparsers(
        dest="storage_command",
        metavar="subcommand",
    )
    storage_subparsers.required = True

    init_parser = storage_subparsers.add_parser(
        "init",
        help="Create generated data directories and initialize SQLite schema.",
    )
    init_parser.set_defaults(handler=_handle_storage_init)

    check_parser = storage_subparsers.add_parser(
        "check",
        help="Check generated storage directories, SQLite schema, and FTS5 support.",
    )
    check_parser.set_defaults(handler=_handle_storage_check)


def _add_inventory_commands(subparsers: argparse._SubParsersAction) -> None:
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
    run_parser.set_defaults(handler=_handle_inventory_run)

    summary_parser = inventory_subparsers.add_parser(
        "summary",
        help="Print aggregate inventory counts from SQLite.",
    )
    summary_parser.set_defaults(handler=_handle_inventory_summary)


def _add_extract_commands(subparsers: argparse._SubParsersAction) -> None:
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
    run_parser.set_defaults(handler=_handle_extract_run)

    status_parser = extract_subparsers.add_parser(
        "status",
        help="Print extraction and chunk coverage from SQLite.",
    )
    status_parser.set_defaults(handler=_handle_extract_status)


def _add_stub_group(
    subparsers: argparse._SubParsersAction,
    group_name: str,
    help_text: str,
    commands: dict[str, tuple[str, str]],
) -> None:
    group_parser = subparsers.add_parser(group_name, help=help_text)
    group_subparsers = group_parser.add_subparsers(
        dest=f"{group_name}_command",
        metavar="subcommand",
    )
    group_subparsers.required = True

    for subcommand, (command_name, feature_spec) in commands.items():
        command_parser = group_subparsers.add_parser(subcommand)
        command_parser.set_defaults(
            handler=_not_implemented_handler(command_name, feature_spec),
        )


def _add_retrieve_command(subparsers: argparse._SubParsersAction) -> None:
    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Retrieve source-grounded evidence for a query.",
    )
    retrieve_parser.add_argument("query", nargs="+", help="Query text.")
    retrieve_parser.set_defaults(
        handler=_not_implemented_handler(
            "retrieve",
            "Feature Spec 08: Query Routing and Hybrid Retrieval",
        ),
    )


def _handle_config_check(_: argparse.Namespace) -> int:
    try:
        config = load_config()
        validate_config(config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR

    print("Configuration OK")
    for key, value in config.as_safe_dict().items():
        print(f"{key}: {value}")
    return SUCCESS


def _handle_storage_init(_: argparse.Namespace) -> int:
    try:
        config = load_config()
        ensure_data_dirs(config)
        with connect_sqlite(config) as connection:
            initialize_schema(connection)
        result = check_storage(config)
    except (ConfigError, StorageError) as exc:
        print(f"Storage initialization error: {exc}", file=sys.stderr)
        return STORAGE_ERROR

    print("Storage initialized")
    _print_storage_result(result)
    return SUCCESS if result.ok else STORAGE_ERROR


def _handle_storage_check(_: argparse.Namespace) -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR

    result = check_storage(config)
    print("Storage OK" if result.ok else "Storage check failed")
    _print_storage_result(result)
    return SUCCESS if result.ok else STORAGE_ERROR


def _handle_inventory_run(_: argparse.Namespace) -> int:
    try:
        config = load_config()
        result = inventory_courses(config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except (StorageError, InventoryError) as exc:
        print(f"Inventory error: {exc}", file=sys.stderr)
        return INVENTORY_ERROR

    print("Inventory run completed")
    _print_inventory_run_result(result)
    return SUCCESS


def _handle_inventory_summary(_: argparse.Namespace) -> int:
    try:
        config = load_config()
        summary = load_inventory_summary(config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except (StorageError, InventoryError) as exc:
        print(f"Inventory summary error: {exc}", file=sys.stderr)
        return INVENTORY_ERROR

    print("Inventory summary")
    _print_inventory_summary(summary)
    return SUCCESS


def _handle_extract_run(args: argparse.Namespace) -> int:
    try:
        config = load_config()
        result = extract_pending_files(config, category=args.category)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except (StorageError, ExtractionError) as exc:
        print(f"Extraction error: {exc}", file=sys.stderr)
        return EXTRACTION_ERROR

    print("Extraction run completed")
    _print_extraction_run_result(result)
    return SUCCESS


def _handle_extract_status(_: argparse.Namespace) -> int:
    try:
        config = load_config()
        status = load_extraction_status(config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except (StorageError, ExtractionError) as exc:
        print(f"Extraction status error: {exc}", file=sys.stderr)
        return EXTRACTION_ERROR

    print("Extraction status")
    _print_extraction_status(status)
    return SUCCESS


def _print_storage_result(result: StorageCheckResult) -> None:
    safe = result.as_safe_dict()
    for key in (
        "data_dir",
        "sqlite_path",
        "extracted_dir",
        "chroma_dir",
        "runs_dir",
        "sqlite_exists",
        "fts5_available",
    ):
        print(f"{key}: {safe[key]}")
    print(f"required_tables_present: {len(result.required_tables_present)}")
    if result.missing_tables:
        print(f"missing_tables: {', '.join(result.missing_tables)}")
    else:
        print("missing_tables: none")
    if result.diagnostics:
        print("diagnostics:")
        for diagnostic in result.diagnostics:
            print(f"- {diagnostic}")


def _print_inventory_run_result(result: InventoryRunResult) -> None:
    print(f"run_id: {result.run_id}")
    print(f"status: {result.status}")
    print(f"started_at: {result.started_at}")
    print(f"finished_at: {result.finished_at}")
    print(f"courses_seen: {result.courses_seen}")
    print(f"files_seen: {result.files_seen}")
    print(f"files_pending: {result.files_pending}")
    print(f"files_metadata_only: {result.files_metadata_only}")
    print(f"files_failed: {result.files_failed}")
    print(f"files_missing: {result.files_missing}")
    print(f"bytes_seen: {result.bytes_seen}")
    _print_count_mapping("by_status", result.by_status)
    _print_count_mapping("by_category", result.by_category)
    _print_count_mapping("by_extension", result.by_extension)
    _print_count_mapping("by_reason", result.by_reason)
    if result.diagnostics:
        print("diagnostics:")
        for diagnostic in result.diagnostics:
            print(f"- {diagnostic}")


def _print_inventory_summary(summary: InventorySummary) -> None:
    print(f"latest_inventory_run_id: {summary.latest_inventory_run_id}")
    print(f"latest_inventory_started_at: {summary.latest_inventory_started_at}")
    print(f"courses_total: {summary.courses_total}")
    print(f"files_total: {summary.files_total}")
    print(f"files_missing: {summary.files_missing}")
    print(f"bytes_total: {summary.bytes_total}")
    if summary.by_course:
        print("by_course:")
        for course in summary.by_course:
            print(f"- {course.name}: files={course.file_count}, bytes={course.total_bytes}")
    _print_count_mapping("by_status", summary.by_status)
    _print_count_mapping("by_category", summary.by_category)
    _print_count_mapping("by_extension", summary.by_extension)
    _print_count_mapping("by_reason", summary.by_reason)


def _print_extraction_run_result(result: ExtractionRunResult) -> None:
    print(f"run_id: {result.run_id}")
    print(f"status: {result.status}")
    print(f"started_at: {result.started_at}")
    print(f"finished_at: {result.finished_at}")
    print(f"category: {result.category or 'all'}")
    print(f"files_seen: {result.files_seen}")
    print(f"files_indexed: {result.files_indexed}")
    print(f"files_failed: {result.files_failed}")
    print(f"chunks_created: {result.chunks_created}")
    _print_count_mapping("by_source_type", result.by_source_type)
    if result.failures:
        print("failures:")
        for failure in result.failures:
            print(f"- file_id={failure.file_id} path={failure.path}: {failure.error}")
    if result.diagnostics:
        print("diagnostics:")
        for diagnostic in result.diagnostics:
            print(f"- {diagnostic}")


def _print_extraction_status(status: ExtractionStatus) -> None:
    print(f"latest_extraction_run_id: {status.latest_extraction_run_id}")
    print(f"latest_extraction_started_at: {status.latest_extraction_started_at}")
    print(f"pending_text_files: {status.pending_text_files}")
    print(f"indexed_text_files: {status.indexed_text_files}")
    print(f"failed_text_files: {status.failed_text_files}")
    print(f"extracted_documents: {status.extracted_documents}")
    print(f"chunks_total: {status.chunks_total}")
    _print_count_mapping("pending_by_category", status.pending_by_category)
    _print_count_mapping("chunks_by_source_type", status.chunks_by_source_type)
    if status.recent_failures:
        print("recent_failures:")
        for failure in status.recent_failures:
            print(f"- file_id={failure.file_id} path={failure.path}: {failure.error}")


def _print_count_mapping(label: str, counts: Mapping[str, int]) -> None:
    if not counts:
        print(f"{label}: none")
        return
    print(f"{label}:")
    for key, value in counts.items():
        print(f"- {key}: {value}")


def _not_implemented_handler(command_name: str, feature_spec: str) -> CommandHandler:
    def handler(_: argparse.Namespace) -> int:
        print(
            f"Command '{command_name}' is registered but not implemented yet. "
            f"Expected implementation: {feature_spec}.",
            file=sys.stderr,
        )
        return NOT_IMPLEMENTED

    return handler
