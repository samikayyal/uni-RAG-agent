"""Command line dispatcher for Uni RAG Agent."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from logging import Logger
from typing import Callable, Protocol

from . import __version__
from .config import Config, ConfigError, load_config, validate_config
from .extraction import (
    DataSummaryRunResult,
    ExtractionError,
    ExtractionRunResult,
    ExtractionStatus,
    extract_pending_files,
    load_extraction_status,
    summarize_data_files,
)
from .inventory import (
    InventoryError,
    InventoryRunResult,
    InventorySummary,
    inventory_courses,
    load_inventory_summary,
)
from .indexing import (
    KeywordIndexError,
    KeywordIndexResult,
    KeywordSearchError,
    SemanticSearchError,
    VectorIndexError,
    VectorIndexResult,
    keyword_query_terms,
    keyword_search,
    sync_keyword_index,
)
from .retrieval import RetrievalResult
from .storage import (
    StorageCheckResult,
    StorageError,
    check_storage,
    connect_sqlite,
    ensure_data_dirs,
    initialize_schema,
)
from .logging_config import build_run_log_path, configure_logging

SUCCESS = 0
NOT_IMPLEMENTED = 1
CONFIG_ERROR = 2
STORAGE_ERROR = 3
INVENTORY_ERROR = 4
EXTRACTION_ERROR = 5
INDEX_ERROR = 6
SEARCH_ERROR = 7

CommandHandler = Callable[[argparse.Namespace], int]

COMMAND_EXAMPLES = """\
Available command shapes:
  uv run -m uni_rag_agent config check
  uv run -m uni_rag_agent storage init
  uv run -m uni_rag_agent storage check
  uv run -m uni_rag_agent inventory run
  uv run -m uni_rag_agent extract run
  uv run -m uni_rag_agent extract data-summaries
  uv run -m uni_rag_agent index keyword
  uv run -m uni_rag_agent index vector
  uv run -m uni_rag_agent search keyword "query text"
  uv run -m uni_rag_agent search semantic "query text"
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
    _add_index_commands(subparsers)
    _add_search_commands(subparsers)
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

    data_summary_parser = extract_subparsers.add_parser(
        "data-summaries",
        help="Summarize pending CSV/XLSX/JSON/JSONL/SQLite data files.",
    )
    data_summary_parser.add_argument(
        "--file-id",
        type=int,
        help="Limit data-summary extraction to one pending data_schema file id.",
    )
    data_summary_parser.set_defaults(handler=_handle_extract_data_summaries)


def _add_index_commands(subparsers: argparse._SubParsersAction) -> None:
    index_parser = subparsers.add_parser(
        "index",
        help="Index maintenance commands.",
    )
    index_subparsers = index_parser.add_subparsers(
        dest="index_command",
        metavar="subcommand",
    )
    index_subparsers.required = True

    keyword_parser = index_subparsers.add_parser(
        "keyword",
        help="Rebuild the SQLite FTS5 keyword projection from current chunks.",
    )
    keyword_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Explicitly request the MVP full rebuild behavior.",
    )
    keyword_parser.set_defaults(handler=_handle_index_keyword)

    vector_parser = index_subparsers.add_parser(
        "vector",
        help="Embed current eligible chunks into ChromaDB for the selected model.",
    )
    vector_parser.add_argument(
        "--collection",
        help="Limit indexing to one logical index such as document_index.",
    )
    vector_parser.add_argument(
        "--model",
        help=(
            "Supported Hugging Face profile; falls back to "
            "UNI_RAG_EMBEDDING_MODEL when omitted."
        ),
    )
    vector_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear and repopulate only the selected model/profile and collection.",
    )
    vector_parser.set_defaults(handler=_handle_index_vector)


def _add_search_commands(subparsers: argparse._SubParsersAction) -> None:
    search_parser = subparsers.add_parser(
        "search",
        help="Direct search commands.",
    )
    search_subparsers = search_parser.add_subparsers(
        dest="search_command",
        metavar="subcommand",
    )
    search_subparsers.required = True

    keyword_parser = search_subparsers.add_parser(
        "keyword",
        help="Run plain-text SQLite FTS5 keyword search over current chunks.",
    )
    keyword_parser.add_argument("query", nargs="+", help="Plain-text query.")
    keyword_parser.add_argument(
        "--course",
        help="Case-insensitive exact course-name filter.",
    )
    keyword_parser.add_argument(
        "--index",
        dest="indexes",
        action="append",
        help="Logical index filter such as slides_index. May be repeated.",
    )
    keyword_parser.add_argument(
        "--top-k",
        type=int,
        help="Maximum keyword results to return.",
    )
    keyword_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON result objects instead of a table.",
    )
    keyword_parser.set_defaults(handler=_handle_search_keyword)

    semantic_parser = search_subparsers.add_parser(
        "semantic",
        help="Run semantic vector search over current chunks via ChromaDB.",
    )
    semantic_parser.add_argument("query", nargs="+", help="Query text.")
    semantic_parser.add_argument(
        "--course",
        help="Case-insensitive exact course-name filter.",
    )
    semantic_parser.add_argument(
        "--index",
        dest="indexes",
        action="append",
        help="Logical index filter such as slides_index. May be repeated.",
    )
    semantic_parser.add_argument(
        "--top-k",
        type=int,
        help="Maximum semantic results to return.",
    )
    semantic_parser.add_argument(
        "--model",
        help=(
            "Supported Hugging Face profile; falls back to "
            "UNI_RAG_EMBEDDING_MODEL when omitted. Must match the indexed model."
        ),
    )
    semantic_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON result objects instead of a table.",
    )
    semantic_parser.set_defaults(handler=_handle_search_semantic)


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


class _LoggedRunResult(Protocol):
    run_id: int
    status: str
    files_seen: int


def _run_logged_command(
    *,
    command_name: str,
    event_prefix: str,
    error_label: str,
    domain_error: type[Exception],
    error_code: int,
    completed_message: str,
    run: Callable[[Config], _LoggedRunResult],
    print_result: Callable[[_LoggedRunResult], None],
    extra: Mapping[str, object] | None = None,
) -> int:
    """Run a command with uniform config loading, run logging, and error mapping.

    The three long-running commands (inventory, extract, data-summaries) share
    the same load → validate → log-start → run → log-complete skeleton; only the
    callable, error type, exit code, and result printer differ.
    """
    base_extra = dict(extra or {})
    logger: Logger | None = None
    try:
        config = load_config()
        validate_config(config)
        logger = _command_logger(config, command_name)
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


def _handle_inventory_run(_: argparse.Namespace) -> int:
    return _run_logged_command(
        command_name="inventory run",
        event_prefix="inventory",
        error_label="Inventory error",
        domain_error=InventoryError,
        error_code=INVENTORY_ERROR,
        completed_message="Inventory run completed",
        run=inventory_courses,
        print_result=_print_inventory_run_result,
    )


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
    return _run_logged_command(
        command_name="extract run",
        event_prefix="extraction",
        error_label="Extraction error",
        domain_error=ExtractionError,
        error_code=EXTRACTION_ERROR,
        completed_message="Extraction run completed",
        run=lambda config: extract_pending_files(config, category=args.category),
        print_result=_print_extraction_run_result,
    )


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


def _handle_extract_data_summaries(args: argparse.Namespace) -> int:
    return _run_logged_command(
        command_name="extract data-summaries",
        event_prefix="data_summary",
        error_label="Data summary error",
        domain_error=ExtractionError,
        error_code=EXTRACTION_ERROR,
        completed_message="Data summary run completed",
        run=lambda config: summarize_data_files(config, file_id=args.file_id),
        print_result=_print_data_summary_run_result,
        extra={"file_id": args.file_id},
    )


def _handle_index_keyword(args: argparse.Namespace) -> int:
    command_name = "index keyword"
    logger: Logger | None = None
    try:
        config = load_config()
        validate_config(config)
        logger = _command_logger(config, command_name)
        logger.info(
            "keyword index started",
            extra={
                "event": "keyword_index_started",
                "command": command_name,
                "status": "started",
            },
        )
        result = sync_keyword_index(config, rebuild=args.rebuild)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except (StorageError, KeywordIndexError) as exc:
        if logger is not None:
            logger.exception(
                "keyword index failed",
                extra={
                    "event": "keyword_index_failed",
                    "command": command_name,
                    "status": "failed",
                },
            )
        print(f"Keyword index error: {exc}", file=sys.stderr)
        return INDEX_ERROR

    logger.info(
        "keyword index completed",
        extra={
            "event": "keyword_index_completed",
            "command": command_name,
            "status": "completed",
            "count": result.rows_indexed,
            "rows_removed": result.rows_removed,
            "chunks_seen": result.chunks_seen,
            "rows_indexed": result.rows_indexed,
        },
    )
    print("Keyword index completed")
    _print_keyword_index_result(result)
    return SUCCESS


def _handle_search_keyword(args: argparse.Namespace) -> int:
    command_name = "search keyword"
    logger: Logger | None = None
    query_text = " ".join(args.query)
    keyword_terms: tuple[str, ...] = ()
    top_k: int | None = args.top_k
    try:
        config = load_config()
        validate_config(config)
        top_k = args.top_k if args.top_k is not None else config.keyword_top_k
        logger = _command_logger(config, command_name)
        keyword_terms = keyword_query_terms(query_text)
        logger.info(
            "keyword search started",
            extra={
                "event": "keyword_search_started",
                "command": command_name,
                "status": "started",
                "keyword_terms": keyword_terms,
                "course": args.course,
                "indexes": args.indexes or (),
                "top_k": top_k,
            },
        )
        results = keyword_search(
            config,
            query=query_text,
            course=args.course,
            indexes=args.indexes,
            top_k=args.top_k,
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except (StorageError, KeywordSearchError) as exc:
        if logger is not None:
            logger.exception(
                "keyword search failed",
                extra={
                    "event": "keyword_search_failed",
                    "command": command_name,
                    "status": "failed",
                    "keyword_terms": keyword_terms,
                    "course": args.course,
                    "indexes": args.indexes or (),
                    "top_k": top_k,
                    "count": 0,
                    "result_count": 0,
                },
            )
        print(f"Keyword search error: {exc}", file=sys.stderr)
        return SEARCH_ERROR

    logger.info(
        "keyword search completed",
        extra={
            "event": "keyword_search_completed",
            "command": command_name,
            "status": "completed",
            "keyword_terms": keyword_terms,
            "course": args.course,
            "indexes": args.indexes or (),
            "top_k": top_k,
            "count": len(results),
            "result_count": len(results),
        },
    )
    if args.json:
        print(
            json.dumps(
                [result.as_safe_dict() for result in results],
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_keyword_search_results(results)
    return SUCCESS


def _handle_index_vector(args: argparse.Namespace) -> int:
    from .indexing import sync_vector_index

    command_name = "index vector"
    logger: Logger | None = None
    base_extra = {
        "command": command_name,
        "collection": args.collection or "all",
    }
    try:
        config = load_config()
        validate_config(config)
        base_extra["model"] = _embedding_model_log_label(config, args.model)
        logger = _command_logger(config, command_name)
        logger.info(
            "vector index started",
            extra={"event": "vector_index_started", "status": "started", **base_extra},
        )
        result = sync_vector_index(
            config,
            collection=args.collection,
            model=args.model,
            rebuild=args.rebuild,
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except (StorageError, VectorIndexError) as exc:
        if logger is not None:
            logger.exception(
                "vector index failed",
                extra={
                    "event": "vector_index_failed",
                    "status": "failed",
                    **base_extra,
                },
            )
        print(f"Vector index error: {exc}", file=sys.stderr)
        return INDEX_ERROR

    logger.info(
        "vector index completed",
        extra={
            "event": "vector_index_completed",
            "status": "completed",
            "model": result.model,
            "collection": args.collection or "all",
            "command": command_name,
            "count": result.vectors_indexed,
            "rows_removed": result.rows_removed,
            "mappings_removed": result.mappings_removed,
            "vectors_removed": result.vectors_removed,
            "chunks_seen": result.chunks_seen,
        },
    )
    print("Vector index completed")
    _print_vector_index_result(result)
    return SUCCESS


def _handle_search_semantic(args: argparse.Namespace) -> int:
    from .indexing import semantic_search

    command_name = "search semantic"
    logger: Logger | None = None
    query_text = " ".join(args.query)
    top_k: int | None = args.top_k
    base_extra = {
        "command": command_name,
        "course": args.course,
        "indexes": args.indexes or (),
    }
    try:
        config = load_config()
        validate_config(config)
        base_extra["model"] = _embedding_model_log_label(config, args.model)
        top_k = args.top_k if args.top_k is not None else config.semantic_top_k
        logger = _command_logger(config, command_name)
        logger.info(
            "semantic search started",
            extra={
                "event": "semantic_search_started",
                "status": "started",
                "top_k": top_k,
                **base_extra,
            },
        )
        results = semantic_search(
            config,
            query=query_text,
            course=args.course,
            indexes=args.indexes,
            top_k=args.top_k,
            model=args.model,
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR
    except (StorageError, SemanticSearchError) as exc:
        if logger is not None:
            logger.exception(
                "semantic search failed",
                extra={
                    "event": "semantic_search_failed",
                    "status": "failed",
                    "top_k": top_k,
                    "count": 0,
                    "result_count": 0,
                    **base_extra,
                },
            )
        print(f"Semantic search error: {exc}", file=sys.stderr)
        return SEARCH_ERROR

    logger.info(
        "semantic search completed",
        extra={
            "event": "semantic_search_completed",
            "status": "completed",
            "top_k": top_k,
            "count": len(results),
            "result_count": len(results),
            **base_extra,
        },
    )
    if args.json:
        print(
            json.dumps(
                [result.as_safe_dict() for result in results],
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_semantic_search_results(results)
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
            print(
                f"- {course.name}: files={course.file_count}, bytes={course.total_bytes}"
            )
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


def _print_data_summary_run_result(result: DataSummaryRunResult) -> None:
    print(f"run_id: {result.run_id}")
    print(f"status: {result.status}")
    print(f"started_at: {result.started_at}")
    print(f"finished_at: {result.finished_at}")
    print(f"file_id: {result.file_id or 'all'}")
    print(f"files_seen: {result.files_seen}")
    print(f"files_indexed: {result.files_indexed}")
    print(f"files_failed: {result.files_failed}")
    print(f"summaries_created: {result.summaries_created}")
    print(f"chunks_created: {result.chunks_created}")
    _print_count_mapping("by_format", result.by_format)
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


def _print_keyword_index_result(result: KeywordIndexResult) -> None:
    print("mode: rebuild")
    print(f"rows_removed: {result.rows_removed}")
    print(f"chunks_seen: {result.chunks_seen}")
    print(f"rows_indexed: {result.rows_indexed}")
    _print_count_mapping("by_source_type", result.by_source_type)
    if result.diagnostics:
        print("diagnostics:")
        for diagnostic in result.diagnostics:
            print(f"- {diagnostic}")


def _print_keyword_search_results(results: Sequence[RetrievalResult]) -> None:
    if not results:
        print("No keyword results")
        return

    print("rank | score | chunk_id | source/location | course | path | snippet")
    print("---- | ----- | -------- | --------------- | ------ | ---- | -------")
    for result in results:
        location = _format_source_location(result)
        print(
            " | ".join(
                (
                    str(result.rank),
                    f"{result.score:.6g}",
                    str(result.chunk_id),
                    _table_value(location, 24),
                    _table_value(result.course or "", 28),
                    _table_value(result.file_path, 48),
                    _table_value(result.snippet.replace("\n", " "), 80),
                )
            )
        )


def _print_vector_index_result(result: VectorIndexResult) -> None:
    print(f"mode: {'rebuild' if result.rebuild else 'incremental'}")
    print(f"model: {result.model}")
    print(f"provider: {result.provider}")
    print(f"embedding_dim: {result.embedding_dim}")
    print(f"collections: {', '.join(result.collections)}")
    print(f"chunks_seen: {result.chunks_seen}")
    print(f"rows_removed: {result.rows_removed}")
    print(f"mappings_removed: {result.mappings_removed}")
    print(f"vectors_removed: {result.vectors_removed}")
    print(f"vectors_indexed: {result.vectors_indexed}")
    print(f"embeddings_total: {result.embeddings_total}")
    _print_count_mapping("by_source_type", result.by_source_type)
    if result.diagnostics:
        print("diagnostics:")
        for diagnostic in result.diagnostics:
            print(f"- {diagnostic}")


def _print_semantic_search_results(results: Sequence[RetrievalResult]) -> None:
    if not results:
        print("No semantic results")
        return

    print("rank | score | chunk_id | source/location | course | path | snippet")
    print("---- | ----- | -------- | --------------- | ------ | ---- | -------")
    for result in results:
        location = _format_source_location(result)
        print(
            " | ".join(
                (
                    str(result.rank),
                    f"{result.score:.6g}",
                    str(result.chunk_id),
                    _table_value(location, 24),
                    _table_value(result.course or "", 28),
                    _table_value(result.file_path, 48),
                    _table_value(result.snippet.replace("\n", " "), 80),
                )
            )
        )


def _format_source_location(result: RetrievalResult) -> str:
    if result.location_type and result.location_value:
        return f"{result.source_type}:{result.location_type} {result.location_value}"
    if result.location_type:
        return f"{result.source_type}:{result.location_type}"
    return result.source_type


def _table_value(value: str, max_chars: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."


def _command_logger(config: Config, command_name: str) -> Logger:
    return configure_logging(
        level=config.log_level,
        jsonl_path=build_run_log_path(config.runs_dir, command_name),
        console=False,
    )


def _embedding_model_log_label(config: Config, explicit_model: str | None) -> str:
    """Return the normalized model selection represented in command telemetry."""
    explicit = explicit_model.strip() if explicit_model else ""
    return explicit or config.embedding_model or "(unset)"


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
