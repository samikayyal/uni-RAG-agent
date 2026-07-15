"""Command line dispatcher for Uni RAG Agent."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from contextlib import closing
from logging import Logger
from typing import Callable, Protocol

from . import __version__
from .answering import (
    AnswerGenerationError,
    AnswerModelError,
    AnswerResult,
    AnswerSession,
    generate_answer,
    store_answer,
)
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
from .retrieval import (
    EvidenceError,
    EvidenceBuildResult,
    QueryPlanningError,
    RetrievalError,
    RetrievalResult,
    build_evidence,
    load_evidence_packet,
    retrieve,
)
from .retrieval.evidence_models import EvidencePacket
from .retrieval.models import RetrievalRun
from .storage import (
    StorageCheckResult,
    StorageError,
    check_storage,
    connect_sqlite,
    connect_sqlite_read_only,
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
EVIDENCE_ERROR = 8
ANSWER_ERROR = 9

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
  uv run -m uni_rag_agent evidence build "query text"
  uv run -m uni_rag_agent evidence show --search-run-id 1
  uv run -m uni_rag_agent answer --evidence-packet-id 1
  uv run -m uni_rag_agent ask "Explain MapReduce from my courses"
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
    _add_evidence_commands(subparsers)
    _add_answer_commands(subparsers)
    _add_ask_command(subparsers)
    _add_stub_group(
        subparsers,
        "eval",
        "Evaluation commands.",
        {"run": ("eval run", "Feature Spec 12: Evaluation and Hardening")},
    )
    _add_app_commands(subparsers)

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
    retrieve_parser.add_argument(
        "--model",
        help="Supported Hugging Face profile; overrides UNI_RAG_EMBEDDING_MODEL.",
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
    retrieve_parser.set_defaults(handler=_handle_retrieve)


def _add_evidence_commands(subparsers: argparse._SubParsersAction) -> None:
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
        help="Supported Hugging Face profile; overrides UNI_RAG_EMBEDDING_MODEL.",
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
    build_parser.set_defaults(handler=_handle_evidence_build)

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
    show_parser.set_defaults(handler=_handle_evidence_show)


def _add_answer_commands(subparsers: argparse._SubParsersAction) -> None:
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
    answer_parser.set_defaults(handler=_handle_answer)


def _add_ask_command(subparsers: argparse._SubParsersAction) -> None:
    ask_parser = subparsers.add_parser(
        "ask",
        help="Build an evidence packet and answer it in one shot.",
    )
    ask_parser.add_argument("query", nargs="+", help="Query text.")
    ask_parser.add_argument(
        "--model",
        help="Supported Hugging Face embedding profile; overrides UNI_RAG_EMBEDDING_MODEL.",
    )
    ask_parser.add_argument(
        "--json",
        action="store_true",
        help="Print one complete answer result JSON object.",
    )
    ask_parser.set_defaults(handler=_handle_ask)


def _add_app_commands(subparsers: argparse._SubParsersAction) -> None:
    app_parser = subparsers.add_parser(
        "app",
        help="Application server commands.",
    )
    app_subparsers = app_parser.add_subparsers(
        dest="app_command",
        metavar="subcommand",
    )
    app_subparsers.required = True

    serve_parser = app_subparsers.add_parser(
        "serve",
        help="Start the local FastAPI question-answering interface.",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1).",
    )
    serve_parser.add_argument(
        "--port",
        type=_server_port,
        default=8000,
        help="TCP port to bind (default: 8000).",
    )
    serve_parser.set_defaults(handler=_handle_app_serve)


def _server_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _handle_app_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "uni_rag_agent.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
    )
    return SUCCESS


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


def _handle_retrieve(args: argparse.Namespace) -> int:
    command_name = "retrieve"
    query_text = " ".join(args.query)
    logger: Logger | None = None
    model_label = "(unset)"
    try:
        config = load_config()
        validate_config(config)
        model_label = _embedding_model_log_label(config, args.model)
        logger = _command_logger(config, command_name)
        logger.info(
            "retrieval started",
            extra={
                "event": "retrieval_started",
                "command": command_name,
                "status": "started",
                "model": model_label,
            },
        )
        run = retrieve(config, query_text, model=args.model)
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

    _log_retrieval_events(logger, run, model_label, config)
    if args.json:
        print(json.dumps(run.as_safe_dict(), indent=2, sort_keys=True))
    else:
        _print_retrieval_run(run, debug=args.debug)
    return SUCCESS


def _handle_evidence_build(args: argparse.Namespace) -> int:
    command_name = "evidence build"
    query_text = " ".join(args.query)
    logger: Logger | None = None
    model_label = "(unset)"
    try:
        config = load_config()
        validate_config(config)
        model_label = _embedding_model_log_label(config, args.model)
        logger = _command_logger(config, command_name)
        logger.info(
            "evidence build started",
            extra={
                "event": "evidence_build_started",
                "command": command_name,
                "status": "started",
                "model": model_label,
            },
        )
        result = build_evidence(config, query_text, model=args.model)
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

    _log_evidence_events(logger, result, model_label)
    if args.json:
        print(json.dumps(result.as_safe_dict(), indent=2, sort_keys=True))
    else:
        _print_evidence_build_result(result, debug=args.debug, config=config)
    return SUCCESS


def _handle_evidence_show(args: argparse.Namespace) -> int:
    command_name = "evidence show"
    logger: Logger | None = None
    try:
        config = load_config()
        validate_config(config)
        logger = _command_logger(config, command_name)
        packet = load_evidence_packet(config, search_run_id=args.search_run_id)
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
        _print_evidence_packet(packet)
    return SUCCESS


def _handle_answer(args: argparse.Namespace) -> int:
    command_name = "answer"
    logger: Logger | None = None
    try:
        config = load_config()
        validate_config(config)
        logger = _command_logger(config, command_name)
        packet = load_evidence_packet(
            config, evidence_packet_id=args.evidence_packet_id
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
        answer = generate_answer(packet, config=config)
        answer_id = store_answer(
            args.evidence_packet_id,
            answer,
            config=config,
        )
        result = _answer_with_ids(
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

    _log_answer_event(logger, result, command_name)
    _print_answer_result(result, json_output=args.json)
    return SUCCESS


def _handle_ask(args: argparse.Namespace) -> int:
    command_name = "ask"
    query_text = " ".join(args.query)
    logger: Logger | None = None
    model_label = "(unset)"
    packet_id: int | None = None
    try:
        config = load_config()
        validate_config(config)
        model_label = _embedding_model_log_label(config, args.model)
        logger = _command_logger(config, command_name)
        result = build_evidence(config, query_text, model=args.model)
        packet_id = result.evidence_packet_id
        answer = generate_answer(result.packet, config=config)
        answer_id = store_answer(packet_id, answer, config=config)
        answer_result = _answer_with_ids(
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

    _log_answer_event(logger, answer_result, command_name)
    _print_answer_result(answer_result, json_output=args.json)
    return SUCCESS


def _answer_with_ids(
    answer: AnswerResult,
    *,
    answer_id: int,
    evidence_packet_id: int,
    search_run_id: int,
) -> AnswerResult:
    from dataclasses import replace

    return replace(
        answer,
        answer_id=answer_id,
        evidence_packet_id=evidence_packet_id,
        search_run_id=search_run_id,
    )


def _print_answer_result(answer: AnswerResult, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(answer.as_safe_dict(), indent=2, sort_keys=True))
    else:
        print(answer.answer_text)


def _log_answer_event(
    logger: Logger | None,
    answer: AnswerResult,
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


def _log_retrieval_events(
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


def _log_evidence_events(
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
    logger.info(
        "evidence packet created",
        extra={
            "event": "evidence_packet_created",
            "status": result.coverage.status,
            "evidence_count": result.coverage.evidence_count,
            "omitted_count": (
                result.coverage.token_budget_omission_count
                + result.coverage.oversized_evidence_omission_count
            ),
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
            "omitted_count": (
                result.coverage.token_budget_omission_count
                + result.coverage.oversized_evidence_omission_count
            ),
            **common,
        },
    )


def _print_evidence_build_result(
    result: EvidenceBuildResult,
    *,
    debug: bool,
    config: Config,
) -> None:
    packet = result.packet
    coverage = result.coverage
    print(f"status: {coverage.status}")
    print(f"search_run_id: {result.search_run_id}")
    print(f"evidence_packet_id: {result.evidence_packet_id}")
    print(f"query_type: {packet.interpreted_intent}")
    print(f"searched_courses: {', '.join(packet.searched['courses']) or 'none'}")
    print(f"searched_indexes: {', '.join(packet.searched['indexes']) or 'none'}")
    print(f"keyword_terms: {', '.join(packet.searched['keyword_terms']) or 'none'}")
    print(
        f"semantic_queries: {', '.join(packet.searched['semantic_queries']) or 'none'}"
    )
    print(f"raw_result_count: {coverage.raw_result_count}")
    print(f"fused_candidate_count: {coverage.fused_candidate_count}")
    print(f"selectable_candidate_count: {coverage.selectable_candidate_count}")
    print(f"evidence_count: {coverage.evidence_count}")
    print(f"evidence_token_count: {coverage.evidence_token_count}")
    if packet.evidence:
        print("selected_evidence:")
        for item in packet.evidence:
            print(
                f"- rank={item.rank} score={item.score:.6g} course={item.course} "
                f"file={item.file} location={item.location.label} "
                f"tokens={item.token_count} text={_table_value(item.text, 160)}"
            )
    else:
        print("selected_evidence: none")
    _print_evidence_weaknesses(packet.weaknesses)
    if debug:
        _print_persisted_evidence_debug(config, result.search_run_id)


def _print_evidence_packet(packet: EvidencePacket) -> None:
    print(f"search_run_id: {packet.search_run_id}")
    print(f"query: {packet.query}")
    print(f"query_type: {packet.interpreted_intent}")
    print(f"status: {packet.coverage.status}")
    print(f"evidence_count: {packet.coverage.evidence_count}")
    print(f"evidence_token_count: {packet.coverage.evidence_token_count}")
    print("searched:")
    for key in ("courses", "indexes", "keyword_terms", "semantic_queries"):
        print(f"- {key}: {', '.join(packet.searched[key]) or 'none'}")
    if packet.evidence:
        print("evidence:")
        for item in packet.evidence:
            print(
                f"- rank={item.rank} score={item.score:.6g} course={item.course} "
                f"file={item.file} location={item.location.label} "
                f"tokens={item.token_count} text={_table_value(item.text, 160)}"
            )
    else:
        print("evidence: none")
    _print_evidence_weaknesses(packet.weaknesses)


def _print_evidence_weaknesses(weaknesses: Sequence[str]) -> None:
    if not weaknesses:
        print("weaknesses: none")
        return
    print("weaknesses:")
    for weakness in weaknesses:
        print(f"- {weakness}")


def _print_persisted_evidence_debug(config: Config, search_run_id: int) -> None:
    print("debug_persisted_result_rows:")
    with closing(connect_sqlite_read_only(config)) as connection:
        rows = connection.execute(
            """
            SELECT retrieval_method, rank, score, selected_for_evidence, result_json
            FROM search_results
            WHERE search_run_id = ?
            ORDER BY CASE retrieval_method
                         WHEN 'metadata' THEN 1
                         WHEN 'keyword' THEN 2
                         WHEN 'semantic' THEN 3
                         WHEN 'hybrid' THEN 4
                         ELSE 5
                     END,
                     rank,
                     id
            """,
            (search_run_id,),
        ).fetchall()
    for row in rows:
        print(
            f"- method={row['retrieval_method']} rank={row['rank']} "
            f"score={row['score']} selected={row['selected_for_evidence']} "
            f"json={row['result_json']}"
        )


def _print_retrieval_run(run: RetrievalRun, *, debug: bool) -> None:
    plan = run.query_plan
    print(f"status: {run.status}")
    print(f"query: {run.query}")
    print(f"query_type: {plan.query_type}")
    print(f"plan_confidence: {plan.plan_confidence:.6g}")
    print(f"plan_reason: {plan.plan_reason}")
    print(f"embedding_model: {run.embedding_model}")
    print(f"searched_courses: {', '.join(run.searched_courses) or 'none'}")
    print(f"searched_indexes: {', '.join(run.searched_indexes) or 'none'}")
    print(f"keyword_terms: {', '.join(run.keyword_terms) or 'none'}")
    print(f"semantic_queries: {', '.join(run.semantic_queries) or 'none'}")
    if run.results:
        print(
            "rank | rrf_score | course | file | source/location | contributions | snippet"
        )
        print(
            "---- | --------- | ------ | ---- | --------------- | ------------- | -------"
        )
        for result in run.results:
            contribution_summary = ", ".join(
                f"{item.result_set_id}@{item.source_rank}:{item.rrf_contribution:.4f}"
                for item in result.contributions
            )
            print(
                " | ".join(
                    (
                        str(result.rank),
                        f"{result.score:.6g}",
                        _table_value(result.course or "", 28),
                        _table_value(result.file_path, 48),
                        _table_value(_format_source_location(result), 24),
                        _table_value(contribution_summary, 34),
                        _table_value(result.snippet.replace("\n", " "), 80),
                    )
                )
            )
    else:
        print("No fused retrieval results")
    if run.weaknesses:
        print("weaknesses:")
        for weakness in run.weaknesses:
            print(f"- {weakness}")
    if debug:
        print("debug_result_sets:")
        for result_set in run.result_sets:
            print(
                f"- {result_set.result_set_id} ({result_set.retrieval_method}) "
                f"query={result_set.query!r} count={len(result_set.results)}"
            )
            for result in result_set.results:
                print(
                    f"  - rank={result.rank} score={result.score:.6g} "
                    f"chunk_id={result.chunk_id} file_id={result.file_id} "
                    f"path={result.file_path} "
                    f"status={result.file_index_status or '-'} "
                    f"reason={result.reason_not_indexed or '-'} "
                    f"matched={','.join(result.matched_fields) or '-'}"
                )
        print("debug_rrf_contributions:")
        for result in run.results:
            print(
                f"- rank={result.rank} chunk_id={result.chunk_id} file_id={result.file_id}"
            )
            for contribution in result.contributions:
                print(f"  - {json.dumps(contribution.as_safe_dict(), sort_keys=True)}")


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
        prefix = f"{result.source_type}:" if result.source_type else ""
        return f"{prefix}{result.location_type} {result.location_value}"
    if result.location_type:
        prefix = f"{result.source_type}:" if result.source_type else ""
        return f"{prefix}{result.location_type}"
    return result.source_type or "file"


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
