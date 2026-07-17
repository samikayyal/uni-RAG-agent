"""Thin command-line parser and dispatcher for Uni RAG Agent."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from . import __version__
from .answering import (
    AnswerGenerationError,
    AnswerModelError,
    AnswerResult,
    AnswerSession,
    generate_answer,
    store_answer,
)
from .cli_commands import app as app_commands
from .cli_commands import config_storage
from .cli_commands import evaluation as evaluation_commands
from .cli_commands import evidence_answering
from .cli_commands import indexing as indexing_commands
from .cli_commands import ingestion
from .cli_commands import retrieval as retrieval_commands
from .cli_commands.config_storage import ConfigStorageServices
from .cli_commands.evaluation import EvaluationServices
from .cli_commands.evidence_answering import EvidenceAnsweringServices
from .cli_commands.indexing import IndexingServices
from .cli_commands.ingestion import IngestionServices
from .cli_commands.retrieval import RetrievalServices
from .cli_support.constants import (
    ANSWER_ERROR,
    CONFIG_ERROR,
    EVAL_ERROR,
    EVALUATION_ERROR,
    EVIDENCE_ERROR,
    EXTRACTION_ERROR,
    INDEX_ERROR,
    INVENTORY_ERROR,
    NOT_IMPLEMENTED,
    SEARCH_ERROR,
    STORAGE_ERROR,
    SUCCESS,
)
from .cli_support.renderers import (
    format_source_location as _format_source_location,
    print_answer_result as _print_answer_result,
    print_count_mapping as _print_count_mapping,
    print_data_summary_run_result as _print_data_summary_run_result,
    print_evidence_build_result as _print_evidence_build_result,
    print_evidence_packet as _print_evidence_packet,
    print_evidence_weaknesses as _print_evidence_weaknesses,
    print_extraction_run_result as _print_extraction_run_result,
    print_extraction_status as _print_extraction_status,
    print_inventory_run_result as _print_inventory_run_result,
    print_inventory_summary as _print_inventory_summary,
    print_keyword_index_result as _print_keyword_index_result,
    print_keyword_search_results as _print_keyword_search_results,
    print_persisted_evidence_debug as _print_persisted_evidence_debug,
    print_retrieval_run as _print_retrieval_run,
    print_semantic_search_results as _print_semantic_search_results,
    print_storage_result as _print_storage_result,
    print_vector_index_result as _print_vector_index_result,
    table_value as _table_value,
)
from .cli_support.telemetry import (
    command_logger as _command_logger,
    embedding_model_log_label as _embedding_model_log_label,
    log_answer_event as _log_answer_event,
    log_evidence_events as _log_evidence_events,
    log_retrieval_events as _log_retrieval_events,
    run_logged_command as _run_logged_command,
)
from .config import Config, ConfigError, load_config, validate_config
from .evaluation import EvaluationError, prepare_fixture_state, run_eval_set
from .extraction import (
    DataSummaryRunResult,
    ExtractionError,
    ExtractionRunResult,
    ExtractionStatus,
    extract_pending_files,
    load_extraction_status,
    summarize_data_files,
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
    resolve_embedding_profile,
    sync_keyword_index,
)
from .inventory import (
    InventoryError,
    InventoryRunResult,
    InventorySummary,
    inventory_courses,
    load_inventory_summary,
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
from .retrieval.evidence_persistence import sanitize_error
from .retrieval.evidence_models import EvidencePacket
from .storage import (
    StorageCheckResult,
    StorageError,
    check_storage,
    connect_sqlite,
    connect_sqlite_read_only,
    ensure_data_dirs,
    initialize_schema,
)

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
    """Build the public parser by registering each cohesive command family."""

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

    config_storage.register_commands(
        subparsers,
        config_check_handler=_handle_config_check,
        storage_init_handler=_handle_storage_init,
        storage_check_handler=_handle_storage_check,
    )
    ingestion.register_commands(
        subparsers,
        inventory_run_handler=_handle_inventory_run,
        inventory_summary_handler=_handle_inventory_summary,
        extract_run_handler=_handle_extract_run,
        extract_status_handler=_handle_extract_status,
        extract_data_summaries_handler=_handle_extract_data_summaries,
    )
    indexing_commands.register_commands(
        subparsers,
        index_keyword_handler=_handle_index_keyword,
        index_vector_handler=_handle_index_vector,
        search_keyword_handler=_handle_search_keyword,
        search_semantic_handler=_handle_search_semantic,
    )
    retrieval_commands.register_command(subparsers, handler=_handle_retrieve)
    evidence_answering.register_commands(
        subparsers,
        evidence_build_handler=_handle_evidence_build,
        evidence_show_handler=_handle_evidence_show,
        answer_handler=_handle_answer,
        ask_handler=_handle_ask,
    )
    evaluation_commands.register_commands(
        subparsers,
        prepare_handler=_handle_eval_prepare_fixtures,
        run_handler=_handle_eval_run,
    )
    app_commands.register_commands(subparsers, handler=_handle_app_serve)
    return parser


def _config_storage_services() -> ConfigStorageServices:
    return ConfigStorageServices(
        load_config=load_config,
        validate_config=validate_config,
        ensure_data_dirs=ensure_data_dirs,
        connect_sqlite=connect_sqlite,
        initialize_schema=initialize_schema,
        check_storage=check_storage,
        print_storage_result=_print_storage_result,
    )


def _ingestion_services() -> IngestionServices:
    def run_logged_command(**kwargs: object) -> int:
        return _run_logged_command(
            load_config=load_config,
            validate_config=validate_config,
            logger_factory=_command_logger,
            **kwargs,
        )

    return IngestionServices(
        load_config=load_config,
        inventory_courses=inventory_courses,
        load_inventory_summary=load_inventory_summary,
        extract_pending_files=extract_pending_files,
        load_extraction_status=load_extraction_status,
        summarize_data_files=summarize_data_files,
        run_logged_command=run_logged_command,
        print_inventory_run_result=_print_inventory_run_result,
        print_inventory_summary=_print_inventory_summary,
        print_extraction_run_result=_print_extraction_run_result,
        print_extraction_status=_print_extraction_status,
        print_data_summary_run_result=_print_data_summary_run_result,
    )


def _sync_vector_index(*args: object, **kwargs: object) -> object:
    """Keep the optional vector backend lazy while exposing a testable seam."""

    from .indexing import sync_vector_index

    return sync_vector_index(*args, **kwargs)


def _indexing_services() -> IndexingServices:
    return IndexingServices(
        load_config=load_config,
        validate_config=validate_config,
        command_logger=_command_logger,
        keyword_query_terms=keyword_query_terms,
        keyword_search=keyword_search,
        sync_keyword_index=sync_keyword_index,
        semantic_search=_semantic_search,
        sync_vector_index=_sync_vector_index,
        embedding_model_log_label=_embedding_model_log_label,
        print_keyword_index_result=_print_keyword_index_result,
        print_keyword_search_results=_print_keyword_search_results,
        print_vector_index_result=_print_vector_index_result,
        print_semantic_search_results=_print_semantic_search_results,
    )


def _semantic_search(*args: object, **kwargs: object) -> object:
    """Keep the optional vector backend lazy while exposing a testable seam."""

    from .indexing import semantic_search

    return semantic_search(*args, **kwargs)


def _retrieval_services() -> RetrievalServices:
    return RetrievalServices(
        load_config=load_config,
        validate_config=validate_config,
        retrieve=retrieve,
        command_logger=_command_logger,
        embedding_model_log_label=_embedding_model_log_label,
        log_retrieval_events=_log_retrieval_events,
        print_retrieval_run=_print_retrieval_run,
    )


def _evidence_answering_services() -> EvidenceAnsweringServices:
    return EvidenceAnsweringServices(
        load_config=load_config,
        validate_config=validate_config,
        build_evidence=build_evidence,
        load_evidence_packet=load_evidence_packet,
        generate_answer=generate_answer,
        store_answer=store_answer,
        command_logger=_command_logger,
        embedding_model_log_label=_embedding_model_log_label,
        log_evidence_events=_log_evidence_events,
        log_answer_event=_log_answer_event,
        print_evidence_build_result=_print_evidence_build_result,
        print_evidence_packet=_print_evidence_packet,
        print_answer_result=_print_answer_result,
    )


def _evaluation_services() -> EvaluationServices:
    return EvaluationServices(
        load_config=load_config,
        validate_config=validate_config,
        prepare_fixture_state=prepare_fixture_state,
        run_eval_set=run_eval_set,
        sanitize_error=sanitize_error,
    )


def _handle_config_check(args: argparse.Namespace) -> int:
    return config_storage.handle_config_check(
        args,
        services=_config_storage_services(),
    )


def _handle_storage_init(args: argparse.Namespace) -> int:
    return config_storage.handle_storage_init(
        args,
        services=_config_storage_services(),
    )


def _handle_storage_check(args: argparse.Namespace) -> int:
    return config_storage.handle_storage_check(
        args,
        services=_config_storage_services(),
    )


def _handle_inventory_run(args: argparse.Namespace) -> int:
    return ingestion.handle_inventory_run(args, services=_ingestion_services())


def _handle_inventory_summary(args: argparse.Namespace) -> int:
    return ingestion.handle_inventory_summary(args, services=_ingestion_services())


def _handle_extract_run(args: argparse.Namespace) -> int:
    return ingestion.handle_extract_run(args, services=_ingestion_services())


def _handle_extract_status(args: argparse.Namespace) -> int:
    return ingestion.handle_extract_status(args, services=_ingestion_services())


def _handle_extract_data_summaries(args: argparse.Namespace) -> int:
    return ingestion.handle_extract_data_summaries(
        args,
        services=_ingestion_services(),
    )


def _handle_index_keyword(args: argparse.Namespace) -> int:
    return indexing_commands.handle_index_keyword(
        args,
        services=_indexing_services(),
    )


def _handle_search_keyword(args: argparse.Namespace) -> int:
    return indexing_commands.handle_search_keyword(
        args,
        services=_indexing_services(),
    )


def _handle_index_vector(args: argparse.Namespace) -> int:
    return indexing_commands.handle_index_vector(
        args,
        services=_indexing_services(),
    )


def _handle_search_semantic(args: argparse.Namespace) -> int:
    return indexing_commands.handle_search_semantic(
        args,
        services=_indexing_services(),
    )


def _handle_retrieve(args: argparse.Namespace) -> int:
    return retrieval_commands.handle_retrieve(
        args,
        services=_retrieval_services(),
    )


def _handle_evidence_build(args: argparse.Namespace) -> int:
    return evidence_answering.handle_evidence_build(
        args,
        services=_evidence_answering_services(),
    )


def _handle_evidence_show(args: argparse.Namespace) -> int:
    return evidence_answering.handle_evidence_show(
        args,
        services=_evidence_answering_services(),
    )


def _handle_answer(args: argparse.Namespace) -> int:
    return evidence_answering.handle_answer(
        args,
        services=_evidence_answering_services(),
    )


def _handle_ask(args: argparse.Namespace) -> int:
    return evidence_answering.handle_ask(
        args,
        services=_evidence_answering_services(),
    )


def _handle_eval_prepare_fixtures(args: argparse.Namespace) -> int:
    return evaluation_commands.handle_prepare_fixtures(
        args,
        services=_evaluation_services(),
    )


def _handle_eval_run(args: argparse.Namespace) -> int:
    return evaluation_commands.handle_run(args, services=_evaluation_services())


def _handle_app_serve(args: argparse.Namespace) -> int:
    return app_commands.handle_serve(args)


def _answer_with_ids(
    answer: AnswerResult,
    *,
    answer_id: int,
    evidence_packet_id: int,
    search_run_id: int,
) -> AnswerResult:
    return evidence_answering.answer_with_ids(
        answer,
        answer_id=answer_id,
        evidence_packet_id=evidence_packet_id,
        search_run_id=search_run_id,
    )


def _server_port(value: str) -> int:
    return app_commands.server_port(value)


def _not_implemented_handler(
    command_name: str,
    feature_spec: str,
) -> CommandHandler:
    def handler(_: argparse.Namespace) -> int:
        print(
            f"Command '{command_name}' is registered but not implemented yet. "
            f"Expected implementation: {feature_spec}.",
            file=sys.stderr,
        )
        return NOT_IMPLEMENTED

    return handler


__all__ = [
    "ANSWER_ERROR",
    "CONFIG_ERROR",
    "EVAL_ERROR",
    "EVALUATION_ERROR",
    "EVIDENCE_ERROR",
    "EXTRACTION_ERROR",
    "INDEX_ERROR",
    "INVENTORY_ERROR",
    "NOT_IMPLEMENTED",
    "SEARCH_ERROR",
    "STORAGE_ERROR",
    "SUCCESS",
    "build_parser",
    "main",
]
