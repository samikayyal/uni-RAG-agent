"""Keyword/vector indexing and direct-search CLI commands."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..cli_support.constants import CONFIG_ERROR, INDEX_ERROR, SEARCH_ERROR, SUCCESS
from ..config import ConfigError
from ..indexing import (
    KeywordIndexError,
    KeywordSearchError,
    SemanticSearchError,
    VectorIndexError,
)
from ..storage import StorageError


@dataclass(frozen=True)
class IndexingServices:
    load_config: Callable[[], Any]
    validate_config: Callable[[Any], None]
    command_logger: Callable[[Any, str], Any]
    keyword_query_terms: Callable[[str], tuple[str, ...]]
    keyword_search: Callable[..., Any]
    sync_keyword_index: Callable[..., Any]
    semantic_search: Callable[..., Any]
    sync_vector_index: Callable[..., Any]
    embedding_model_log_label: Callable[[Any, str | None], str]
    print_keyword_index_result: Callable[[Any], None]
    print_keyword_search_results: Callable[[Any], None]
    print_vector_index_result: Callable[[Any], None]
    print_semantic_search_results: Callable[[Any], None]


def register_commands(
    subparsers: argparse._SubParsersAction,
    *,
    index_keyword_handler: Callable[[argparse.Namespace], int],
    index_vector_handler: Callable[[argparse.Namespace], int],
    search_keyword_handler: Callable[[argparse.Namespace], int],
    search_semantic_handler: Callable[[argparse.Namespace], int],
) -> None:
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
    keyword_parser.set_defaults(handler=index_keyword_handler)

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
            "Supported reviewed local or hosted embedding profile; falls back to "
            "UNI_RAG_EMBEDDING_MODEL when omitted."
        ),
    )
    vector_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear and repopulate only the selected model/profile and collection.",
    )
    vector_parser.set_defaults(handler=index_vector_handler)

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
    keyword_parser.set_defaults(handler=search_keyword_handler)

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
            "Supported reviewed local or hosted embedding profile; falls back to "
            "UNI_RAG_EMBEDDING_MODEL when omitted. Must match the indexed model."
        ),
    )
    semantic_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON result objects instead of a table.",
    )
    semantic_parser.set_defaults(handler=search_semantic_handler)


def handle_index_keyword(
    args: argparse.Namespace,
    *,
    services: IndexingServices,
) -> int:
    command_name = "index keyword"
    logger = None
    try:
        config = services.load_config()
        services.validate_config(config)
        logger = services.command_logger(config, command_name)
        logger.info(
            "keyword index started",
            extra={
                "event": "keyword_index_started",
                "command": command_name,
                "status": "started",
            },
        )
        result = services.sync_keyword_index(config, rebuild=args.rebuild)
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
    services.print_keyword_index_result(result)
    return SUCCESS


def handle_search_keyword(
    args: argparse.Namespace,
    *,
    services: IndexingServices,
) -> int:
    command_name = "search keyword"
    logger = None
    query_text = " ".join(args.query)
    keyword_terms: tuple[str, ...] = ()
    top_k: int | None = args.top_k
    try:
        config = services.load_config()
        services.validate_config(config)
        top_k = args.top_k if args.top_k is not None else config.keyword_top_k
        logger = services.command_logger(config, command_name)
        keyword_terms = services.keyword_query_terms(query_text)
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
        results = services.keyword_search(
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
        services.print_keyword_search_results(results)
    return SUCCESS


def handle_index_vector(
    args: argparse.Namespace,
    *,
    services: IndexingServices,
) -> int:
    command_name = "index vector"
    logger = None
    base_extra = {
        "command": command_name,
        "collection": args.collection or "all",
    }
    try:
        config = services.load_config()
        services.validate_config(config)
        base_extra["model"] = services.embedding_model_log_label(config, args.model)
        logger = services.command_logger(config, command_name)
        logger.info(
            "vector index started",
            extra={"event": "vector_index_started", "status": "started", **base_extra},
        )
        result = services.sync_vector_index(
            config,
            collection=args.collection,
            model=args.model,
            rebuild=args.rebuild,
            show_progress=True,
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
    services.print_vector_index_result(result)
    return SUCCESS


def handle_search_semantic(
    args: argparse.Namespace,
    *,
    services: IndexingServices,
) -> int:
    command_name = "search semantic"
    logger = None
    query_text = " ".join(args.query)
    top_k: int | None = args.top_k
    base_extra = {
        "command": command_name,
        "course": args.course,
        "indexes": args.indexes or (),
    }
    try:
        config = services.load_config()
        services.validate_config(config)
        base_extra["model"] = services.embedding_model_log_label(config, args.model)
        top_k = args.top_k if args.top_k is not None else config.semantic_top_k
        logger = services.command_logger(config, command_name)
        logger.info(
            "semantic search started",
            extra={
                "event": "semantic_search_started",
                "status": "started",
                "top_k": top_k,
                **base_extra,
            },
        )
        results = services.semantic_search(
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
        services.print_semantic_search_results(results)
    return SUCCESS


__all__ = [
    "IndexingServices",
    "handle_index_keyword",
    "handle_index_vector",
    "handle_search_keyword",
    "handle_search_semantic",
    "register_commands",
]
