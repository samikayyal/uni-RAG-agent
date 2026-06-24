"""Command line dispatcher for Uni RAG Agent."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Callable

from . import __version__
from .config import ConfigError, load_settings

SUCCESS = 0
NOT_IMPLEMENTED = 1
CONFIG_ERROR = 2

CommandHandler = Callable[[argparse.Namespace], int]

COMMAND_EXAMPLES = """\
Available command shapes:
  uv run -m uni_rag_agent config check
  uv run -m uni_rag_agent storage init
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
    _add_stub_group(
        subparsers,
        "storage",
        "Storage and schema commands.",
        {"init": ("storage init", "Feature Spec 02: Configuration and Storage")},
    )
    _add_stub_group(
        subparsers,
        "inventory",
        "Course archive inventory commands.",
        {"run": ("inventory run", "Feature Spec 03: Inventory and File Classification")},
    )
    _add_stub_group(
        subparsers,
        "extract",
        "Text extraction commands.",
        {"run": ("extract run", "Feature Spec 04: Text Extraction and Chunking")},
    )
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
    config_subparsers = config_parser.add_subparsers(dest="config_command", metavar="subcommand")
    config_subparsers.required = True

    check_parser = config_subparsers.add_parser(
        "check",
        help="Load configuration and print non-secret resolved values.",
    )
    check_parser.set_defaults(handler=_handle_config_check)


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
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR

    print("Configuration OK")
    for key, value in settings.as_safe_dict().items():
        print(f"{key}: {value}")
    return SUCCESS


def _not_implemented_handler(command_name: str, feature_spec: str) -> CommandHandler:
    def handler(_: argparse.Namespace) -> int:
        print(
            f"Command '{command_name}' is registered but not implemented yet. "
            f"Expected implementation: {feature_spec}.",
            file=sys.stderr,
        )
        return NOT_IMPLEMENTED

    return handler
