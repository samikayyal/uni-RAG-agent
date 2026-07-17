"""Local FastAPI application-serving CLI command."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from ..cli_support.constants import SUCCESS


def register_commands(
    subparsers: argparse._SubParsersAction,
    *,
    handler: Callable[[argparse.Namespace], int],
) -> None:
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
        type=server_port,
        default=8000,
        help="TCP port to bind (default: 8000).",
    )
    serve_parser.set_defaults(handler=handler)


def server_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def handle_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "uni_rag_agent.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
    )
    return SUCCESS


__all__ = ["handle_serve", "register_commands", "server_port"]
