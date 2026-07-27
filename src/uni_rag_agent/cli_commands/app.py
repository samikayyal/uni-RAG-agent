"""Local FastAPI application-serving CLI command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from ..cli_support.constants import CONFIG_ERROR, SUCCESS


def register_commands(
    subparsers: argparse._SubParsersAction,
    *,
    serve_handler: Callable[[argparse.Namespace], int],
    dashboard_handler: Callable[[argparse.Namespace], int],
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
    serve_parser.set_defaults(handler=serve_handler)

    dashboard_parser = app_subparsers.add_parser(
        "audit-dashboard",
        help="View durable public ask records on a localhost-only dashboard.",
    )
    dashboard_parser.add_argument(
        "--port",
        type=server_port,
        default=8001,
        help="Loopback TCP port to bind (default: 8001).",
    )
    dashboard_parser.add_argument(
        "--geoip-db",
        type=Path,
        default=Path("data/GeoLite2-City.mmdb"),
        help=(
            "Local MaxMind GeoLite2 City .mmdb path (default: data/GeoLite2-City.mmdb)."
        ),
    )
    dashboard_parser.add_argument(
        "--cache-db",
        type=Path,
        default=Path("data/ask_audit_cache.sqlite"),
        help="Generated local SQLite cache path (default: data/ask_audit_cache.sqlite).",
    )
    dashboard_parser.add_argument(
        "--offline",
        action="store_true",
        help="Serve the existing SQLite cache without initializing Firestore.",
    )
    dashboard_parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild the local cache from every Firestore audit record.",
    )
    dashboard_parser.add_argument(
        "--refresh-locations",
        action="store_true",
        help="Re-resolve every cached IP using the current local GeoLite2 database.",
    )
    dashboard_parser.set_defaults(handler=dashboard_handler)


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


def handle_audit_dashboard(args: argparse.Namespace) -> int:
    import uvicorn

    from ..app.audit_dashboard import create_audit_dashboard

    if args.offline and args.rebuild_cache:
        print("--rebuild-cache cannot be used with --offline.", file=sys.stderr)
        return CONFIG_ERROR
    app = create_audit_dashboard(
        geoip_database=args.geoip_db.resolve(),
        cache_database=args.cache_db.resolve(),
        offline=args.offline,
        rebuild_cache=args.rebuild_cache,
        refresh_locations=args.refresh_locations,
    )
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
    )
    return SUCCESS


__all__ = [
    "handle_audit_dashboard",
    "handle_serve",
    "register_commands",
    "server_port",
]
