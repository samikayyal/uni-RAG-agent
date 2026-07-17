"""Configuration and generated-storage CLI commands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from ..config import ConfigError
from ..storage import StorageError
from ..cli_support.constants import CONFIG_ERROR, STORAGE_ERROR, SUCCESS


@dataclass(frozen=True)
class ConfigStorageServices:
    load_config: Callable[[], Any]
    validate_config: Callable[[Any], None]
    ensure_data_dirs: Callable[[Any], None]
    connect_sqlite: Callable[[Any], AbstractContextManager[Any]]
    initialize_schema: Callable[[Any], None]
    check_storage: Callable[[Any], Any]
    print_storage_result: Callable[[Any], None]


def register_commands(
    subparsers: argparse._SubParsersAction,
    *,
    config_check_handler: Callable[[argparse.Namespace], int],
    storage_init_handler: Callable[[argparse.Namespace], int],
    storage_check_handler: Callable[[argparse.Namespace], int],
) -> None:
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
    check_parser.set_defaults(handler=config_check_handler)

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
    init_parser.set_defaults(handler=storage_init_handler)

    check_parser = storage_subparsers.add_parser(
        "check",
        help="Check generated storage directories, SQLite schema, and FTS5 support.",
    )
    check_parser.set_defaults(handler=storage_check_handler)


def handle_config_check(
    _: argparse.Namespace,
    *,
    services: ConfigStorageServices,
) -> int:
    try:
        config = services.load_config()
        services.validate_config(config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR

    print("Configuration OK")
    for key, value in config.as_safe_dict().items():
        print(f"{key}: {value}")
    return SUCCESS


def handle_storage_init(
    _: argparse.Namespace,
    *,
    services: ConfigStorageServices,
) -> int:
    try:
        config = services.load_config()
        services.ensure_data_dirs(config)
        with services.connect_sqlite(config) as connection:
            services.initialize_schema(connection)
        result = services.check_storage(config)
    except (ConfigError, StorageError) as exc:
        print(f"Storage initialization error: {exc}", file=sys.stderr)
        return STORAGE_ERROR

    print("Storage initialized")
    services.print_storage_result(result)
    return SUCCESS if result.ok else STORAGE_ERROR


def handle_storage_check(
    _: argparse.Namespace,
    *,
    services: ConfigStorageServices,
) -> int:
    try:
        config = services.load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return CONFIG_ERROR

    result = services.check_storage(config)
    print("Storage OK" if result.ok else "Storage check failed")
    services.print_storage_result(result)
    return SUCCESS if result.ok else STORAGE_ERROR


__all__ = [
    "ConfigStorageServices",
    "handle_config_check",
    "handle_storage_check",
    "handle_storage_init",
    "register_commands",
]
