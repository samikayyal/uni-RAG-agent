"""Source-code extraction helpers."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

from .._textutils import _read_text_file
from ..constants import (
    CPP_FUNCTION_RE,
    MATLAB_FUNCTION_RE,
    R_FUNCTION_RE,
)
from ..models import RawChunk


def _extract_python(path: Path) -> tuple[RawChunk, ...]:
    source = _read_text_file(path)
    source_lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    raw_chunks: list[RawChunk] = []
    handled_line_numbers: set[int] = set()

    import_nodes = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if import_nodes:
        import_lines = _source_for_nodes(source_lines, import_nodes)
        handled_line_numbers.update(_line_numbers_for_nodes(import_nodes))
        raw_chunks.append(
            RawChunk(
                source_type="code",
                title="Imports",
                text=import_lines,
                location_type="module",
                location_value="imports",
                metadata={"language": "python", "section": "imports"},
            )
        )

    module_docstring = ast.get_docstring(tree)
    if module_docstring:
        module_docstring_node = _module_docstring_node(tree)
        if module_docstring_node is not None:
            handled_line_numbers.update(
                _line_numbers_for_nodes([module_docstring_node])
            )
        raw_chunks.append(
            RawChunk(
                source_type="code",
                title="Module docstring",
                text=module_docstring,
                location_type="module",
                location_value="docstring",
                metadata={"language": "python", "section": "docstring"},
            )
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            handled_line_numbers.update(_line_numbers_for_nodes([node]))
            node_text = _source_for_node(source_lines, node)
            location_type = "class" if isinstance(node, ast.ClassDef) else "function"
            raw_chunks.append(
                RawChunk(
                    source_type="code",
                    title=node.name,
                    text=node_text,
                    location_type=location_type,
                    location_value=node.name,
                    metadata={
                        "language": "python",
                        "line_start": node.lineno,
                        "line_end": getattr(node, "end_lineno", node.lineno),
                    },
                )
            )

    module_lines = [
        line
        for line_number, line in enumerate(source_lines, start=1)
        if line_number not in handled_line_numbers and line.strip()
    ]
    module_text = "\n".join(module_lines).strip()
    if module_text:
        raw_chunks.append(
            RawChunk(
                source_type="code",
                title="Module",
                text=module_text,
                location_type="module",
                location_value="module",
                metadata={"language": "python", "section": "module"},
            )
        )

    return tuple(raw_chunks)


def _extract_other_code(path: Path, extension: str) -> tuple[RawChunk, ...]:
    text = _read_text_file(path)
    language = {
        ".r": "r",
        ".cpp": "cpp",
        ".h": "cpp-header",
        ".m": "matlab",
    }[extension]
    matches = _function_matches_for_code(text, extension)
    if not matches:
        return (
            RawChunk(
                source_type="code",
                title=path.name,
                text=text.strip(),
                location_type="module",
                location_value="module",
                metadata={"language": language, "fallback": "whole_file"},
            ),
        )

    raw_chunks: list[RawChunk] = []
    for index, (name, start, line_number) in enumerate(matches):
        end = matches[index + 1][1] if index + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        if not chunk_text:
            continue
        raw_chunks.append(
            RawChunk(
                source_type="code",
                title=name,
                text=chunk_text,
                location_type="function",
                location_value=name,
                metadata={
                    "language": language,
                    "line_start": line_number,
                    "fallback": "regex",
                },
            )
        )
    return tuple(raw_chunks)


def _source_for_nodes(source_lines: list[str], nodes: Iterable[ast.AST]) -> str:
    return "\n".join(_source_for_node(source_lines, node) for node in nodes).strip()


def _source_for_node(source_lines: list[str], node: ast.AST) -> str:
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start)
    return "\n".join(source_lines[start - 1 : end]).strip()


def _module_docstring_node(tree: ast.Module) -> ast.Expr | None:
    if not tree.body:
        return None
    node = tree.body[0]
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
        if isinstance(node.value.value, str):
            return node
    return None


def _line_numbers_for_nodes(nodes: Iterable[ast.AST]) -> set[int]:
    line_numbers: set[int] = set()
    for node in nodes:
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start)
        line_numbers.update(range(start, end + 1))
    return line_numbers


def _function_matches_for_code(
    text: str,
    extension: str,
) -> list[tuple[str, int, int]]:
    if extension == ".r":
        regex = R_FUNCTION_RE
    elif extension in {".cpp", ".h"}:
        regex = CPP_FUNCTION_RE
    else:
        regex = MATLAB_FUNCTION_RE

    matches: list[tuple[str, int, int]] = []
    for match in regex.finditer(text):
        name = match.groupdict().get("name") or match.group(1)
        line_number = text.count("\n", 0, match.start()) + 1
        matches.append((name, match.start(), line_number))
    return matches
