"""Jupyter notebook extraction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..constants import NOTEBOOK_OUTPUT_CHAR_LIMIT
from ..models import RawChunk


def _extract_notebook(path: Path) -> tuple[RawChunk, ...]:
    import nbformat

    notebook = nbformat.read(path, as_version=4)
    raw_chunks: list[RawChunk] = []
    for cell_index, cell in enumerate(notebook.cells, start=1):
        cell_type = cell.get("cell_type")
        if cell_type not in {"markdown", "code"}:
            continue
        source = str(cell.get("source", "")).strip()
        text = source
        output_text = ""
        if cell_type == "code":
            output_text = _notebook_output_text(cell.get("outputs", []))
            if output_text:
                text = f"{source}\n\nOutput:\n{output_text}".strip()
        if not text:
            continue
        raw_chunks.append(
            RawChunk(
                source_type="notebook",
                title=f"{cell_type} cell {cell_index}",
                text=text,
                location_type="notebook_cell",
                location_value=str(cell_index),
                metadata={
                    "cell": cell_index,
                    "cell_type": cell_type,
                    "output_truncated_to_chars": (
                        NOTEBOOK_OUTPUT_CHAR_LIMIT if output_text else None
                    ),
                },
            )
        )
    return tuple(raw_chunks)


def _notebook_output_text(outputs: Iterable[Mapping[str, Any]]) -> str:
    pieces: list[str] = []
    for output in outputs:
        output_type = output.get("output_type")
        if output_type == "stream":
            text = output.get("text", "")
            pieces.append(_stringify_notebook_text(text))
        elif output_type in {"execute_result", "display_data"}:
            data = output.get("data", {})
            if isinstance(data, Mapping) and "text/plain" in data:
                pieces.append(_stringify_notebook_text(data["text/plain"]))
        elif output_type == "error":
            trace_lines = output.get("traceback", [])
            pieces.append(_stringify_notebook_text(trace_lines))

    text = "\n".join(piece for piece in pieces if piece.strip()).strip()
    return _truncate(text, NOTEBOOK_OUTPUT_CHAR_LIMIT)


def _stringify_notebook_text(value: object) -> str:
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    return str(value)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."
