"""Build deterministic data-summary payloads and retrieval text."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

from .._textutils import _json_dumps, _truncate
from ..constants import TEXT_ENCODINGS
from ..models import DataSummary, DataSummarySection

DATA_SCHEMA_CATEGORY = "data_schema"
DATA_SCHEMA_SOURCE_TYPE = "data_schema"
DATA_SCHEMA_EXTRACTOR_NAME = "data-schema-summary"
DATA_SCHEMA_EXTRACTOR_VERSION = "1"
DATA_SCHEMA_EXTENSIONS = {".csv", ".xlsx", ".json", ".jsonl", ".sqlite", ".db"}
SAMPLE_ROW_LIMIT = 5
SAMPLE_VALUE_CHAR_LIMIT = 120
MAX_JSON_FULL_LOAD_BYTES = 2_000_000
JSON_PREVIEW_CHAR_LIMIT = 16_000


def _build_section(
    *,
    name: str,
    kind: str,
    location_value: str,
    row_count: int | None,
    fieldnames: Iterable[str],
    sample_rows: Iterable[Mapping[str, object]],
    column_overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> DataSummarySection:
    normalized_rows = tuple(
        {str(key): _json_safe(value) for key, value in row.items()}
        for row in sample_rows
    )
    names = tuple(fieldnames)
    columns: list[Mapping[str, object]] = []
    overrides = column_overrides or {}
    for column_name in names:
        column_values = [row.get(column_name) for row in normalized_rows]
        payload: dict[str, object] = {
            "name": column_name,
            "type": _infer_column_type(column_values),
        }
        payload.update(overrides.get(column_name, {}))
        columns.append(payload)
    section = DataSummarySection(
        name=name,
        kind=kind,
        location_value=location_value,
        row_count=row_count,
        column_count=len(names),
        columns=tuple(columns),
        sample_rows=normalized_rows,
        summary_text="",
    )
    return replace(section, summary_text=_section_summary_text(section))


def _build_data_summary(
    *,
    file_id: int,
    file_format: str,
    sections: Iterable[DataSummarySection],
    table_count: int | None = None,
    sheet_count: int | None = None,
    metadata: Mapping[str, object] | None = None,
) -> DataSummary:
    section_tuple = tuple(sections)
    known_rows = [section.row_count for section in section_tuple]
    row_count = (
        sum(row for row in known_rows if row is not None)
        if all(row is not None for row in known_rows)
        else None
    )
    column_counts = [section.column_count for section in section_tuple]
    column_count = (
        sum(count for count in column_counts if count is not None)
        if all(count is not None for count in column_counts)
        else None
    )
    schema_payload: dict[str, object] = {
        "format": file_format,
        "sections": [
            {
                "name": section.name,
                "kind": section.kind,
                "location_value": section.location_value,
                "row_count": section.row_count,
                "column_count": section.column_count,
                "columns": list(section.columns),
            }
            for section in section_tuple
        ],
    }
    if metadata:
        schema_payload["metadata"] = dict(metadata)
    sample_payload = {
        "sections": [
            {
                "name": section.name,
                "kind": section.kind,
                "location_value": section.location_value,
                "rows": list(section.sample_rows),
            }
            for section in section_tuple
        ]
    }
    summary_text = _summary_text(
        file_format=file_format,
        row_count=row_count,
        column_count=column_count,
        table_count=table_count,
        sheet_count=sheet_count,
        sections=section_tuple,
    )
    return DataSummary(
        file_id=file_id,
        format=file_format,
        row_count=row_count,
        column_count=column_count,
        table_count=table_count,
        sheet_count=sheet_count,
        schema_json=_json_dumps(schema_payload),
        sample_json=_json_dumps(sample_payload),
        summary_text=summary_text,
        sections=section_tuple,
    )


def _summary_text(
    *,
    file_format: str,
    row_count: int | None,
    column_count: int | None,
    table_count: int | None,
    sheet_count: int | None,
    sections: tuple[DataSummarySection, ...],
) -> str:
    lines = [
        "Data schema summary",
        f"Format: {file_format}",
        f"Rows: {_display_count(row_count)}",
        f"Columns: {_display_count(column_count)}",
    ]
    if table_count is not None:
        lines.append(f"Tables: {table_count}")
    if sheet_count is not None:
        lines.append(f"Sheets: {sheet_count}")
    section_text = "\n\n".join(section.summary_text for section in sections)
    if not section_text:
        return "\n".join(lines)
    return "\n".join(lines) + "\n\n" + section_text


def _section_summary_text(section: DataSummarySection) -> str:
    lines = [
        f"{section.kind.title()}: {section.name}",
        f"Rows: {_display_count(section.row_count)}",
        f"Columns: {_display_count(section.column_count)}",
    ]
    if section.columns:
        column_parts = [
            f"{column['name']} ({column['type']})" for column in section.columns
        ]
        lines.append("Column schema: " + ", ".join(column_parts))
    if section.sample_rows:
        lines.append("Sample rows:")
        for index, row in enumerate(section.sample_rows, start=1):
            lines.append(f"{index}. {_row_summary(row)}")
    return "\n".join(lines)


def _infer_column_type(values: Iterable[object]) -> str:
    inferred = {_infer_scalar_type(value) for value in values}
    inferred.discard("null")
    if not inferred:
        return "null"
    if inferred == {"integer"}:
        return "integer"
    if inferred <= {"integer", "number"}:
        return "number"
    if len(inferred) == 1:
        return next(iter(inferred))
    return "mixed"


def _infer_scalar_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    text = str(value).strip()
    if not text:
        return "null"
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return "boolean"
    try:
        int(text)
    except ValueError:
        pass
    else:
        return "integer"
    try:
        float(text)
    except ValueError:
        pass
    else:
        return "number"
    return "string"


def _read_text_preview(path: Path, max_chars: int) -> str:
    for encoding in TEXT_ENCODINGS:
        try:
            with path.open("r", encoding=encoding) as handle:
                return handle.read(max_chars)
        except UnicodeDecodeError:
            continue
    return ""


def _row_from_sequence(
    headers: list[str],
    values: Iterable[object],
) -> dict[str, object]:
    return {
        header: _json_safe(value)
        for header, value in zip(headers, values, strict=False)
    }


def _clean_column_name(value: object, index: int) -> str:
    text = "" if value is None else str(value).strip()
    return text or f"column_{index}"


def _deduplicate_column_names(names: Iterable[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        count = seen.get(name, 0)
        seen[name] = count + 1
        result.append(name if count == 0 else f"{name}_{count + 1}")
    return result


def _merge_fieldnames(existing: list[str], new_names: Iterable[str]) -> list[str]:
    result = list(existing)
    for name in new_names:
        if name not in result:
            result.append(str(name))
    return result


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate(value, SAMPLE_VALUE_CHAR_LIMIT)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in list(value.items())[:SAMPLE_ROW_LIMIT]
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in list(value)[:SAMPLE_ROW_LIMIT]]
    return _truncate(str(value), SAMPLE_VALUE_CHAR_LIMIT)


def _row_summary(row: Mapping[str, object]) -> str:
    if not row:
        return "<empty row>"
    return "; ".join(
        f"{key}={_display_value(value)}" for key, value in sorted(row.items())
    )


def _display_value(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (dict, list)):
        return _truncate(json.dumps(value, sort_keys=True), SAMPLE_VALUE_CHAR_LIMIT)
    return _truncate(str(value), SAMPLE_VALUE_CHAR_LIMIT)


def _display_count(value: int | None) -> str:
    return "unknown" if value is None else str(value)
