"""Read-only audit helpers for persisted Feature 10 answer traces."""

from __future__ import annotations

import json
import re
from typing import Any

_CITATION_KEYS = {
    "citation_id",
    "evidence_index",
    "course",
    "file_id",
    "chunk_id",
    "file_path",
    "source_type",
    "location_type",
    "location_value",
    "location_label",
}
_MARKER_RE = re.compile(r"\[(E[1-9][0-9]*)\]")


def audit_stored_answer(
    citations_json: object,
    packet_json: object,
    answer_text: object,
) -> dict[str, object]:
    """Validate persisted citations against the immutable packet and rendering."""
    citations, citations_error = _parse_json(citations_json)
    packet, packet_error = _parse_json(packet_json)
    result: dict[str, object] = {
        "valid": False,
        "diagnostic": "",
        "citations": citations if isinstance(citations, list) else [],
        "citations_parsed": citations_error is None,
        "packet_parsed": packet_error is None,
    }
    if citations_error:
        result["diagnostic"] = f"citations_json {citations_error}"
        return result
    if packet_error:
        result["diagnostic"] = f"packet_json {packet_error}"
        return result
    if not isinstance(citations, list):
        result["diagnostic"] = "citations_json must contain a JSON array"
        return result
    if not isinstance(packet, dict) or not isinstance(packet.get("evidence"), list):
        result["diagnostic"] = "packet_json must contain an evidence array"
        return result
    if not isinstance(answer_text, str):
        result["diagnostic"] = "answer_text must be a string"
        return result

    evidence = packet["evidence"]
    expected_ids: list[str] = []
    expected_reference_lines: list[str] = []
    for position, citation in enumerate(citations, start=1):
        if not isinstance(citation, dict) or set(citation) != _CITATION_KEYS:
            result["diagnostic"] = (
                f"citation {position} must contain the complete canonical field set"
            )
            return result
        evidence_index = citation.get("evidence_index")
        if (
            not isinstance(evidence_index, int)
            or isinstance(evidence_index, bool)
            or not 1 <= evidence_index <= len(evidence)
        ):
            result["diagnostic"] = f"citation {position} has an invalid evidence_index"
            return result
        item = evidence[evidence_index - 1]
        expected = _expected_citation(evidence_index, item)
        if expected is None or citation != expected:
            result["diagnostic"] = (
                f"citation {position} does not match packet evidence E{evidence_index}"
            )
            return result
        citation_id = expected["citation_id"]
        if citation_id in expected_ids:
            result["diagnostic"] = f"duplicate structured citation: {citation_id}"
            return result
        expected_ids.append(citation_id)
        expected_reference_lines.append(
            f"- [{citation_id}] {expected['course']} - {expected['file_path']} - "
            f"{expected['location_label']}"
        )

    render_error = _rendering_error(
        answer_text,
        expected_ids,
        expected_reference_lines,
    )
    if render_error:
        result["diagnostic"] = render_error
        return result
    result["valid"] = True
    result["diagnostic"] = "valid"
    return result


def _parse_json(value: object) -> tuple[Any, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, "is missing or blank"
    try:
        return json.loads(value), None
    except (TypeError, json.JSONDecodeError):
        return None, "is malformed"


def _expected_citation(
    evidence_index: int,
    item: object,
) -> dict[str, object] | None:
    if not isinstance(item, dict) or not isinstance(item.get("location"), dict):
        return None
    location = item["location"]
    expected = {
        "citation_id": f"E{evidence_index}",
        "evidence_index": evidence_index,
        "course": item.get("course"),
        "file_id": item.get("file_id"),
        "chunk_id": item.get("chunk_id"),
        "file_path": item.get("file"),
        "source_type": item.get("source_type"),
        "location_type": location.get("type"),
        "location_value": location.get("value"),
        "location_label": location.get("label"),
    }
    if any(
        not isinstance(expected[name], str) or not str(expected[name]).strip()
        for name in ("course", "file_path", "source_type", "location_label")
    ):
        return None
    if any(
        not isinstance(expected[name], int)
        or isinstance(expected[name], bool)
        or int(expected[name]) <= 0
        for name in ("file_id", "chunk_id")
    ):
        return None
    for name in ("location_type", "location_value"):
        value = expected[name]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            return None
    return expected


def _rendering_error(
    answer_text: str,
    expected_ids: list[str],
    expected_reference_lines: list[str],
) -> str | None:
    body, separator, tail = answer_text.partition("\n\nReferences:\n")
    marker_ids: list[str] = []
    for citation_id in _MARKER_RE.findall(body):
        if citation_id not in marker_ids:
            marker_ids.append(citation_id)
    if marker_ids != expected_ids:
        return "rendered citation markers do not match structured citation order"
    if not expected_ids:
        if separator:
            return "uncited answer must not contain a References section"
        return None
    if not separator:
        return "cited answer is missing the canonical References section"
    references_text = tail.split("\n\nLimitations:\n", 1)[0]
    if references_text.splitlines() != expected_reference_lines:
        return "rendered References section does not match structured citations"
    return None
