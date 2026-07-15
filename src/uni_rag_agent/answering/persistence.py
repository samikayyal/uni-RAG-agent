"""Append-only answer trace persistence."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timezone

from ..config import Config, load_config
from ..retrieval import EvidenceError, load_evidence_packet
from ..retrieval.evidence_models import canonical_json
from ..storage import StorageError, connect_sqlite, connect_sqlite_read_only
from .models import AnswerCitation, AnswerModelError, AnswerResult
from .core import validate_answer_for_storage


def store_answer(
    evidence_packet_id: int | Config,
    answer: AnswerResult | int,
    maybe_answer: AnswerResult | None = None,
    *,
    config: Config | None = None,
    commit_guard: Callable[[Callable[[], None]], None] | None = None,
) -> int:
    """Insert one completed answer trace and return its generated id.

    The documented form is ``store_answer(evidence_packet_id, answer,
    config=config)``. For callers following the repository's config-first
    service convention, ``store_answer(config, evidence_packet_id, answer)`` is
    accepted as a compatibility form.
    """
    if isinstance(evidence_packet_id, Config):
        resolved_config = evidence_packet_id
        if not isinstance(answer, int) or maybe_answer is None:
            raise TypeError(
                "config-first store_answer requires (config, packet_id, answer)"
            )
        packet_id = answer
        resolved_answer = maybe_answer
    else:
        packet_id = evidence_packet_id
        if isinstance(maybe_answer, Config):
            # Also accept store_answer(packet_id, answer, config) for simple
            # service-call sites that avoid keyword arguments.
            resolved_config = maybe_answer
        else:
            resolved_config = config or load_config()
        if not isinstance(answer, AnswerResult):
            raise TypeError("store_answer requires an AnswerResult")
        resolved_answer = answer
    if not isinstance(packet_id, int) or isinstance(packet_id, bool) or packet_id <= 0:
        raise AnswerModelError("evidence_packet_id must be a positive integer")
    if not isinstance(resolved_answer, AnswerResult):
        raise TypeError("store_answer requires an AnswerResult")
    try:
        packet = load_evidence_packet(
            resolved_config,
            evidence_packet_id=packet_id,
        )
    except EvidenceError as exc:
        raise StorageError(
            f"Could not validate evidence packet {packet_id} for answer storage."
        ) from exc
    validate_answer_for_storage(resolved_answer, packet, resolved_config)
    try:
        with closing(connect_sqlite(resolved_config)) as connection:
            packet_row = connection.execute(
                "SELECT id FROM evidence_packets WHERE id = ?", (packet_id,)
            ).fetchone()
            if packet_row is None:
                raise StorageError(f"Evidence packet {packet_id} does not exist.")
            cursor = connection.execute(
                """
                INSERT INTO answers (
                    evidence_packet_id,
                    answer_text,
                    citations_json,
                    limitations_json,
                    model_name,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    packet_id,
                    resolved_answer.answer_text,
                    canonical_json(
                        [item.as_safe_dict() for item in resolved_answer.citations]
                    ),
                    canonical_json(list(resolved_answer.limitations)),
                    resolved_answer.model_name,
                    _utc_now(),
                ),
            )
            answer_id = int(cursor.lastrowid or 0)
            if answer_id <= 0:
                raise StorageError("Could not create answer: missing identifier")
            if commit_guard is None:
                connection.commit()
            else:
                commit_guard(connection.commit)
            return answer_id
    except StorageError:
        raise
    except sqlite3.Error as exc:
        raise StorageError(f"Could not persist answer: {exc}") from exc


def load_answer(config: Config, answer_id: int) -> AnswerResult:
    """Load a persisted answer trace without exposing prompts or raw output."""
    if not isinstance(answer_id, int) or isinstance(answer_id, bool) or answer_id <= 0:
        raise StorageError("answer_id must be a positive integer")
    try:
        with closing(connect_sqlite_read_only(config)) as connection:
            row = connection.execute(
                """
                SELECT id, evidence_packet_id, answer_text, citations_json,
                       limitations_json, model_name
                FROM answers WHERE id = ?
                """,
                (answer_id,),
            ).fetchone()
    except sqlite3.Error as exc:
        raise StorageError(f"Could not load answer: {exc}") from exc
    if row is None:
        raise StorageError(f"Answer {answer_id} does not exist.")
    try:
        citation_values = json.loads(row["citations_json"])
        limitation_values = json.loads(row["limitations_json"] or "[]")
        if not isinstance(citation_values, list):
            raise AnswerModelError("persisted citations must be a JSON array")
        if not isinstance(limitation_values, list) or any(
            not isinstance(value, str) or not value.strip()
            for value in limitation_values
        ):
            raise AnswerModelError("persisted limitations must be a JSON string array")
        citations = tuple(_citation_from_dict(value) for value in citation_values)
        limitations = tuple(limitation_values)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StorageError(f"Persisted answer {answer_id} is invalid: {exc}") from exc
    return AnswerResult(
        answer_text=str(row["answer_text"]),
        citations=citations,
        limitations=limitations,
        model_name=row["model_name"],
        answer_id=int(row["id"]),
        evidence_packet_id=int(row["evidence_packet_id"]),
    )


def _citation_from_dict(value: object) -> AnswerCitation:
    if not isinstance(value, dict):
        raise AnswerModelError("persisted citation must be an object")
    required = {
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
    if set(value) != required:
        raise AnswerModelError("persisted citation has unexpected fields")
    return AnswerCitation(
        citation_id=value["citation_id"],
        evidence_index=value["evidence_index"],
        course=value["course"],
        file_id=value["file_id"],
        chunk_id=value["chunk_id"],
        file_path=value["file_path"],
        source_type=value["source_type"],
        location_type=value["location_type"],
        location_value=value["location_value"],
        location_label=value["location_label"],
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
