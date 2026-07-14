"""Strict answer-generation contracts for Feature 10.

The model-facing shape intentionally stays smaller than the persisted shape. A
chat model returns only paragraphs, citation ids, and limitations; the
application resolves ids to immutable packet evidence and renders references.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..retrieval.evidence_models import EvidenceItem, EvidencePacket, EvidenceModelError

_CITATION_ID_RE = re.compile(r"^E([1-9][0-9]*)$")
_CITATION_MARKER_RE = re.compile(r"\[\s*e\s*\d+\s*\]", re.IGNORECASE)


class AnswerModelError(ValueError):
    """Raised when a generated answer does not satisfy the strict contract."""


class AnswerGenerationError(RuntimeError):
    """Raised for provider construction or invocation failures."""


# Short aliases keep the public boundary ergonomic without creating a second
# exception taxonomy.
AnswerError = AnswerGenerationError
AnswerValidationError = AnswerModelError


@dataclass(frozen=True)
class AnswerParagraph:
    text: str
    citation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise AnswerModelError("answer paragraph text must be nonblank")
        if contains_citation_like_marker(self.text):
            raise AnswerModelError(
                "answer paragraph text must not contain citation markers"
            )
        if not isinstance(self.citation_ids, tuple):
            raise AnswerModelError("citation_ids must be a tuple")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.citation_ids
        ):
            raise AnswerModelError("citation_ids must contain nonblank strings")

    def as_safe_dict(self) -> dict[str, object]:
        return {"text": self.text, "citation_ids": list(self.citation_ids)}


@dataclass(frozen=True)
class AnswerCitation:
    """A structured citation resolved from one packet evidence position."""

    citation_id: str
    evidence_index: int
    course: str
    file_id: int
    chunk_id: int
    file_path: str
    source_type: str
    location_type: str | None
    location_value: str | None
    location_label: str

    def __post_init__(self) -> None:
        if not _CITATION_ID_RE.fullmatch(self.citation_id):
            raise AnswerModelError("citation_id must match E<number>")
        if self.evidence_index <= 0 or self.file_id <= 0 or self.chunk_id <= 0:
            raise AnswerModelError("citation identifiers must be positive")
        for name in ("course", "file_path", "source_type", "location_label"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise AnswerModelError(f"{name} must be nonblank")
        for name in ("location_type", "location_value"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise AnswerModelError(f"{name} must be null or nonblank")

    @classmethod
    def from_evidence(cls, evidence_index: int, item: EvidenceItem) -> "AnswerCitation":
        return cls(
            citation_id=f"E{evidence_index}",
            evidence_index=evidence_index,
            course=item.course,
            file_id=item.file_id,
            chunk_id=item.chunk_id,
            file_path=item.file,
            source_type=item.source_type,
            location_type=item.location.type,
            location_value=item.location.value,
            location_label=item.location.label,
        )

    def as_safe_dict(self) -> dict[str, object]:
        # Keep these exact names stable: answers.citations_json is an audit
        # projection, not a rendered string.
        return {
            "citation_id": self.citation_id,
            "evidence_index": self.evidence_index,
            "course": self.course,
            "file_id": self.file_id,
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "source_type": self.source_type,
            "location_type": self.location_type,
            "location_value": self.location_value,
            "location_label": self.location_label,
        }


@dataclass(frozen=True)
class CitationValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()
    citations: tuple[AnswerCitation, ...] = ()
    paragraphs: tuple[AnswerParagraph, ...] = ()

    def __bool__(self) -> bool:
        return self.valid

    def as_safe_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "citations": [citation.as_safe_dict() for citation in self.citations],
            "paragraphs": [paragraph.as_safe_dict() for paragraph in self.paragraphs],
        }


@dataclass(frozen=True)
class AnswerResult:
    """A fully rendered, validated answer and its structured audit fields."""

    answer_text: str
    citations: tuple[AnswerCitation, ...] = ()
    limitations: tuple[str, ...] = ()
    model_name: str | None = None
    paragraphs: tuple[AnswerParagraph, ...] = ()
    answer_id: int | None = None
    evidence_packet_id: int | None = None
    search_run_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.answer_text, str) or not self.answer_text.strip():
            raise AnswerModelError("answer_text must be nonblank")
        if self.model_name is not None and (
            not isinstance(self.model_name, str) or not self.model_name.strip()
        ):
            raise AnswerModelError("model_name must be null or nonblank")
        if any(not isinstance(value, AnswerCitation) for value in self.citations):
            raise AnswerModelError("citations must contain AnswerCitation values")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.limitations
        ):
            raise AnswerModelError("limitations must contain nonblank strings")

    def as_safe_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "answer_text": self.answer_text,
            "citations": [citation.as_safe_dict() for citation in self.citations],
            "limitations": list(self.limitations),
            "model_name": self.model_name,
        }
        if self.answer_id is not None:
            payload["answer_id"] = self.answer_id
        if self.evidence_packet_id is not None:
            payload["evidence_packet_id"] = self.evidence_packet_id
        if self.search_run_id is not None:
            payload["search_run_id"] = self.search_run_id
        return payload


def parse_model_answer(value: object) -> tuple[AnswerParagraph, ...]:
    """Parse exactly the model JSON object required by the answer contract."""
    if not isinstance(value, Mapping) or set(value) != {
        "answer_paragraphs",
        "limitations",
    }:
        raise AnswerModelError(
            "answer model output must contain exactly answer_paragraphs and limitations"
        )
    raw_paragraphs = value["answer_paragraphs"]
    raw_limitations = value["limitations"]
    if not isinstance(raw_paragraphs, list):
        raise AnswerModelError("answer_paragraphs must be a JSON array")
    if not isinstance(raw_limitations, list) or any(
        not isinstance(item, str) or not item.strip() for item in raw_limitations
    ):
        raise AnswerModelError("limitations must be a JSON string array")
    paragraphs: list[AnswerParagraph] = []
    for value in raw_paragraphs:
        if not isinstance(value, Mapping) or set(value) != {"text", "citation_ids"}:
            raise AnswerModelError(
                "each answer paragraph must contain exactly text and citation_ids"
            )
        text = value["text"]
        citation_ids = value["citation_ids"]
        if not isinstance(text, str) or not text.strip():
            raise AnswerModelError("answer paragraph text must be nonblank")
        if not isinstance(citation_ids, list) or any(
            not isinstance(item, str) or not item.strip() for item in citation_ids
        ):
            raise AnswerModelError("citation_ids must be a JSON string array")
        paragraphs.append(
            AnswerParagraph(text=text.strip(), citation_ids=tuple(citation_ids))
        )
    # Attach limitations separately in the generation core. Returning only the
    # paragraph tuple keeps this parser useful to callers that validate raw JSON.
    return tuple(paragraphs)


def parse_model_limitations(value: object) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        raise AnswerModelError("answer model output must be a JSON object")
    raw = value.get("limitations")
    if not isinstance(raw, list) or any(
        not isinstance(item, str) or not item.strip() for item in raw
    ):
        raise AnswerModelError("limitations must be a JSON string array")
    return tuple(item.strip() for item in raw)


def evidence_citation_map(packet: EvidencePacket) -> dict[str, AnswerCitation]:
    """Map only stable packet-position ids to authoritative citations."""
    return {
        f"E{index}": AnswerCitation.from_evidence(index, item)
        for index, item in enumerate(packet.evidence, start=1)
    }


def contains_citation_like_marker(text: str) -> bool:
    """Reject model-authored markers, including malformed lookalikes."""
    return bool(_CITATION_MARKER_RE.search(text))
