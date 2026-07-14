"""Evidence-only answer generation, citation validation, and rendering."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..config import (
    ALLOWED_LLM_PROVIDERS,
    DEFAULT_ANSWER_PROMPT_MAX_TOKENS,
    Config,
    load_config,
)
from ..retrieval.evidence_models import EvidenceItem, EvidencePacket
from .models import (
    AnswerCitation,
    AnswerGenerationError,
    AnswerModelError,
    AnswerParagraph,
    AnswerResult,
    CitationValidationResult,
    contains_citation_like_marker,
    evidence_citation_map,
    parse_model_answer,
    parse_model_limitations,
)
from .providers import build_answer_chat_model

# Compatibility seam for deterministic test doubles and callers that mirror
# retrieval.planner's loader name. Planner and answer configuration remain
# separate; this alias only exposes the answer-specific constructor.
build_chat_model = build_answer_chat_model

_MARKER_RE = re.compile(r"\[E[1-9][0-9]*\]")
_MARKDOWN_PREFIX_RE = re.compile(
    r"^(?:>\s*|#{1,6}\s*|[-+*]\s+|\d+[.)]\s+)", re.IGNORECASE
)
_RENDERED_SECTION_RE = re.compile(r"^(?:references|limitations)\s*:", re.IGNORECASE)
_MARKDOWN_WRAPPERS = (
    ("**", "**"),
    ("__", "__"),
    ("~~", "~~"),
    ("`", "`"),
    ("*", "*"),
    ("_", "_"),
)
_RETRY_ERROR_TOKEN_RESERVE = 64
_MAX_RETRY_ERROR_CHARS = 256


def generate_answer(
    packet: EvidencePacket,
    conversation_context: Sequence[Mapping[str, str]] | None = None,
    *,
    config: Config | None = None,
    chat_model: object | None = None,
) -> AnswerResult:
    """Generate and render one answer from an immutable evidence packet.

    ``conversation_context`` remains in the public signature for compatibility
    with the Feature 10 spec, but it is deliberately ignored. Follow-up
    context is a planner-only concern owned by :class:`AnswerSession`; raw
    context is never placed in this prompt or persisted with the answer.
    """
    del conversation_context
    if not isinstance(packet, EvidencePacket):
        raise AnswerModelError("generate_answer requires an EvidencePacket")

    if not packet.evidence:
        return _insufficient_evidence_answer(packet)

    effective_config = config
    model = chat_model
    if effective_config is None and model is None:
        effective_config = load_config()
    prompt_budget = (
        effective_config.answer_prompt_max_tokens
        if effective_config is not None
        else DEFAULT_ANSWER_PROMPT_MAX_TOKENS
    )
    prompt_evidence_indexes = _select_prompt_evidence(packet, prompt_budget)
    if not prompt_evidence_indexes:
        return _prompt_budget_insufficient_answer(packet, prompt_budget)
    prompt_limitations = _prompt_budget_limitations(
        packet,
        prompt_evidence_indexes,
        prompt_budget,
    )

    retries = effective_config.answer_max_retries if effective_config else 1
    if model is None:
        if effective_config is None:  # defensive: the branch above loads it
            raise AnswerGenerationError("Answer configuration is unavailable")
        model = build_chat_model(effective_config)
    model_name = _answer_model_name(effective_config, model)
    prompt = _answer_prompt(packet, prompt_evidence_indexes)
    validation_errors: tuple[str, ...] = ()
    for attempt in range(retries + 1):
        attempt_prompt = (
            prompt
            if not validation_errors
            else _retry_prompt(packet, prompt_evidence_indexes, validation_errors)
        )
        try:
            response = model.invoke(attempt_prompt)  # type: ignore[attr-defined]
        except Exception as exc:  # provider failure is intentionally not persisted
            raise AnswerGenerationError(
                f"Answer LLM invocation failed: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            payload = _decode_response(response)
            paragraphs = parse_model_answer(payload)
            model_limitations = parse_model_limitations(payload)
            validation = _validate_paragraphs(
                paragraphs,
                packet,
                evidence_indexes=prompt_evidence_indexes,
            )
            if not validation.valid:
                raise AnswerModelError("; ".join(validation.errors))
            paragraphs = validation.paragraphs
            citations = validation.citations
            limitations = _dedupe(
                (*model_limitations, *packet.weaknesses, *prompt_limitations)
            )
            answer_text = _render_answer(paragraphs, citations, limitations)
            return AnswerResult(
                answer_text=answer_text,
                citations=citations,
                limitations=limitations,
                model_name=model_name,
                paragraphs=paragraphs,
            )
        except (AnswerModelError, TypeError, ValueError, json.JSONDecodeError) as exc:
            validation_errors = (str(exc) or "invalid answer model output",)
            if attempt < retries:
                continue
            return _safe_validation_refusal(
                packet,
                model_name=model_name,
                attempts=retries + 1,
                errors=validation_errors,
                prompt_limitations=prompt_limitations,
            )
    # The loop always returns or raises; keep a defensive branch for type
    # checkers and unusual custom iterables.
    raise AnswerGenerationError("Answer generation did not produce a result")


def validate_answer_citations(
    answer: AnswerResult | Mapping[str, object],
    packet: EvidencePacket,
    *,
    allowed_evidence_indexes: Sequence[int] | None = None,
) -> CitationValidationResult:
    """Validate that all paragraph citations resolve to packet evidence.

    Validation accepts stable ``E<evidence_index>`` ids and unambiguous
    ``chunk:<chunk_id>`` aliases. Structured output is always canonical.
    """
    if not isinstance(packet, EvidencePacket):
        return CitationValidationResult(False, ("packet must be an EvidencePacket",))
    if isinstance(answer, Mapping):
        try:
            paragraphs = parse_model_answer(answer)
        except Exception as exc:  # noqa: BLE001 - diagnostics are returned
            return CitationValidationResult(False, (str(exc),))
    elif isinstance(answer, AnswerResult):
        paragraphs = answer.paragraphs
        if not paragraphs:
            paragraphs = _paragraphs_from_rendered(answer.answer_text)
    else:
        return CitationValidationResult(False, ("answer must be an AnswerResult",))
    return _validate_paragraphs(
        paragraphs,
        packet,
        evidence_indexes=allowed_evidence_indexes,
    )


def format_citation(value: EvidenceItem | AnswerCitation) -> str:
    """Render one deterministic references-list citation (without markers)."""
    if isinstance(value, AnswerCitation):
        return f"{value.course} - {value.file_path} - {value.location_label}"
    if isinstance(value, EvidenceItem):
        return f"{value.course} - {value.file} - {value.location.label}"
    raise TypeError("format_citation requires EvidenceItem or AnswerCitation")


def _contains_rendered_section(text: str) -> bool:
    return any(
        _RENDERED_SECTION_RE.match(_strip_markdown_decoration(line))
        for line in text.splitlines()
    )


def _strip_markdown_decoration(line: str) -> str:
    value = line.strip()
    while value:
        undecorated = _MARKDOWN_PREFIX_RE.sub("", value, count=1).strip()
        if undecorated != value:
            value = undecorated
            continue
        unwrapped = value
        for opening, closing in _MARKDOWN_WRAPPERS:
            if (
                value.startswith(opening)
                and value.endswith(closing)
                and len(value) > len(opening) + len(closing)
            ):
                unwrapped = value[len(opening) : -len(closing)].strip()
                break
        if unwrapped == value:
            break
        value = unwrapped
    return value


def _decode_response(response: object) -> Mapping[str, object]:
    content: Any = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, Mapping) else str(item)
            for item in content
        )
    if not isinstance(content, str):
        raise AnswerModelError("answer model output must be a JSON object")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AnswerModelError("answer model output must be one JSON object") from exc
    if not isinstance(payload, Mapping):
        raise AnswerModelError("answer model output must be one JSON object")
    return payload


def _validate_paragraphs(
    paragraphs: Sequence[AnswerParagraph],
    packet: EvidencePacket,
    *,
    evidence_indexes: Sequence[int] | None = None,
) -> CitationValidationResult:
    if not paragraphs:
        return CitationValidationResult(
            False, ("answer_paragraphs must contain at least one paragraph",)
        )
    aliases = evidence_citation_map(packet, evidence_indexes)
    errors: list[str] = []
    resolved: list[AnswerCitation] = []
    canonical_paragraphs: list[AnswerParagraph] = []
    for index, paragraph in enumerate(paragraphs, start=1):
        if not paragraph.text.strip():
            errors.append(f"paragraph {index} is blank")
        if contains_citation_like_marker(paragraph.text):
            errors.append(f"paragraph {index} contains citation markers")
        if _contains_rendered_section(paragraph.text):
            errors.append(f"paragraph {index} contains a rendered section")
        if not paragraph.citation_ids and packet.evidence:
            errors.append(f"paragraph {index} must cite at least one evidence item")
        canonical_ids: list[str] = []
        for citation_id in paragraph.citation_ids:
            citation = aliases.get(citation_id)
            if citation is None:
                errors.append(
                    f"paragraph {index} contains unknown citation id: {citation_id}"
                )
            else:
                if citation.citation_id not in canonical_ids:
                    canonical_ids.append(citation.citation_id)
                if citation not in resolved:
                    resolved.append(citation)
        canonical_paragraphs.append(
            AnswerParagraph(paragraph.text, tuple(canonical_ids))
        )
    return CitationValidationResult(
        valid=not errors,
        errors=tuple(_dedupe(errors)),
        citations=tuple(resolved),
        paragraphs=tuple(canonical_paragraphs),
    )


def _select_prompt_evidence(
    packet: EvidencePacket,
    prompt_budget: int,
) -> tuple[int, ...]:
    """Select complete evidence in packet rank order within the full prompt budget."""
    selected: list[int] = []
    ranked_indexes = sorted(
        range(1, len(packet.evidence) + 1),
        key=lambda index: (packet.evidence[index - 1].rank, index),
    )
    for index in ranked_indexes:
        candidate = (*selected, index)
        prompt = _answer_prompt(packet, candidate)
        if (
            _estimate_prompt_tokens(prompt) + _RETRY_ERROR_TOKEN_RESERVE
            <= prompt_budget
        ):
            selected.append(index)
    return tuple(selected)


def _estimate_prompt_tokens(prompt: str) -> int:
    """Use the same deterministic whitespace estimate as evidence selection."""
    return len(prompt.split())


def _prompt_budget_limitations(
    packet: EvidencePacket,
    evidence_indexes: Sequence[int],
    prompt_budget: int,
) -> tuple[str, ...]:
    omitted = len(packet.evidence) - len(evidence_indexes)
    if omitted <= 0:
        return ()
    return (
        f"{omitted} evidence item(s) were omitted from the answer prompt to fit "
        f"the {prompt_budget}-token budget.",
    )


def _bounded_validation_errors(errors: Sequence[str]) -> tuple[str, ...]:
    if not errors:
        return ()
    normalized = " ".join(str(errors[0]).split())[:_MAX_RETRY_ERROR_CHARS]
    return (" ".join(normalized.split()[:48]) or "invalid answer model output",)


def _answer_prompt(
    packet: EvidencePacket,
    evidence_indexes: Sequence[int],
    *,
    validation_errors: Sequence[str] = (),
) -> str:
    evidence = []
    for index in evidence_indexes:
        item = packet.evidence[index - 1]
        evidence.append(
            {
                "citation_id": f"E{index}",
                "chunk_alias": f"chunk:{item.chunk_id}",
                "evidence_index": index,
                "course": item.course,
                "file_id": item.file_id,
                "chunk_id": item.chunk_id,
                "file_path": item.file,
                "source_type": item.source_type,
                "location": item.location.as_safe_dict(),
                "text": item.text,
            }
        )
    return json.dumps(
        {
            "task": (
                "Retry and return only one corrected strict JSON object."
                if validation_errors
                else "Return only one strict JSON object; answer only from the supplied evidence."
            ),
            "schema": {
                "answer_paragraphs": [
                    {
                        "text": "nonblank prose without citation markers",
                        "citation_ids": ["E1"],
                    }
                ],
                "limitations": ["optional nonblank limitation"],
            },
            "query": packet.query,
            "evidence": evidence,
            "validation_errors": list(_bounded_validation_errors(validation_errors)),
            "packet_weaknesses": list(packet.weaknesses),
            "answer_constraints": list(packet.answer_constraints),
            "rules": [
                "Use only supplied evidence; do not use memory.",
                "Every paragraph must cite one or more allowed citation ids.",
                "Citation ids may use canonical E<n> or the supplied chunk:<id> alias.",
                "Do not render References, Limitations, paths, Markdown links, or citation markers.",
                "Return exactly the schema fields and no markdown fences.",
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _retry_prompt(
    packet: EvidencePacket,
    evidence_indexes: Sequence[int],
    errors: Sequence[str],
) -> str:
    # Include the same evidence and constraints on every retry. The only new
    # content is bounded validation diagnostics; no conversation context or
    # prior invalid model output is echoed.
    return _answer_prompt(
        packet,
        evidence_indexes,
        validation_errors=errors,
    )


def _render_answer(
    paragraphs: Sequence[AnswerParagraph],
    citations: Sequence[AnswerCitation],
    limitations: Sequence[str],
) -> str:
    # Marker order follows first appearance while references remain globally
    # unique and stable.
    citation_by_id = {citation.citation_id: citation for citation in citations}
    lines: list[str] = []
    ordered_refs: list[AnswerCitation] = []
    for paragraph in paragraphs:
        markers: list[str] = []
        for citation_id in paragraph.citation_ids:
            citation = citation_by_id.get(citation_id)
            if citation is None:
                continue
            marker = f"[{citation.citation_id}]"
            if marker not in markers:
                markers.append(marker)
            if citation not in ordered_refs:
                ordered_refs.append(citation)
        lines.append(f"{paragraph.text.strip()} {' '.join(markers)}".strip())
    if ordered_refs:
        lines.append("")
        lines.append("References:")
        lines.extend(
            f"- [{citation.citation_id}] {format_citation(citation)}"
            for citation in ordered_refs
        )
    if limitations:
        lines.append("")
        lines.append("Limitations:")
        lines.extend(f"- {limitation}" for limitation in limitations)
    return "\n".join(lines)


def _insufficient_evidence_answer(packet: EvidencePacket) -> AnswerResult:
    searched_courses = ", ".join(packet.searched["courses"]) or "none"
    searched_indexes = ", ".join(packet.searched["indexes"]) or "none"
    missing: list[str] = []
    if packet.coverage.courses_without_chunk_hits:
        missing.append(
            "courses without chunk hits: "
            + ", ".join(packet.coverage.courses_without_chunk_hits)
        )
    if packet.coverage.indexes_without_chunk_hits:
        missing.append(
            "indexes without chunk hits: "
            + ", ".join(packet.coverage.indexes_without_chunk_hits)
        )
    if packet.coverage.semantic_queries_without_hits:
        missing.append(
            "semantic queries without hits: "
            + ", ".join(packet.coverage.semantic_queries_without_hits)
        )
    if packet.coverage.missing_capabilities:
        missing.append(
            "missing capabilities: " + ", ".join(packet.coverage.missing_capabilities)
        )
    detail = "; ".join(missing) or "no authoritative chunks were selected"
    paragraph = (
        "Insufficient evidence to answer safely from the indexed course materials. "
        f"Searched courses: {searched_courses}. Searched indexes: {searched_indexes}. "
        f"{detail}."
    )
    limitations = _dedupe(
        (
            *packet.weaknesses,
            "No answer model was invoked because the evidence packet is empty.",
        )
    )
    return AnswerResult(
        answer_text=_render_plain_answer(paragraph, limitations),
        citations=(),
        limitations=limitations,
        model_name=None,
        paragraphs=(AnswerParagraph(text=paragraph, citation_ids=()),),
    )


def _prompt_budget_insufficient_answer(
    packet: EvidencePacket,
    prompt_budget: int,
) -> AnswerResult:
    paragraph = (
        "Insufficient evidence to answer safely because no complete evidence item "
        "fit within the configured answer prompt budget."
    )
    limitations = _dedupe(
        (
            *packet.weaknesses,
            "No answer model was invoked because no complete evidence item fit "
            f"within the {prompt_budget}-token answer prompt budget.",
        )
    )
    return AnswerResult(
        answer_text=_render_plain_answer(paragraph, limitations),
        citations=(),
        limitations=limitations,
        model_name=None,
        paragraphs=(AnswerParagraph(text=paragraph, citation_ids=()),),
    )


def _safe_validation_refusal(
    packet: EvidencePacket,
    *,
    model_name: str | None,
    attempts: int,
    errors: Sequence[str],
    prompt_limitations: Sequence[str] = (),
) -> AnswerResult:
    paragraph = (
        "I cannot provide a source-grounded answer because the answer model "
        "did not return a valid cited response."
    )
    limitations = _dedupe(
        (
            *packet.weaknesses,
            *prompt_limitations,
            f"Answer validation failed after {attempts} attempt(s); no model answer was accepted.",
        )
    )
    del errors  # Diagnostics stay in telemetry, never in persisted answer text.
    return AnswerResult(
        answer_text=_render_plain_answer(paragraph, limitations),
        citations=(),
        limitations=limitations,
        model_name=model_name,
        paragraphs=(AnswerParagraph(text=paragraph, citation_ids=()),),
    )


def _render_plain_answer(paragraph: str, limitations: Sequence[str]) -> str:
    lines = [paragraph]
    if limitations:
        lines.extend(("", "Limitations:", *(f"- {value}" for value in limitations)))
    return "\n".join(lines)


def _paragraphs_from_rendered(answer_text: str) -> tuple[AnswerParagraph, ...]:
    body = answer_text.split("\n\nReferences:", 1)[0]
    body = body.split("\n\nLimitations:", 1)[0]
    paragraphs: list[AnswerParagraph] = []
    for line in body.splitlines():
        markers = tuple(marker[1:-1] for marker in _MARKER_RE.findall(line))
        text = _MARKER_RE.sub("", line).strip()
        if text:
            paragraphs.append(AnswerParagraph(text=text, citation_ids=markers))
    return tuple(paragraphs)


def _answer_model_name(config: Config | None, model: object) -> str | None:
    del model
    if config is not None and config.answer_llm_provider and config.answer_llm_model:
        return f"{config.answer_llm_provider}:{config.answer_llm_model}"
    return None


def validate_answer_for_storage(
    answer: AnswerResult,
    packet: EvidencePacket,
    config: Config,
) -> None:
    """Enforce the immutable packet/answer invariant before append-only storage."""
    if not packet.evidence:
        expected = _insufficient_evidence_answer(packet)
        _require_same_stored_answer(answer, expected, "insufficient-evidence")
        return

    prompt_evidence_indexes = _select_prompt_evidence(
        packet,
        config.answer_prompt_max_tokens,
    )
    if not prompt_evidence_indexes:
        expected = _prompt_budget_insufficient_answer(
            packet,
            config.answer_prompt_max_tokens,
        )
        _require_same_stored_answer(answer, expected, "prompt-budget-insufficient")
        return
    prompt_limitations = _prompt_budget_limitations(
        packet,
        prompt_evidence_indexes,
        config.answer_prompt_max_tokens,
    )

    provider = config.answer_llm_provider
    model = config.answer_llm_model
    if provider not in ALLOWED_LLM_PROVIDERS or not model:
        raise AnswerModelError(
            "persisting an evidence-backed answer requires configured answer provider and model"
        )
    expected_model_name = f"{provider}:{model}"
    if answer.model_name != expected_model_name:
        raise AnswerModelError("answer model_name must match configured provider:model")

    if not answer.citations:
        expected = _safe_validation_refusal(
            packet,
            model_name=expected_model_name,
            attempts=config.answer_max_retries + 1,
            errors=(),
            prompt_limitations=prompt_limitations,
        )
        _require_same_stored_answer(answer, expected, "validation-refusal")
        return

    validation = validate_answer_citations(
        answer,
        packet,
        allowed_evidence_indexes=prompt_evidence_indexes,
    )
    if not validation.valid:
        raise AnswerModelError(
            "answer citations do not validate against the evidence packet: "
            + "; ".join(validation.errors)
        )
    if tuple(answer.citations) != tuple(validation.citations):
        raise AnswerModelError(
            "structured citations do not match packet-authoritative evidence"
        )
    missing_weaknesses = [
        weakness
        for weakness in (*packet.weaknesses, *prompt_limitations)
        if weakness not in answer.limitations
    ]
    if missing_weaknesses:
        raise AnswerModelError("answer limitations omit packet weaknesses")
    expected_text = _render_answer(
        validation.paragraphs,
        validation.citations,
        answer.limitations,
    )
    if answer.answer_text != expected_text:
        raise AnswerModelError(
            "answer text, markers, references, or limitations do not match structured fields"
        )


def _require_same_stored_answer(
    actual: AnswerResult,
    expected: AnswerResult,
    outcome: str,
) -> None:
    if (
        actual.answer_text != expected.answer_text
        or tuple(actual.citations) != tuple(expected.citations)
        or tuple(actual.limitations) != tuple(expected.limitations)
        or actual.model_name != expected.model_name
    ):
        raise AnswerModelError(
            f"{outcome} answer does not match the deterministic packet-derived outcome"
        )


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        normalized = " ".join(str(value).split())
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)
