"""Bounded in-process answer sessions.

Only the planner receives prior turns. The answer generator receives an
immutable packet and never receives this memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from ..config import Config, validate_config
from ..retrieval.evidence import build_evidence
from .core import generate_answer
from .models import AnswerResult
from .persistence import store_answer


class AnswerSession:
    """Keep the latest complete user/assistant turns in memory only."""

    def __init__(
        self,
        config: Config,
        *,
        planner_chat_model: object | None = None,
        answer_chat_model: object | None = None,
        planner_model: object | None = None,
        answer_model: object | None = None,
    ) -> None:
        validate_config(config)
        self.config = config
        self._planner_chat_model = (
            planner_chat_model if planner_chat_model is not None else planner_model
        )
        self._answer_chat_model = (
            answer_chat_model if answer_chat_model is not None else answer_model
        )
        self._messages: list[dict[str, str]] = []

    @property
    def conversation_context(self) -> tuple[dict[str, str], ...]:
        return tuple(dict(message) for message in self._messages)

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        return self.conversation_context

    @property
    def history(self) -> tuple[dict[str, str], ...]:
        return self.conversation_context

    def clear(self) -> None:
        self._messages.clear()

    def ask(
        self,
        query: str,
        *,
        model: str | None = None,
        embedding_model: str | None = None,
        planner_chat_model: object | None = None,
        answer_chat_model: object | None = None,
        planner_model: object | None = None,
        answer_model: object | None = None,
    ) -> AnswerResult:
        """Build evidence, answer, store, then append one complete turn.

        A provider failure propagates before the user message is appended. A
        persisted safe refusal is considered a completed answer and is kept in
        memory so subsequent planner turns know what the user saw.
        """
        selected_model = model if model is not None else embedding_model
        evidence_result = build_evidence(
            self.config,
            query,
            conversation_context=self.conversation_context,
            model=selected_model,
            chat_model=(
                planner_chat_model
                if planner_chat_model is not None
                else planner_model
                if planner_model is not None
                else self._planner_chat_model
            ),
        )
        answer = generate_answer(
            evidence_result.packet,
            conversation_context=self.conversation_context,
            config=self.config,
            chat_model=(
                answer_chat_model
                if answer_chat_model is not None
                else answer_model
                if answer_model is not None
                else self._answer_chat_model
            ),
        )
        answer_id = store_answer(
            evidence_result.evidence_packet_id,
            answer,
            config=self.config,
        )
        completed = replace(
            answer,
            answer_id=answer_id,
            evidence_packet_id=evidence_result.evidence_packet_id,
            search_run_id=evidence_result.search_run_id,
        )
        self.record_complete_turn(query, completed.answer_text)
        return completed

    answer = ask

    def get_context(self) -> tuple[dict[str, str], ...]:
        return self.conversation_context

    def record_complete_turn(self, query: str, answer_text: str) -> None:
        """Append one already-persisted complete turn to planner-only memory."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonblank text")
        if not isinstance(answer_text, str) or not answer_text.strip():
            raise ValueError("answer_text must be nonblank text")
        self._messages.extend(
            (
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer_text},
            )
        )
        limit = self.config.answer_session_message_limit
        # Keep complete turns only. An odd message limit leaves one slot unused;
        # a limit of one retains no history because half a turn is invalid.
        max_turns = limit // 2
        self._messages[:] = self._messages[-2 * max_turns :] if max_turns else []
