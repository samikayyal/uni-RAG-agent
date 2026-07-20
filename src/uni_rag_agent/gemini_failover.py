"""Sticky Gemini API-key failover for resource-exhausted (quota) errors."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from threading import RLock

from .config import Config

_LOGGER = logging.getLogger(__name__)
_RESOURCE_EXHAUSTED_MARKERS = (
    "resourceexhausted",
    "resource_exhausted",
    "resource exhausted",
    "quota",
    "429",
)


def build_gemini_with_failover(
    build_model: Callable[[str | None], object],
    config: Config,
) -> object:
    """Build a Gemini chat model that fails over across configured API keys.

    With one key (or none, deferring to provider environment defaults) this
    returns the plain model unchanged; the wrapper only exists when a second
    key gives it somewhere to fail over to.
    """
    keys = tuple(key for key in (config.google_api_key, config.google_api_key_2) if key)
    if len(keys) < 2:
        return build_model(config.google_api_key)
    return GeminiKeyFailoverChatModel(build_model, keys)


class GeminiKeyFailoverChatModel:
    """Invoke-compatible wrapper that rotates API keys on quota exhaustion.

    Rotation wraps around and is sticky: when the active key reports resource
    exhaustion the wrapper moves to the next key (cycling back to the first)
    and later invocations keep using it, so keys whose quota has refreshed get
    another chance. One invocation gives each key at most one attempt; only
    when every key fails within the same call does the final quota error
    propagate. Non-quota errors propagate unchanged so existing
    planner/answer error boundaries keep their behavior.
    """

    def __init__(
        self,
        build_model: Callable[[str | None], object],
        api_keys: Sequence[str],
    ) -> None:
        if not api_keys:
            raise ValueError("GeminiKeyFailoverChatModel requires at least one key")
        self._build_model = build_model
        self._api_keys = tuple(api_keys)
        self._lock = RLock()
        self._index = 0
        self._model: object | None = None

    @property
    def active_key_index(self) -> int:
        with self._lock:
            return self._index

    def invoke(self, prompt: object) -> object:
        attempts = 0
        while True:
            model, index = self._current_model()
            try:
                return model.invoke(prompt)  # type: ignore[attr-defined]
            except Exception as exc:
                attempts += 1
                if not _is_resource_exhausted(exc) or attempts >= len(self._api_keys):
                    raise
                self._rotate(index)

    def _current_model(self) -> tuple[object, int]:
        with self._lock:
            if self._model is None:
                self._model = self._build_model(self._api_keys[self._index])
            return self._model, self._index

    def _rotate(self, failed_index: int) -> None:
        """Move past ``failed_index``, wrapping around to the first key."""
        with self._lock:
            if failed_index != self._index:
                return  # another thread already rotated away from this key
            self._index = (self._index + 1) % len(self._api_keys)
            self._model = None
            _LOGGER.warning(
                "Gemini API key %d reported resource exhaustion; "
                "rotating to key %d of %d.",
                failed_index + 1,
                self._index + 1,
                len(self._api_keys),
            )


def _is_resource_exhausted(exc: BaseException) -> bool:
    """Match quota/rate-limit failures across provider and LangChain wrappers."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__} {current}".lower()
        if any(marker in text for marker in _RESOURCE_EXHAUSTED_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False
