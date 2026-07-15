"""Provider-neutral embedding validation, retry, and error helpers.

The helpers in this module deliberately know nothing about a provider SDK.
They are shared by the local and hosted adapters so malformed vectors and
transient request failures have one consistent policy.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Iterable
from numbers import Number
from typing import TypeVar

TOTAL_ATTEMPTS = 3


class EmbeddingValidationError(ValueError):
    """Raised when an embedding response violates the vector contract."""


class MalformedEmbeddingResponse(EmbeddingValidationError):
    """Raised when a provider response has no usable embedding payload."""


_T = TypeVar("_T")
_STATUS_ATTRIBUTES = ("status_code", "status", "http_status")
_STATUS_TEXT_PATTERN = re.compile(
    r"\b(?:http(?:\s+status)?|status(?:[_\s]+code)?)\s*[:=]?\s*([1-5]\d{2})\b",
    re.IGNORECASE,
)
_NETWORK_NAME_MARKERS = (
    "connection",
    "connecterror",
    "network",
    "readtimeout",
    "timeout",
    "transport",
    "temporarilyunavailable",
    "temporaryfailure",
)


def validate_vectors(
    vectors: Iterable[Iterable[object]] | object,
    *,
    expected_count: int | None = None,
    expected_dimension: int | None = None,
    context: str = "embedding response",
) -> list[list[float]]:
    """Validate and normalize a batch of provider vectors.

    Validation is positional: the returned list retains the exact order in
    which the provider returned the vectors. No response-index sorting is
    performed here or in the provider adapters.
    """
    if isinstance(vectors, (str, bytes, bytearray)) or vectors is None:
        raise MalformedEmbeddingResponse(f"{context} is not a vector sequence.")
    try:
        vector_rows = list(vectors)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise MalformedEmbeddingResponse(
            f"{context} is not a vector sequence."
        ) from exc

    if expected_count is not None and len(vector_rows) != expected_count:
        raise EmbeddingValidationError(
            f"{context} returned {len(vector_rows)} vector(s); "
            f"expected {expected_count}."
        )
    if expected_dimension is not None and expected_dimension <= 0:
        raise EmbeddingValidationError(
            f"{context} expected dimension must be greater than zero."
        )

    normalized: list[list[float]] = []
    for row_number, vector in enumerate(vector_rows):
        if isinstance(vector, (str, bytes, bytearray)) or vector is None:
            raise MalformedEmbeddingResponse(
                f"{context} vector {row_number} is malformed."
            )
        try:
            values = list(vector)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise MalformedEmbeddingResponse(
                f"{context} vector {row_number} is malformed."
            ) from exc
        if not values:
            raise EmbeddingValidationError(f"{context} vector {row_number} is empty.")
        if expected_dimension is not None and len(values) != expected_dimension:
            raise EmbeddingValidationError(
                f"{context} vector {row_number} has dimension {len(values)}; "
                f"expected {expected_dimension}."
            )

        normalized_row: list[float] = []
        for value_number, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, Number):
                raise EmbeddingValidationError(
                    f"{context} vector {row_number} contains a non-numeric "
                    f"value at position {value_number}."
                )
            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise EmbeddingValidationError(
                    f"{context} vector {row_number} contains an invalid value."
                ) from exc
            if not math.isfinite(numeric_value):
                raise EmbeddingValidationError(
                    f"{context} vector {row_number} contains a non-finite value."
                )
            normalized_row.append(numeric_value)
        normalized.append(normalized_row)
    return normalized


def validate_response_order(
    response_items: Iterable[object],
    *,
    expected_count: int,
    context: str = "embedding response",
) -> list[object]:
    """Validate response count/index metadata without reordering items.

    OpenAI-compatible responses commonly include an ``index`` field. When it
    is present, indexes must match the returned positions exactly. The
    provider's returned order is preserved exactly; an out-of-order response
    is rejected rather than silently attaching vectors to the wrong inputs.
    """
    items = list(response_items)
    if len(items) != expected_count:
        raise EmbeddingValidationError(
            f"{context} returned {len(items)} item(s); expected {expected_count}."
        )

    indexes: list[int] = []
    saw_index = False
    for item_number, item in enumerate(items):
        index = _response_field(item, "index")
        if index is None:
            continue
        saw_index = True
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise MalformedEmbeddingResponse(
                f"{context} item {item_number} has an invalid response index."
            )
        indexes.append(index)
    if saw_index and indexes != list(range(expected_count)):
        raise MalformedEmbeddingResponse(
            f"{context} response indexes are incomplete or out of order."
        )
    return items


def retry_transient(
    operation: Callable[[], _T],
    *,
    provider: str = "embedding provider",
    attempts: int = TOTAL_ATTEMPTS,
    sleep: Callable[[float], object] = time.sleep,
    backoff_seconds: float = 0.25,
) -> _T:
    """Run an operation with the shared transient retry policy.

    The default is exactly three total attempts. Only network failures,
    HTTP 408/429, and HTTP 5xx failures are retried. Validation, malformed
    responses, authentication/permission failures, model failures, and other
    HTTP 4xx failures are raised immediately.
    """
    if attempts <= 0:
        raise ValueError("attempts must be greater than zero")
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except EmbeddingValidationError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= attempts or not is_transient_error(exc):
                raise
            delay = max(0.0, backoff_seconds) * (2**attempt)
            if delay:
                sleep(delay)
    # The loop either returns or raises. This keeps type checkers aware that a
    # callable is always returned when the operation succeeds.
    raise RuntimeError(f"{provider} retry loop ended unexpectedly") from last_error


def is_transient_error(exc: BaseException) -> bool:
    """Return whether an exception belongs to the retryable failure classes."""
    # Native network exceptions take precedence over incidental numbers in
    # provider messages (for example, ``timed out after 120 seconds``).
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, OSError) and not isinstance(exc, FileNotFoundError):
        name = type(exc).__name__.casefold()
        if any(marker in name for marker in _NETWORK_NAME_MARKERS):
            return True

    status_code = http_status_code(exc)
    if status_code is not None:
        return status_code in {408, 429} or 500 <= status_code <= 599
    names = " ".join(
        (type(current).__name__ + " " + type(current).__module__).casefold()
        for current in _exception_chain(exc)
    )
    return any(marker in names for marker in _NETWORK_NAME_MARKERS)


def http_status_code(exc: BaseException) -> int | None:
    """Extract a status code without depending on a provider HTTP package."""
    for current in _exception_chain(exc):
        for attribute in _STATUS_ATTRIBUTES:
            value = getattr(current, attribute, None)
            if isinstance(value, int) and not isinstance(value, bool):
                if 100 <= value <= 599:
                    return value
        response = getattr(current, "response", None)
        if response is not None:
            value = getattr(response, "status_code", None)
            if isinstance(value, int) and not isinstance(value, bool):
                if 100 <= value <= 599:
                    return value
        # Some provider exceptions only put the HTTP status in their type or
        # message. The text is used for classification only and is never
        # included in a user-facing error.
        match = _STATUS_TEXT_PATTERN.search(str(current))
        if match:
            return int(match.group(1))
    return None


def sanitize_provider_error(
    exc: BaseException,
    provider: str = "embedding provider",
    *,
    operation: str = "request",
    model: str | None = None,
) -> str:
    """Create a safe provider diagnostic without echoing exception text."""
    status_code = http_status_code(exc)
    if status_code in {401, 403}:
        category = "authentication or permission failure"
    elif status_code == 404:
        category = "requested model or endpoint was not found"
    elif status_code == 408:
        category = "request timeout"
    elif status_code == 429:
        category = "rate limit"
    elif status_code is not None and 400 <= status_code <= 499:
        category = "client request failure"
    elif status_code is not None and 500 <= status_code <= 599:
        category = "provider service failure"
    elif is_transient_error(exc):
        category = "network failure"
    elif isinstance(exc, EmbeddingValidationError):
        category = "invalid embedding response"
    else:
        category = "provider failure"

    model_suffix = f" for model '{model}'" if model else ""
    return f"{provider} embedding {operation} failed{model_suffix}: {category}."


# Descriptive aliases used by provider modules and callers that prefer a
# verb-oriented name. They all point to the same dependency-free policy.
run_with_retries = retry_transient
with_transient_retries = retry_transient
validate_embedding_vectors = validate_vectors


def _exception_chain(exc: BaseException) -> Iterable[BaseException]:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _response_field(item: object, name: str) -> object:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)
