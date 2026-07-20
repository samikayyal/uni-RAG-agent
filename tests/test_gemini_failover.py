"""Sticky Gemini API-key failover on resource-exhausted provider errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from uni_rag_agent.config import load_config
from uni_rag_agent.gemini_failover import (
    GeminiKeyFailoverChatModel,
    build_gemini_with_failover,
)

from .support import make_config

QUOTA_MESSAGE = "429 Resource has been exhausted (e.g. check quota): RESOURCE_EXHAUSTED"


class RecordingModel:
    def __init__(self, api_key: str | None, error: Exception | None = None) -> None:
        self.api_key = api_key
        self.error = error
        self.invocations = 0

    def invoke(self, prompt: object) -> str:
        self.invocations += 1
        if self.error is not None:
            raise self.error
        return f"answer from {self.api_key}"


def _builder(errors_by_key: dict[str, Exception]) -> tuple[callable, list]:
    built: list[RecordingModel] = []

    def build(api_key: str | None) -> RecordingModel:
        model = RecordingModel(api_key, errors_by_key.get(api_key))
        built.append(model)
        return model

    return build, built


def test_failover_switches_key_on_resource_exhausted_and_stays_switched() -> None:
    build, built = _builder({"key-1": RuntimeError(QUOTA_MESSAGE)})
    model = GeminiKeyFailoverChatModel(build, ("key-1", "key-2"))

    assert model.invoke("first prompt") == "answer from key-2"
    assert model.invoke("second prompt") == "answer from key-2"

    assert [entry.api_key for entry in built] == ["key-1", "key-2"]
    assert built[0].invocations == 1  # exhausted key is not re-probed
    assert built[1].invocations == 2
    assert model.active_key_index == 1


def test_failover_matches_wrapped_resource_exhausted_causes() -> None:
    wrapped = RuntimeError("planner call failed")
    wrapped.__cause__ = ValueError(QUOTA_MESSAGE)
    build, _ = _builder({"key-1": wrapped})
    model = GeminiKeyFailoverChatModel(build, ("key-1", "key-2"))

    assert model.invoke("prompt") == "answer from key-2"


def test_non_quota_errors_propagate_without_switching() -> None:
    build, built = _builder({"key-1": RuntimeError("invalid request payload")})
    model = GeminiKeyFailoverChatModel(build, ("key-1", "key-2"))

    try:
        model.invoke("prompt")
    except RuntimeError as exc:
        assert "invalid request payload" in str(exc)
    else:
        raise AssertionError("expected the provider error to propagate")
    assert model.active_key_index == 0
    assert len(built) == 1


def test_rotation_wraps_back_to_first_key_after_quota_refresh() -> None:
    errors = {"key-1": RuntimeError(QUOTA_MESSAGE)}
    build, built = _builder(errors)
    model = GeminiKeyFailoverChatModel(build, ("key-1", "key-2"))

    assert model.invoke("prompt") == "answer from key-2"

    # Later, key-2's quota exhausts while key-1's has refreshed.
    built[-1].error = RuntimeError(QUOTA_MESSAGE)
    del errors["key-1"]

    assert model.invoke("prompt") == "answer from key-1"
    assert model.active_key_index == 0


def test_all_keys_exhausted_in_one_call_reraises() -> None:
    build, _ = _builder(
        {
            "key-1": RuntimeError(QUOTA_MESSAGE),
            "key-2": RuntimeError(QUOTA_MESSAGE),
        }
    )
    model = GeminiKeyFailoverChatModel(build, ("key-1", "key-2"))

    try:
        model.invoke("prompt")
    except RuntimeError as exc:
        assert "RESOURCE_EXHAUSTED" in str(exc)
    else:
        raise AssertionError("expected the final quota error to propagate")
    assert model.active_key_index == 1


def test_single_key_config_returns_unwrapped_model(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        google_api_key="key-1",
        google_api_key_2=None,
    )
    build, built = _builder({})

    model = build_gemini_with_failover(build, config)

    assert isinstance(model, RecordingModel)
    assert built[0].api_key == "key-1"


def test_two_key_config_returns_failover_wrapper(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        google_api_key="key-1",
        google_api_key_2="key-2",
    )
    build, _ = _builder({})

    model = build_gemini_with_failover(build, config)

    assert isinstance(model, GeminiKeyFailoverChatModel)
    assert model.invoke("prompt") == "answer from key-1"


def test_load_config_reads_second_google_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY_2", raising=False)
    (tmp_path / "Courses").mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GOOGLE_API_KEY=primary\nGOOGLE_API_KEY_2=secondary\n",
        encoding="utf-8",
    )

    config = load_config(repo_root=tmp_path, env_file=env_file)

    assert config.google_api_key == "primary"
    assert config.google_api_key_2 == "secondary"
    # Keys never appear in the loggable snapshot or the dataclass repr.
    assert "google_api_key_2" not in config.as_safe_dict()
    assert "secondary" not in repr(config)
