"""Bounded web-adjustable retrieval settings persisted under the data directory.

Only the explicitly allowlisted, non-sensitive tuning values below can be
changed through HTTP. Provider/model selection, credentials, storage paths,
log level, OCR, retry, prompt-budget, and timeout settings are never
web-settable; they remain environment configuration.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any

from ..config import Config
from ..indexing.profiles import EMBEDDING_PROFILES, resolve_embedding_profile

SETTINGS_FILENAME = "app_settings.json"


class SettingsError(ValueError):
    """Raised with a safe message when a submitted web setting is invalid."""


@dataclass(frozen=True)
class NumericBounds:
    minimum: float
    maximum: float


INT_SETTINGS: dict[str, NumericBounds] = {
    "keyword_top_k": NumericBounds(1, 200),
    "semantic_top_k": NumericBounds(1, 200),
    "metadata_top_k": NumericBounds(1, 200),
    "final_top_k": NumericBounds(1, 50),
    "rrf_k": NumericBounds(1, 1_000),
    "semantic_query_limit": NumericBounds(1, 10),
    "filename_fuzzy_threshold": NumericBounds(0, 100),
    "path_fuzzy_threshold": NumericBounds(0, 100),
    "evidence_max_tokens": NumericBounds(500, 100_000),
}
FLOAT_SETTINGS: dict[str, NumericBounds] = {
    "query_plan_min_confidence": NumericBounds(0.0, 1.0),
}
WEB_SETTING_NAMES: tuple[str, ...] = (
    "embedding_model",
    *INT_SETTINGS,
    *FLOAT_SETTINGS,
)


def _validated_value(name: str, value: object) -> object:
    if name == "embedding_model":
        if not isinstance(value, str) or not value.strip():
            raise SettingsError("embedding_model must be a supported profile name.")
        profile = resolve_embedding_profile(None, value, error=SettingsError)
        return profile.model_name
    if name in INT_SETTINGS:
        bounds = INT_SETTINGS[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise SettingsError(
                f"{name} must be an integer between "
                f"{int(bounds.minimum)} and {int(bounds.maximum)}."
            )
        if not bounds.minimum <= value <= bounds.maximum:
            raise SettingsError(
                f"{name} must be between {int(bounds.minimum)} "
                f"and {int(bounds.maximum)}."
            )
        return value
    if name in FLOAT_SETTINGS:
        bounds = FLOAT_SETTINGS[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(
                f"{name} must be a number between {bounds.minimum} "
                f"and {bounds.maximum}."
            )
        if not bounds.minimum <= float(value) <= bounds.maximum:
            raise SettingsError(
                f"{name} must be between {bounds.minimum} and {bounds.maximum}."
            )
        return float(value)
    raise SettingsError(f"{name} is not a web-adjustable setting.")


class WebSettingsStore:
    """Load, validate, persist, and apply web-adjustable settings overrides."""

    def __init__(self) -> None:
        self._lock = Lock()

    def settings_path(self, config: Config) -> Path:
        return config.data_dir / SETTINGS_FILENAME

    def apply(self, config: Config) -> Config:
        """Return the config with stored overrides layered on top."""
        overrides = self.load_overrides(config)
        return replace(config, **overrides) if overrides else config

    def load_overrides(self, config: Config) -> dict[str, Any]:
        with self._lock:
            return self._read(self.settings_path(config))

    def update(self, config: Config, changes: dict[str, Any]) -> dict[str, Any]:
        """Merge validated changes into the stored overrides.

        A ``None`` value clears that override so the environment default
        applies again. Values are validated before anything is written.
        """
        validated = {
            name: None if value is None else _validated_value(name, value)
            for name, value in changes.items()
            if name in WEB_SETTING_NAMES
        }
        with self._lock:
            path = self.settings_path(config)
            overrides = self._read(path)
            for name, value in validated.items():
                if value is None:
                    overrides.pop(name, None)
                else:
                    overrides[name] = value
            self._write(path, overrides)
        return overrides

    def _read(self, path: Path) -> dict[str, Any]:
        # A missing, unreadable, or hand-corrupted file must never break the
        # app; unknown names and invalid values are dropped rather than raised.
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        overrides: dict[str, Any] = {}
        for name, value in raw.items():
            if name not in WEB_SETTING_NAMES or value is None:
                continue
            try:
                overrides[name] = _validated_value(name, value)
            except SettingsError:
                continue
        return overrides

    def _write(self, path: Path, overrides: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + ".tmp")
        temp_path.write_text(
            json.dumps(overrides, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)


def describe_settings(
    base_config: Config, overrides: dict[str, Any]
) -> dict[str, object]:
    """Build the public settings payload from env defaults plus overrides."""
    defaults = {name: getattr(base_config, name) for name in WEB_SETTING_NAMES}
    limits = {
        name: {"min": bounds.minimum, "max": bounds.maximum}
        for name, bounds in {**INT_SETTINGS, **FLOAT_SETTINGS}.items()
    }
    return {
        "settings": {**defaults, **overrides},
        "defaults": defaults,
        "overrides": dict(overrides),
        "embedding_model_profiles": [
            {
                "model_name": profile.model_name,
                "provider": profile.provider,
                "dimension": profile.dimension,
                "requires_extra": profile.requires_extra,
            }
            for _, profile in sorted(EMBEDDING_PROFILES.items())
        ],
        "limits": limits,
    }
