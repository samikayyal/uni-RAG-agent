"""Inventory file classification vocabulary."""

from __future__ import annotations

from pathlib import Path

from .models import FileClassification

EXTRACTABLE_CATEGORIES = {
    "document",
    "slides",
    "notebook",
    "code",
    "data_schema",
    "transcript",
}

METADATA_ONLY_CATEGORIES = {
    "image_metadata_only",
    "media_metadata_only",
    "archive_metadata_only",
    "binary_metadata_only",
    "installer_metadata_only",
    "model_metadata_only",
    "unknown_metadata_only",
}

EXTENSION_CATEGORY_MAP = {
    ".pdf": "document",
    ".docx": "document",
    ".doc": "document",
    ".txt": "document",
    ".md": "document",
    ".pptx": "slides",
    ".ppt": "slides",
    ".ipynb": "notebook",
    ".py": "code",
    ".r": "code",
    ".cpp": "code",
    ".h": "code",
    ".m": "code",
    ".csv": "data_schema",
    ".xlsx": "data_schema",
    ".json": "data_schema",
    ".jsonl": "data_schema",
    ".sqlite": "data_schema",
    ".db": "data_schema",
    ".vtt": "transcript",
    ".png": "image_metadata_only",
    ".jpg": "image_metadata_only",
    ".jpeg": "image_metadata_only",
    ".tif": "image_metadata_only",
    ".jfif": "image_metadata_only",
    ".mp4": "media_metadata_only",
    ".mov": "media_metadata_only",
    ".mkv": "media_metadata_only",
    ".avi": "media_metadata_only",
    ".m4a": "media_metadata_only",
    ".wav": "media_metadata_only",
    ".zip": "archive_metadata_only",
    ".rar": "archive_metadata_only",
    ".7z": "archive_metadata_only",
    ".exe": "installer_metadata_only",
    ".msi": "installer_metadata_only",
    ".cab": "installer_metadata_only",
    ".bin": "model_metadata_only",
    ".joblib": "model_metadata_only",
    ".weights": "model_metadata_only",
    ".tflite": "model_metadata_only",
    ".pt": "model_metadata_only",
    ".pkl": "model_metadata_only",
    ".rdata": "model_metadata_only",
    ".rds": "model_metadata_only",
    ".dll": "binary_metadata_only",
    ".so": "binary_metadata_only",
    ".dylib": "binary_metadata_only",
    ".o": "binary_metadata_only",
    ".obj": "binary_metadata_only",
    ".class": "binary_metadata_only",
}

METADATA_ONLY_REASONS = {
    "image_metadata_only": "standalone image metadata-only by project decision",
    "media_metadata_only": "audio/video media metadata-only; transcription is opt-in later",
    "archive_metadata_only": "archive metadata-only; archives are not decompressed",
    "binary_metadata_only": "binary artifact metadata-only",
    "installer_metadata_only": "installer metadata-only; installers are never executed",
    "model_metadata_only": "model or serialized artifact metadata-only; unsafe or noisy for MVP indexing",
    "unknown_metadata_only": "unknown or unsupported extension metadata-only",
}


def classify_file(path: Path) -> FileClassification:
    """Classify a file by lowercased extension using the MVP vocabulary."""
    extension = path.suffix.lower()
    category = EXTENSION_CATEGORY_MAP.get(extension, "unknown_metadata_only")
    if category in EXTRACTABLE_CATEGORIES:
        return FileClassification(
            extension=extension,
            category=category,
            index_status="pending",
            reason_not_indexed=None,
        )
    return FileClassification(
        extension=extension,
        category=category,
        index_status="metadata_only",
        reason_not_indexed=METADATA_ONLY_REASONS[category],
    )
