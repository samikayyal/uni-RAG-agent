"""Text extraction constants and vocabulary."""

from __future__ import annotations

import re

TEXT_EXTRACTION_CATEGORIES = {
    "document",
    "slides",
    "notebook",
    "code",
    "transcript",
}

SUPPORTED_TEXT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".pptx",
    ".ipynb",
    ".py",
    ".r",
    ".cpp",
    ".h",
    ".m",
    ".vtt",
}

LEGACY_EXTENSIONS = {".doc", ".ppt"}
LEGACY_FORMAT_REASON = "legacy format not supported yet"
SCANNED_PDF_OCR_REASON = "scanned PDF, OCR not available"
NO_TEXT_REASON = "no extractable text found"

DEFAULT_MAX_CHUNK_TOKENS = 1000
NOTEBOOK_OUTPUT_CHAR_LIMIT = 500
ERROR_CHAR_LIMIT = 4000
PDF_SCANNED_TEXT_CHAR_THRESHOLD = 20

TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
VTT_TIMESTAMP_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}(?::\d{2})?(?:\.\d{3})?)\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}(?::\d{2})?(?:\.\d{3})?)"
)
R_FUNCTION_RE = re.compile(r"^\s*([A-Za-z.][\w.]*)\s*(?:<-|=)\s*function\s*\(", re.M)
CPP_FUNCTION_RE = re.compile(
    r"^\s*(?:[\w:<>,~*&]+\s+)+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?\{",
    re.M,
)
MATLAB_FUNCTION_RE = re.compile(
    r"^\s*function\s+(?:\[[^\]]+\]\s*=\s*|[A-Za-z_]\w*\s*=\s*)?"
    r"(?P<name>[A-Za-z_]\w*)",
    re.M,
)
