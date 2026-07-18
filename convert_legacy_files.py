"""Convert legacy Office files (.ppt/.doc) to modern formats in place.

Standalone script — no project imports, stdlib only. For every legacy file
found under the courses root, a converted sibling is written next to it
(``Lecture 3.ppt`` -> ``Lecture 3.pptx``). Originals are never modified or
deleted. Conversion uses LibreOffice headless, so LibreOffice must be
installed (``winget install TheDocumentFoundation.LibreOffice``).

Usage:
    uv run convert_legacy_files.py [--courses-root Courses] [--dry-run] [--force]

Exit code 0 when nothing failed, 1 when at least one conversion failed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CONVERSIONS = {".ppt": "pptx", ".doc": "docx"}
PER_FILE_TIMEOUT_SECONDS = 180

SOFFICE_CANDIDATES = (
    Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
    Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
)


def find_soffice() -> str | None:
    resolved = shutil.which("soffice")
    if resolved:
        return resolved
    for candidate in SOFFICE_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def discover_legacy_files(courses_root: Path) -> list[Path]:
    found: list[Path] = []
    for path in sorted(courses_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in CONVERSIONS:
            continue
        # Office lock/temp files and macOS resource forks are not real content.
        if path.name.startswith(("~$", "._")):
            continue
        if "__MACOSX" in path.parts:
            continue
        found.append(path)
    return found


def convert_one(soffice: str, source: Path, target: Path) -> tuple[bool, str]:
    """Convert into a temp dir first so an interrupted run leaves no partial file."""
    target_format = CONVERSIONS[source.suffix.lower()]
    with tempfile.TemporaryDirectory(prefix="legacy_convert_") as temp_dir:
        try:
            completed = subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--norestore",
                    "--convert-to",
                    target_format,
                    "--outdir",
                    temp_dir,
                    str(source),
                ],
                capture_output=True,
                text=True,
                timeout=PER_FILE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return False, f"timed out after {PER_FILE_TIMEOUT_SECONDS}s"

        converted = Path(temp_dir) / f"{source.stem}.{target_format}"
        if completed.returncode != 0 or not converted.exists():
            detail = (completed.stderr or completed.stdout or "no output").strip()
            return False, detail[:500]

        shutil.move(str(converted), str(target))
        return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--courses-root",
        type=Path,
        default=Path("Courses"),
        help="Root directory to scan recursively (default: ./Courses).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be converted without converting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even when the modern sibling already exists.",
    )
    args = parser.parse_args()

    courses_root = args.courses_root.resolve()
    if not courses_root.is_dir():
        print(f"ERROR: courses root not found: {courses_root}")
        return 1

    legacy_files = discover_legacy_files(courses_root)
    if not legacy_files:
        print(f"No legacy .ppt/.doc files found under {courses_root}.")
        return 0

    soffice = None
    if not args.dry_run:
        soffice = find_soffice()
        if soffice is None:
            print(
                "ERROR: LibreOffice (soffice) not found on PATH or in default "
                "install locations.\nInstall it first: "
                "winget install TheDocumentFoundation.LibreOffice"
            )
            return 1
        print(f"Using LibreOffice: {soffice}")

    converted = skipped = failed = 0
    for source in legacy_files:
        target = source.with_suffix("." + CONVERSIONS[source.suffix.lower()])
        relative = source.relative_to(courses_root)

        if target.exists() and not args.force:
            print(f"SKIP  (exists) {relative}")
            skipped += 1
            continue
        if args.dry_run:
            print(f"WOULD CONVERT  {relative} -> {target.name}")
            converted += 1
            continue

        ok, error = convert_one(soffice, source, target)
        if ok:
            print(f"OK    {relative} -> {target.name}")
            converted += 1
        else:
            print(f"FAIL  {relative}: {error}")
            failed += 1

    verb = "would convert" if args.dry_run else "converted"
    print(
        f"\nDone: {converted} {verb}, {skipped} skipped (already converted), "
        f"{failed} failed, {len(legacy_files)} legacy files total."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
