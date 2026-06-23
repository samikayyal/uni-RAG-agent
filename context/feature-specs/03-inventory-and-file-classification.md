# Feature Spec 03: Inventory and File Classification

## Purpose

Inventory every file under `Courses`, classify it into an indexing category, preserve exact paths and course names, and store skip reasons for metadata-only files.

This spec creates the foundation for extraction, search coverage reporting, and metadata queries.

## Depends On

- [01-project-foundation.md](01-project-foundation.md)
- [02-configuration-and-storage.md](02-configuration-and-storage.md)
- `context/architecture.md` tables: `courses`, `files`, `extraction_runs`
- DEC-002, DEC-003, DEC-006, DEC-007, DEC-023

## In Scope

- Discover direct child folders under `Courses` as courses.
- Recursively inventory files under each course.
- Preserve exact course names and path spellings.
- Classify files by extension into extractable or metadata-only categories.
- Store file size, modified time, relative path, extension, category, index status, and reason.
- Use timestamp-first and hash-on-change behavior for changed files.
- Mark missing files without hard-deleting rows.
- Produce inventory run summaries.

## Out of Scope

- Extracting text or chunks.
- Opening unsafe artifacts.
- Decompressing archives.
- Standalone image OCR/captioning, transcription, or image analysis.
- Full hard-delete purge behavior beyond documenting the future command.

## Public Interfaces

Command:

```powershell
uv run -m uni_rag_agent inventory run
uv run -m uni_rag_agent inventory summary
```

Internal interfaces:

```python
inventory_courses(config: Config) -> InventoryRunResult
classify_file(path: Path) -> FileClassification
upsert_course(course: CourseRecord) -> int
upsert_file(file: FileRecord) -> int
mark_missing_files(seen_paths: set[str]) -> int
```

Classification categories:

```text
document
slides
notebook
code
data_schema
transcript
image_metadata_only
media_metadata_only
archive_metadata_only
binary_metadata_only
installer_metadata_only
model_metadata_only
unknown_metadata_only
```

Index statuses:

```text
pending
indexed
metadata_only
failed
skipped
```

## Storage and Schema Impact

Populate:

- `courses`
- `files`
- `extraction_runs` for inventory run accounting, or a shared run record if the implementation names the inventory run as an extraction run with no extraction step.

Required status behavior:

- Extractable categories start as `pending`.
- Metadata-only categories use `metadata_only`.
- Unsupported legacy `.doc` and `.ppt` are classified as `document` or `slides` but should be marked for later extraction failure with reason `legacy format not supported yet` when extraction runs.
- Removed files remain in `files`; update `last_seen_at` behavior must make them identifiable as missing without deleting chunks or embeddings.

## Workflow

1. Load config and validate storage.
2. List direct child directories of `Courses` as courses.
3. Walk files under each course with streaming traversal.
4. Classify by lowercased extension while preserving original filename/path.
5. Upsert courses and files.
6. For unchanged files, avoid rehashing when modified time and size are unchanged.
7. For changed files, compute SHA-256 to confirm content changes.
8. Mark files not seen in the current run as missing/soft-deleted according to the architecture.
9. Emit counts by course, category, extension, status, and skipped reason.

## Failure and Safety Rules

- Never mutate, rename, move, or delete files under `Courses`.
- Permission errors should mark individual files as failed or skipped with diagnostics and continue.
- Large file hashing should be streaming.
- Archives, installers, binaries, and model artifacts must not be opened beyond metadata and optional streaming hash.
- The run must tolerate unusual extensions and no-extension files.

## Tests

- Automated fixture test with two course folders and mixed file types.
- Verify exact course names and relative paths are preserved.
- Verify images, media, archives, installers, binaries, and model artifacts become metadata-only with reasons.
- Verify extractable files become pending.
- Verify rerunning inventory is idempotent.
- Verify a changed timestamp triggers hash comparison.
- Verify a missing file is marked without hard deletion.
- Optional smoke: run inventory on a tiny selected subtree copied into a temp fixture, not the full `Courses` archive.

## Acceptance Criteria

- `uv run -m uni_rag_agent inventory run` fills `courses` and `files`.
- Every file has exactly one category and one index status.
- Metadata-only files have useful `reason_not_indexed`.
- Inventory summaries can explain the mixed archive without extracting content.
- No automated test requires traversing the full `Courses` archive.
