# Feature Spec 03: Inventory and File Classification

## Purpose

Inventory every eligible file under `Courses`, classify it into an indexing category, preserve exact paths and course names, and store skip reasons for metadata-only files. Jupyter checkpoint trees are excluded before classification and do not receive metadata rows.

This spec creates the foundation for extraction, search coverage reporting, and metadata queries.

## Depends On

- [01-project-foundation.md](01-project-foundation.md)
- [02-configuration-and-storage.md](02-configuration-and-storage.md)
- `context/architecture.md` tables: `courses`, `files`, `extraction_runs`
- DEC-002, DEC-003, DEC-006, DEC-007, DEC-023

## In Scope

- Discover direct child folders under `Courses` as courses.
- Recursively inventory files under each course.
- Exclude `.ipynb_checkpoints` directories and descendants before creating file
  metadata rows.
- Preserve exact course names and path spellings.
- Classify files by extension into extractable or metadata-only categories.
- Store file size, modified time, relative path, extension, category, index status, and reason.
- Use timestamp-first and hash-on-change behavior for changed files.
- Mark missing files without hard-deleting rows.
- Produce inventory run summaries.
- Provide a pandas-based, read-only EDA notebook for exploring the SQLite inventory output by course, category, extension, status, size, skip reason, and freshness.

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

Notebook:

```text
notebooks/inventory_eda.ipynb
```

Update this notebook whenever inventory changes the `courses`, `files`, or inventory `extraction_runs` fields, status vocabulary, metadata-only reasons, summary interpretation, or `inventory run` command behavior.

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
10. Use `notebooks/inventory_eda.ipynb` after an inventory run for read-only exploratory analysis of `data/uni_rag.sqlite`.

## Failure and Safety Rules

- Never mutate, rename, move, or delete files under `Courses`.
- Jupyter checkpoint trees are outside the inventory corpus; do not classify,
  hash, soft-delete, or report their files as metadata.
- Permission errors should mark individual files as failed or skipped with diagnostics and continue.
- Large file hashing should be streaming.
- Archives, installers, binaries, and model artifacts must not be opened beyond metadata and optional streaming hash.
- The run must tolerate unusual extensions and no-extension files.
- The EDA notebook must open generated app data read-only and must not mutate SQLite, `Courses`, or any course file.
- The EDA notebook must not execute course code or course notebooks.
- The EDA notebook must be kept in sync with inventory schema, command, and status/skip-reason semantics.
- Notebook outputs and execution counts should be cleared before commit.

## Tests

- Automated fixture test with two course folders and mixed file types.
- Verify exact course names and relative paths are preserved.
- Verify images, media, archives, installers, binaries, and model artifacts become metadata-only with reasons.
- Verify extractable files become pending.
- Verify rerunning inventory is idempotent.
- Verify a changed timestamp triggers hash comparison.
- Verify a missing file is marked without hard deletion.
- Verify the inventory EDA notebook is valid notebook JSON, imports pandas successfully, and documents its read-only safety boundary.
- Verify the inventory EDA notebook is updated when inventory output fields or interpretation rules change.
- Optional smoke: run inventory on a tiny selected subtree copied into a temp fixture, not the full `Courses` archive.

## Acceptance Criteria

- `uv run -m uni_rag_agent inventory run` fills `courses` and `files`.
- Every file has exactly one category and one index status.
- Metadata-only files have useful `reason_not_indexed`.
- Inventory summaries can explain the mixed archive without extracting content.
- `notebooks/inventory_eda.ipynb` can be opened after `inventory run` to inspect inventory distribution and extraction backlog without mutating app data or source course files.
- No automated test requires traversing the full `Courses` archive.
