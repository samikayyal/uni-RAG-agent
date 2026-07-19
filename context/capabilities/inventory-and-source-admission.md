# Inventory and source admission

## Current behavior

`inventory_courses()` walks the configured `Courses` root without following
symlinks, prunes every path containing a case-insensitive
`.ipynb_checkpoints` component, preserves exact course/file names and relative
paths, and records an idempotent inventory run. Extension classification marks
supported documents, slides, notebooks, code, data schemas, and existing VTT
transcripts as pending; images, media, archives, installers, binaries, model
artifacts, and unknown extensions are metadata-only with a reason. Hashes are
recomputed when file metadata indicates change. Files absent from a later run
are soft-marked `skipped` rather than removed.

Files directly in the `Courses` root (outside any course directory) are
assigned to a synthetic course named `General Resources` (reserved id `-1`,
fallback `999` if `-1` is taken) so course-scoped retrieval can reach them;
previously they carried `course_id NULL` and were unreachable. The synthetic
row's identity is its *path* (the Courses root itself), not the name; the name
`General Resources` is reserved, and an inventory run fails with a clear
diagnostic if a real course directory uses it.

## Public entry points

- `uv run -m uni_rag_agent inventory run`
- `uv run -m uni_rag_agent inventory summary`
- Python: `inventory_courses(config)` and `load_inventory_summary(config)`.

## Source, tests, and artifacts

- Source: `src/uni_rag_agent/inventory/{core,file_io,classification,models}.py`
  and `src/uni_rag_agent/source_filters.py`.
- Tests: `tests/test_inventory.py` (including failure, idempotency, missing-file,
  and checkpoint-path coverage).
- Notebook: `notebooks/inventory_eda.ipynb` (read-only inventory EDA).
- SQLite: `courses`, `files`, and `extraction_runs`; run results and sanitized
  lifecycle events are also written under `data/runs/`.

## Invariants and failure boundaries

- `Courses/` is read-only. A root-listing failure fails the run; an individual
  stat/hash/directory diagnostic is recorded without silently admitting an
  unknown file.
- Every admitted path retains its exact spelling; no normalization or aliasing
  of course names is allowed.
- Checkpoint trees are excluded before classification, hashing, or persistence;
  they must not appear in downstream tables or indexes.
- Classification is metadata-only for unsafe/noisy types. Inventory never
  extracts content or executes a file.

Binding decisions: [DEC-001](../decisions.md#dec-001--course-archive-intelligence-not-generic-folder-chat),
[DEC-002/003/006/007](../decisions.md#dec-002003006007--selective-non-destructive-source-admission),
and [DEC-023/029/028/040](../decisions.md#dec-023029028040--current-file-and-deletion-semantics).
