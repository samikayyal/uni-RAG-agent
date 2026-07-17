# Progress tracker

## Current state

The implemented pipeline is complete through evaluation hardening:

1. Typed configuration, safe logging, SQLite/FTS5 storage, and generated-state
   health checks.
2. Idempotent inventory with exact path preservation, selective source admission,
   soft-missing state, and `.ipynb_checkpoints` pruning.
3. Per-file extraction/chunking for supported documents, slides, notebooks,
   code, and transcripts; schema/sample summaries for CSV, XLSX, JSON, JSONL,
   SQLite, and DB files.
4. Current-file-only FTS5 and Chroma indexing with reviewed local/hosted
   embedding profiles, reconciliation, canonical model identity, and safe
   semantic search.
5. Mandatory LLM query planning, metadata/keyword/semantic orchestration, RRF
   provenance, non-persisting `retrieve` execution with CLI run telemetry, and
   persisted evidence packets and coverage.
6. Strict packet-only answer generation, deterministic citations/references,
   append-only answer traces, bounded planner-only sessions, and `ask`.
7. Provider-lazy FastAPI/UI routes with timeout-safe persistence boundaries.
8. Fixture-isolated evaluation preparation, deterministic scoring, atomic state
   activation, drift validation, and redacted JSON/Markdown reports.

This documentation layer now mirrors those live contracts through
`context/README.md`, the compact overview/architecture/glossary/operations and
decisions pages, and the eight pages under `context/capabilities/`.

## Open work

- Tune slow full-archive filesystem scans if real-corpus measurements justify
  it; preserve inventory idempotency and checkpoint pruning while doing so.
- Keep read-only EDA notebooks aligned when a producing command, table, JSON
  artifact, status vocabulary, or interpretation rule changes.
- Optional future capabilities remain deliberately out of the MVP: opt-in
  audio/video transcription, selected standalone-image OCR/captioning,
  knowledge-graph exploration, portfolio mode, and study/quiz mode.

No current item changes the public contracts above. Any new behavior that does
must add a short binding decision and update the affected capability page.
