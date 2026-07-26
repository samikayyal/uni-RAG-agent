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
   single/multi-query semantic search with request-scoped provider and Chroma
   reuse.
5. Mandatory LLM query planning, metadata/keyword/semantic orchestration, RRF
   provenance, non-persisting `retrieve` execution with CLI run telemetry, and
   persisted evidence packets and coverage.
6. Strict packet-only Markdown answer generation with short claim-focused
   paragraphs, minimal direct citations, deterministic response/paragraph fence
   normalization, provider-compatible empty limitations handling, deterministic
   citations/references, append-only answer traces, bounded planner-only
   sessions, and `ask`.
7. FastAPI/UI routes with startup-constructed process-scoped planner and answer
   models, timeout-safe persistence boundaries, server-verified session resume,
   stale-history reconciliation, deduplicated structured answer rendering,
   visible failure states, bidirectional answer text, active ask phase/elapsed
   feedback, safe cancellation that prevents late answer persistence, and a
   bounded web-settings surface (embedding profile and retrieval tuning
   overrides persisted in `data/app_settings.json`, DEC-041).
8. Fixture-isolated evaluation preparation, deterministic scoring, atomic state
   activation, drift validation, and redacted JSON/Markdown reports.
9. Cross-cutting maintenance hardening: one canonical logical-index taxonomy and
   a thin CLI composition root with separated command families, renderers, and
   telemetry adapters.
10. Gemini document indexing uses the direct asynchronous Batch API with
    inline requests and per-job polling; interactive query batches remain
    synchronous so semantic search does not wait on a batch job.
11. An explicit public-demo mode preserves the local web contract while adding
    Turnstile-to-signed-token exchange, client-bound session ownership,
    Firestore-backed idempotent quotas, a two-slot ask semaphore, request-scoped
    bounded settings, per-tab history, and no public numeric history routes.
12. Cloud Run deployment is operator-owned: the image consumes the selected
    SQLite database and Chroma vectors directly from `data/`, only the ignored
    offline model is staged with PowerShell, and the non-root multi-stage image
    uses `/app` for packaged files and `/data` for mutable runtime state. On
    2026-07-25, revision `uni-rag-agent-00005-v2q` was deployed from immutable
    image digest `sha256:0cede5317d2e9de552cd781006a25140cd33cb5de93154d710277cd23de79109`;
    its startup/readiness probes and live storage/EmbeddingGemma readiness
    checks passed. The operator runbook now also records reproducible `gcloud`
    bootstrap/identity checks and the DNS-only
    `unirag.samikayyal.com` custom-domain cutover; no custom-domain mapping is
    claimed as live until its own Turnstile and browser checks pass.
13. Public-demo hardening rejects completed request-id replays before provider
    work or capacity acquisition, normalizes all quota infrastructure failures
    to the abuse-service 503 boundary. Deployment scanning, completeness checks,
    and optional benchmark/smoke scripts are intentionally operator-owned and
    are not part of the application.
14. Direct Gemini embeddings use a dedicated `GOOGLE_API_KEY_EMBEDDING`;
    planner and answer Gemini key rotation remains on `GOOGLE_API_KEY` and
    optional `GOOGLE_API_KEY_2`.
15. Hosted readiness now checks storage plus the serving default vector space
    without constructing an embedding provider. Optional local Hugging Face
    profiles remain lazy, can be prepared through the authenticated quota-free
    browser control route before settings persist, and retain ask-time lazy
    loading as a fallback.
16. Web UI reliability/accessibility hardening now prevents concurrent asks,
    clears stale result state on each new ask, preserves Enter-to-submit and
    Shift+Enter-to-newline input, bounds mobile layout content, restores input
    focus and reachable dialog actions, reduces screen-reader announcement
    noise, keeps citation touch targets in normal answer flow, wraps long
    retrieval contribution metadata inside trace cards, applies strict settings
    input/resource-id HTTP validation, grows the
    non-resizable composer with typed content, clears it as soon as a question
    is accepted for submission, restores the submitted draft after cancellation
    or failure, uses its own visible empty-question validation, keeps the Ask
    action visible in short phone viewports, maintains even mobile citation-line
    rhythm, gives the zero-edit Undo action a plain label, and preserves an explicit
    light/dark appearance choice locally in the browser. A responsive history
    header now also provides an immediate no-confirmation clear action that
    removes all browser-held sessions and returns the conversation surface to
    its initial blank state; versioned script URLs prevent stale cached
    JavaScript from leaving the control unbound after an update. Session and
    request IDs also retain a cryptographically random UUID fallback for
    plain-HTTP LAN browsers that omit `crypto.randomUUID()`.
    Preparing an explicitly selected local embedding profile also shows a
    top-bar loading indicator, with an explicit Gemma label for
    EmbeddingGemma's lazy-load request. Settings saves are diff-based and
    locally validated, preserve field-level errors, close the dialog on
    success, retain an in-flight operation across dialog reopen, and reject a
    selected local profile that lacks a usable vector collection. The settings
    dialog now presents effective next-ask values in compact retrieval-stage
    groups, marks edits with prior values and undo controls, and keeps its
    actions reachable in a full-height phone layout. The web
    surface no longer serves an OpenAPI schema.
17. The web UI and API no longer expose archive-wide index composition; index
    counts and per-course indexing state remain internal to the offline
    inventory, extraction, and retrieval workflows.

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
