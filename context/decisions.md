# Binding decisions

This file contains only decisions that still constrain public behavior. Source
code and focused tests define details not listed here. Historical or superseded
alternatives are intentionally removed; use Git history when that rationale is
needed.

## Product and safety

### DEC-001 — Course archive intelligence, not generic folder chat

**Decision:** Answers are grounded in the configured university archive and
must expose supporting paths/locations and search coverage.

**Why:** The useful product is auditable course knowledge, not unconstrained
conversation.

**Constraints/consequences:** Prefer an explicit insufficiency or “not found in
indexed materials” result to unsupported synthesis; preserve exact source names
and paths.

### DEC-002/003/006/007 — Selective, non-destructive source admission

**Decision:** Inventory records every discovered file but extracts/indexes only
supported text and schema categories. Images, audio/video, archives, installers,
binaries, serialized/model artifacts, and unknown extensions are metadata-only;
existing transcripts are eligible, automatic media transcription is not.

**Why:** Full-folder ingestion is noisy and can load unsafe or expensive
artifacts.

**Constraints/consequences:** Never execute course code/notebooks or mutate
`Courses/`. Scanned-PDF OCR is optional and gated by `UNI_RAG_OCR_ENABLED`;
standalone images are not OCR/caption inputs. `.ipynb_checkpoints` subtrees are
excluded before inventory (DEC-040).

### DEC-009/021 — `uv` and environment configuration

**Decision:** `uv` owns dependency installation and Python command execution;
configuration comes from typed environment values with `.env` support.

**Why:** One reproducible local workflow and explicit operational settings.

**Constraints/consequences:** Use `uv add`, `uv sync`, and `uv run`. Provider
and model pairs are validated together; safe config output and telemetry omit
secrets and absolute local paths where the public surface requires it.

## Storage and pipeline boundaries

### DEC-011 — SQLite authority with Chroma logical indexes

**Decision:** SQLite is authoritative for archive metadata, chunks, lifecycle
records, evidence, and answers. Chroma stores vectors in separate physical
collections for stable logical indexes (`document`, `slides`, `notebook`,
`code`, `data_schema`, `transcript`).

**Why:** Relational joins and audit history must remain available even when
vector state is rebuilt or unavailable.

**Constraints/consequences:** Vector rows map back to exact SQLite chunks;
physical collections are model/profile-namespaced. Full DDL and migrations live
in [`storage/core.py`](../src/uni_rag_agent/storage/core.py).

### DEC-023/029/028/040 — Current-file and deletion semantics

**Decision:** Normal indexes and retrieval include only chunks joined to files
whose latest `index_status` is `indexed`. Inventory changes are hash/timestamp
aware; missing files are soft-marked. Re-extraction may replace chunks,
historical search-result chunk references become `NULL`, and embedding mappings
are cascaded away. Checkpoint paths never receive rows.

**Why:** Historical diagnostics are valuable, but stale content must not appear
in current answers.

**Constraints/consequences:** FTS5 and Chroma share one eligibility predicate;
reset/rebuild is the supported way to remove generated state after a source
policy change.

### DEC-012/019 — Source-aware bounded chunks

**Decision:** Extraction emits source-aware chunks (one notebook cell or natural
format unit, split at a bounded token size) with location metadata; data files
emit schema/sample summaries rather than full data content.

**Why:** Retrieval and citations need stable, readable units without embedding
large datasets or losing page/slide/cell context.

**Constraints/consequences:** Whole chunks are the evidence selection unit;
source type and location fields are compatibility-sensitive.

### DEC-022/025 — Isolated failures and safe summaries

**Decision:** Inventory/extraction/data-summary runs continue per file and record
sanitized failure state. Supported tabular/JSON/SQLite formats produce bounded
schema, counts, and samples.

**Why:** One malformed or unsupported file must not discard a usable corpus or
force unsafe loading.

**Constraints/consequences:** Failed files remain visible for repair; full data
payloads and arbitrary serialized objects are not loaded into retrieval.

## Retrieval and answering

### DEC-014/033 — Mandatory planner, deterministic hybrid retrieval, RRF

**Decision:** `retrieve` and persisted evidence builds call the configured LLM
planner once, validate a `QueryPlan`, run metadata/keyword/semantic backends with
hard plan filters, and merge ranked results with unweighted RRF. A valid
unsupported plan is an empty successful run; invalid plans or backend/provider
failures are fatal. Two deterministic plan adjustments are part of validation
(2026-07-18, motivated by manual-QA findings BUG-04/BUG-05):

1. *Slides broadening:* a plan scoping to `slides_index` without
   `document_index` has `document_index` appended before execution, because
   slide decks are frequently ingested with `source_type=document` and a
   slides-only scope silently excludes them. This is the only automatic scope
   broadening; the adjusted plan is then applied as a hard filter and is what
   persistence and coverage report.
2. *Low-confidence downgrade:* a structurally valid plan whose
   `plan_confidence` is below `query_plan_min_confidence` is downgraded to an
   `unknown_or_unsupported` plan with empty scopes and an explanatory
   `plan_reason`, instead of raising `QueryPlanningError`. Low confidence is a
   retrieval outcome (honest insufficient-evidence answer), not a provider
   failure.

**Why:** Structured intent and auditable backend provenance are more stable than
duplicated routing rules or an opaque reranker. The two adjustments keep hard
filtering while preventing the two systematic false negatives observed in
testing: false insufficient-evidence for document-typed decks, and 502
provider errors for out-of-scope questions.

**Constraints/consequences:** Planner settings remain separate from answer
settings. `retrieve` never writes search/evidence rows; use `evidence build`
for persistence. An explicit reviewed embedding profile and `llm` extra are
required when retrieval executes.

### DEC-031/039 — Explicit reviewed embedding profiles

**Decision:** There is no production fake/default embedding model. A selected
profile is resolved from the registry, provider is inferred, and the canonical
identity is stored everywhere. Supported profiles are the four reviewed local
Hugging Face models, `google/gemini-embedding-001` (alias
`gemini-embedding-001`), and `Qwen/Qwen3-Embedding-8B`.

**Why:** Vector spaces, dimensions, credentials, and cost boundaries must be
explicit and reproducible.

**Constraints/consequences:** Local profiles use `embeddings`; hosted profiles
use `embeddings-cloud`. Google uses direct Gemini with `GOOGLE_API_KEY` (not
Vertex AI); Nebius uses its fixed Token Factory endpoint and `NEBIUS_API_KEY`.
SDKs load lazily. Batches validate finite vectors and dimensions, retry only
transient/network/408/429/5xx failures (three total attempts), and commit each
successful batch; test doubles exist only at loader seams.

### DEC-034 — Persisted evidence boundary

**Decision:** `evidence build` persists the validated plan/settings, complete raw
result sets and completion envelopes, fused rows, coverage, and one immutable
packet of authoritative current chunks. `retrieve` does not persist SQLite
search/evidence rows or mutate Chroma or `Courses/`; its CLI still writes JSONL
run telemetry under `data/runs/`.

**Why:** Answering needs an exact, replayable handoff rather than mutable or
snippet-only retrieval output.

**Constraints/consequences:** File-only metadata rows are audit/coverage data,
not synthetic evidence. Whole chunks are selected within a positive 12,000-token
default budget and `final_top_k`; omissions become deterministic weaknesses.

### DEC-035/020 — Strict packet-only answers and citations

**Decision:** The answer model receives only packet evidence and returns one JSON
object with `answer_paragraphs` and `limitations`. The application validates
packet-relative citations, canonicalizes `chunk:<id>` aliases to stable `[E1]`
style positions, renders references, and appends an answer trace.

**Why:** Model prose must not expand the evidence boundary or invent source
identifiers.

**Constraints/consequences:** Answer provider/model configuration is separate and
required only for non-empty packets. Prompt size is bounded (16,000 token
default); empty/budget-exhausted packets bypass the provider. Invalid output is
retried according to configuration then becomes a safe no-citation refusal;
provider failure creates no answer row. Session context is bounded and planner-
only.

## Application and evaluation

### DEC-036/017 — Thin local web app with process-scoped models

**Decision:** FastAPI serves a package-owned UI and only answer/inspection
routes. Configured planner and answer models are constructed during application
startup and reused across requests. Omitted session ids are stateless; supplied
ids use a bounded in-process planner-only LRU (20 sessions, two-hour inactivity
TTL). Ask timeout defaults to 120 seconds and cannot append a late answer after
timeout.

**Why:** The web layer should present existing services without turning browser
requests into ingestion or source mutation.

**Constraints/consequences:** `/health` is liveness-only and does not invoke a
provider; startup construction failures are surfaced through sanitized ask
errors. Planner and answer configurations remain separate. Ingestion, indexing,
evaluation, upload, and reset remain CLI operations. Cached models and sessions
disappear on process restart.

### DEC-037/038 — Isolated, safe evaluation

**Decision:** Fixture evaluation uses strict UTF-8 `evals/fixtures.json`, isolated
`data/runs/eval/fixture-state`, atomic validated activation, deterministic
retrieval/citation/limitation scoring, and paired safe JSON/Markdown reports.
Real-archive smoke mode is explicit.

**Why:** Evaluation must be repeatable without mutating normal archive state or
leaking queries, evidence, model output, or credentials.

**Constraints/consequences:** `eval prepare-fixtures` must complete before fixture
`eval run`; manifests detect identity/count drift. Reports retain trace ids,
failures, and p50/p95 timings only. Automated tests use injected doubles; public
fixture commands use configured production providers.
