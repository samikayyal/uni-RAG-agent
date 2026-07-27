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
failures are fatal. Three deterministic plan adjustments are part of validation
(2026-07-18/19, motivated by manual-QA findings BUG-04/BUG-05 and packet 47):

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
3. *Keyword phrase expansion:* planner-supplied keyword terms are normalized,
   tokenized, deduplicated, and OR-matched in FTS5. A multi-word planning term
   is not treated as an exact phrase because that produced systematic zero-hit
   keyword result sets even when relevant chunks contained one or more of its
   informative tokens. Tokenless individual terms are skipped; only an entirely
   tokenless term set fails validation.

**Why:** Structured intent and auditable backend provenance are more stable than
duplicated routing rules or an opaque reranker. These adjustments keep hard
filtering while preventing systematic false negatives observed in testing:
false insufficient-evidence for document-typed decks, 502 provider errors for
out-of-scope questions, and empty keyword sets caused by strict multi-word
phrases.

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
use `embeddings-cloud`. Google uses direct Gemini with the dedicated
`GOOGLE_API_KEY_EMBEDDING` (not Vertex AI); document indexing uses the Gemini
Batch API while interactive query embedding remains synchronous. Gemini planner
and answer models retain `GOOGLE_API_KEY` with optional `GOOGLE_API_KEY_2`
quota failover; Nebius uses its fixed Token Factory
endpoint and `NEBIUS_API_KEY`. SDKs load lazily. Batches validate finite
vectors and dimensions, retry only transient/network/408/429/5xx failures
(three total attempts), and commit each successful batch; Gemini Batch API job
creation is the exception because retrying that non-idempotent submission can
duplicate paid work, while its read-only polling calls retain the retry policy.
Test doubles exist only at loader seams.

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
provider failure creates no answer row. Answer paragraphs use Markdown syntax,
remain short and claim-focused, and request only the smallest directly supporting
citation set. A whole-response lowercase `markdown` fence is removed by an
anchored deterministic normalization before strict JSON parsing, and the same
normalization applies independently to each paragraph text field. Other response
wrappers remain invalid. Session context is bounded and planner-only.

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
disappear on process restart. The browser must verify process-session liveness
before presenting persisted client history as an active continuing conversation;
persisted answers may still be shown after context expiry, but are detached from
the next ask. A browser-generated active request id may expose only transient
phase/elapsed telemetry and a cancellation action. Cancellation must use the
same persistence gate as timeout so an in-flight provider call can unwind
without appending an answer; this transient telemetry is not retained as
history, and the generic busy message remains valid when telemetry is absent.

### DEC-041 — Bounded web-adjustable retrieval settings

**Decision:** The web app exposes `GET/PUT /api/settings` for an explicit
allowlist of non-sensitive retrieval tuning values only: the embedding model
(reviewed profiles, aliases canonicalized), `keyword_top_k`, `semantic_top_k`,
`metadata_top_k`, `final_top_k`, `rrf_k`, `semantic_query_limit`,
`query_plan_min_confidence`, `filename_fuzzy_threshold`,
`path_fuzzy_threshold`, and `evidence_max_tokens`. Overrides persist in
`data/app_settings.json` and layer on top of environment configuration for web
requests only.

**Why:** Retrieval tuning is safe to iterate from the browser; provider,
credential, storage, and operational-guard settings are not.

**Constraints/consequences:** LLM/answer provider-model pairs, API keys,
storage paths, log level, OCR, `answer_max_retries`,
`answer_prompt_max_tokens`, `answer_session_message_limit`, and
`ask_timeout_seconds` are never web-settable; requests naming them are
rejected. Every numeric value is bounds-checked before persistence, `null`
clears one override, and invalid or unknown entries in the stored file are
dropped on read instead of failing requests. The CLI ignores web overrides.
DEC-036/017's ban on per-request provider/model overrides is unchanged.

### DEC-042 — Explicit public-demo boundary and deployment contract

**Decision:** Local web behavior remains the default. Public behavior is enabled
only by `UNI_RAG_PUBLIC_DEMO_ENABLED=true` and is a separate bounded contract:
Cloudflare Turnstile is exchanged once for a short-lived signed client-bound
token; browser state and complete answer payloads live per tab in
`sessionStorage`; settings are request-scoped and restricted to the three
predeployed embedding profiles; historical numeric answer, packet, and coverage
routes are hidden. Firestore atomically enforces idempotent request reservations
with 3 starts per rolling minute, 10 per client digest per UTC day, and 100
globally per UTC day. A request consumes quota only after acquiring one of two
process ask slots; subsequent provider failure or client cancellation still
counts. Cloud Run request concurrency is four so control/static requests remain
responsive while two asks run.

Every authenticated public ask submission is also retained indefinitely in a
separate Firestore `demo_asks` document. Recording begins after bearer-token
validation but before replay, capacity, session-ownership, or quota decisions,
so valid rejected attempts are visible alongside accepted asks. The terminal
record contains the raw query and raw client IP, bounded User-Agent/language
metadata, requested/effective safe settings, model identities, quota state,
sanitized error state, trace ids, phase timings, and—when completed—the full
safe public response including answer, citations, limitations, coverage, and
evidence packet. Unauthenticated requests and payloads rejected by FastAPI
schema validation are not retained. Audit writes fail closed. The audit store
never receives bearer tokens, Turnstile responses, authorization headers,
secrets, raw exception details, or stack traces.

New ledger documents use `schema_version: 2` and Firestore server timestamps
for `updated_at` both on creation and every later update; schema-v1 records
without that field remain readable. The local audit dashboard synchronizes this
indefinite ledger into the separate, generated, sensitive SQLite cache
`data/ask_audit_cache.sqlite`. Bootstrap imports all records; later runs query
ascending `updated_at` from an inclusive timestamp high-water mark and use
idempotent upserts, so equal timestamps, retries, and terminal updates are not
lost. Firestore deletion is intentionally not mirrored during ordinary sync;
`--rebuild-cache` is the explicit exact reconstruction operation.

A request id is a one-shot acceptance identity within its signed demo-token
nonce. A completed replay returns a stable conflict and cannot start retrieval,
consume capacity, or mutate session ownership, even though the existing
reservation remains readable for Firestore transaction idempotency. Firestore
infrastructure failures, including quota-summary reads, are abuse-service
outages and therefore return the safe 503 contract.

Hosted runtime assets are staged by the operator and baked into a CPU-only,
non-root image. The offline EmbeddingGemma snapshot remains packaged, but the
model is lazy: saving a local-profile selection prepares it over the browser
request, while direct asks retain lazy construction as a fallback. Providers
and Chroma clients are process scoped, with per-profile local construction and
encoding locks. `/ready` checks storage plus only the serving default's vector
space, without constructing an embedding provider or probing external
providers. Optional profiles never gate readiness. Application code never
creates cloud resources; the operator follows the checked-in GCP and Cloudflare
runbook.

**Why:** A public unauthenticated demo has materially different privacy, abuse,
cost, memory, and state-lifetime risks from the private local UI. Making that
mode explicit keeps local workflows compatible while enforcing server-owned
limits. A durable cross-instance ledger is required to inspect every
authenticated submission and its outcome after Cloud Run instances disappear.

**Constraints/consequences:** Public startup fails closed when Turnstile,
signing, Firestore, offline-model, or upper-bound configuration is missing or
unsafe. The `demo_asks` collection is server/admin-only, has no public listing
route, and is intentionally exempted from indexing for large/nested fields;
only fields needed for dashboard ordering/filtering remain indexed. Retention
is indefinite by explicit product choice, the public UI discloses raw-IP/full
trace retention, and IP-derived locations are resolved only by the separate
localhost dashboard using an operator-supplied GeoLite2 database. Local mode
does not upload private asks. Locations are the first successful local lookup
snapshot—not true event-time geography—and are frozen across ordinary syncs;
`--refresh-locations` is explicit. Selecting, reviewing, and staging the SQLite
database, Chroma indexes, and accepted offline model are operator
responsibilities; the application does not scan or certify deployment inputs.
GCP budget alerts are warnings, not spending caps; provider and platform limits
remain operator responsibilities.

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
