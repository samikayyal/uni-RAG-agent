# Web application

## Current behavior

`create_app()` builds a FastAPI application and serves the package-owned
HTML/JavaScript screen in one of two explicit modes. Local mode remains the
default and preserves file-backed web settings, durable server-resource
lookups, and browser `localStorage`. Public demo mode is enabled only by
`UNI_RAG_PUBLIC_DEMO_ENABLED=true`; it uses request-scoped retrieval settings,
per-tab `sessionStorage`, Cloudflare Turnstile exchange, signed client-bound
demo tokens, Firestore-backed usage quotas, and a process semaphore around ask
work. During normal application startup, configured planner
and answer models are constructed once and retained in a process-scoped
registry for reuse across requests. Startup construction failures do not turn
the liveness route into a provider health check; the relevant ask request
surfaces the sanitized configuration/provider failure. The app exposes the
existing ask workflow and safe persisted-resource projections; ingestion,
indexing, evaluation, upload, and source mutation are not web operations. An
omitted `session_id` is stateless. A valid supplied id uses an in-process
planner-only session registry with at most 20 least-recently-used sessions, a
two-hour inactivity TTL, and per-session serialization. The default ask timeout
is 120 seconds; evidence stored before timeout remains inspectable, and no late
answer is appended. The browser reloads the active session's latest persisted
answer on startup and checks server-side session liveness before describing it
as continuing. Expired process context is detached from the stored answer, and
missing persisted answers prune their stale client-history entries. While an
ask is active, the browser displays elapsed time and a four-stage timeline
(planning, keyword search, semantic search, answer generation) driven by the
live phase telemetry, adding the cold-start embedding-model stage when that
phase is reported; stages stay queued when telemetry is unavailable. Stage
durations are measured in the browser from observed phase transitions, so they
are approximate and are labeled as such rather than presented as server-recorded
timings. The user can cancel the active ask, which abandons the response and
prevents a late answer write while in-flight provider work unwinds. Before a
question is asked the screen does not expose archive-wide indexing composition.
Public mode returns the complete
safe packet and remaining-quota projection with the ask response, so it never
needs to expose historical numeric answer, packet, or coverage lookups.

A Settings dialog lets the user adjust a bounded allowlist of retrieval tuning
values: the embedding model (reviewed profiles only, aliases canonicalized),
`keyword/semantic/metadata/final_top_k`, `rrf_k`, `semantic_query_limit`,
`query_plan_min_confidence`, the filename/path fuzzy thresholds, and
`evidence_max_tokens`. Overrides persist in `data/app_settings.json`, layer on
top of environment configuration for web requests only (the CLI never reads
them), and apply from the next ask. A blank/cleared field reverts to the
environment default. A missing, corrupted, or hand-edited file never breaks the
app: unknown names and invalid values are dropped on read. Provider/model
selection, credentials, storage paths, log level, OCR, retry, prompt-budget,
session-limit, and timeout settings are not web-settable. In public mode the
same dialog is per tab and request scoped: it never reads or writes
`data/app_settings.json`, is bounded more tightly, and can select only
EmbeddingGemma, Gemini Embedding, or Qwen/Nebius profiles that were deployed.

The composer grows with the typed question up to its visible maximum instead of
being manually resizable. A top-bar light/dark control persists the selected
appearance in browser-local state; it does not affect retrieval settings or any
server-side state.

## Public entry points

- `uv run -m uni_rag_agent app serve [--host 127.0.0.1] [--port 8000]`.
- Local routes: `GET /health`, `GET /config`, `GET/PUT /api/settings`,
  `POST /api/ask`,
  `GET /api/asks/{request_id}/progress`,
  `POST /api/asks/{request_id}/cancel`,
  `GET /api/sessions/{session_id}`,
  `GET /api/search-runs/{search_run_id}/coverage`,
  `GET /api/evidence-packets/{evidence_packet_id}`, and
  `GET /api/answers/{answer_id}`. `/` serves the UI and `/static` serves its
  assets.
- Public-only `POST /api/demo/session` verifies a single-use Turnstile token
  and returns a signed 30-minute demo token. Protected public routes require
  `Authorization: Bearer <demo-token>` and bind that token to a privacy-safe
  client-address digest. `PUT /api/settings` and historical numeric answer,
  packet, and coverage routes return 404 publicly.
- `POST /api/embedding-profiles/prepare` accepts one reviewed local Hugging Face
  profile. In public mode it requires the existing demo bearer token but does
  not reserve an ask slot or consume quota; it keeps the request open while the
  selected local model is constructed, then returns a safe ready/failure result.
- `POST /api/ask` accepts a nonempty query (up to 10,000 characters) and an
  optional alphanumeric/underscore/hyphen session id and client-generated
  request id. Provider/model overrides are not accepted through HTTP.
- `GET /api/settings` reports effective values, environment defaults, stored
  overrides, numeric bounds, and the reviewed embedding profiles. `PUT
  /api/settings` accepts a partial update of allowlisted settings only
  (`null` clears one override); out-of-bounds or unknown-profile values are a
  422 `settings_validation_error`, and any non-allowlisted field is rejected.

## Source, tests, and artifacts

- Source: `src/uni_rag_agent/app/{api,service,public_demo,settings}.py` and
  `src/uni_rag_agent/app/static/`.
- Tests: `tests/test_app.py` (local route projections, validation, sessions,
  cancellation, timeout, settings overrides, and sanitized failures) and
  `tests/test_public_demo.py` (mode split, authentication, quotas, public
  settings, browser state, and embedding registry behavior).
- Artifacts: routes read/write through the evidence and answering stores;
  web settings overrides persist in `data/app_settings.json`.

## Invariants and failure boundaries

- `/health` is provider/storage-independent and returns `{"status":"ok"}`.
  `/config` reports non-secret operational settings and path-existence flags,
  never credentials or absolute local paths.
- Answer, citation, reference, and evidence-packet projections carry
  course-relative file paths; absolute host paths are not exposed (packets
  persisted before this change retain their original absolute paths).
- Answer projections expose structured `answer_body` and `answer_status` fields
  for the UI while retaining the canonical rendered `answer_text`. The UI shows
  references and limitations once, preserves single-newline paragraphs, uses
  automatic bidirectional text direction, and visually distinguishes validation
  failures and insufficient-evidence outcomes. Coverage and packet weaknesses
  are shown only when they are not already present in structured limitations.
- `[E<n>]` markers in `answer_body` render as inline chips that reveal the
  matching cited-evidence card. Those cards quote the packet, so the evidence
  packet is fetched once per answer rather than only when the trace is open; a
  failed packet fetch degrades the cards to course and location text instead of
  blocking the answer. Coverage is reported as which planned sources produced
  chunk-backed hits plus the recorded aggregate counts; per-source hit counts do
  not exist in the coverage projection and are not invented.
- While startup session liveness is unknown, the ask control remains busy and
  submission is rejected rather than silently forking a new session.
- Planner and answer settings remain separate; each configured model is cached
  once per active configuration and shared by stateless and session requests.
- Errors are stable safe JSON: missing resources 404, invalid config 503,
  planner/retrieval/provider failures 502, timeout 504, and storage/unexpected
  failures 500. Successful insufficient-evidence answers remain 200.
- A timed-out/cancelled request cannot append an answer after the response;
  `PersistenceGate` protects the final write while preserving an evidence packet
  already committed. Active-request progress is transient, contains only a
  phase, elapsed seconds, and cancellation state, and disappears when work
  finishes; it is not persisted or exposed as session history. The browser
  permits one active submission only, disables starting a new session until it
  settles, and maps Enter to submit while Shift+Enter adds a newline.
- `/ready` validates storage and, in hosted mode, confirms only the effective
  serving default's Chroma vector space (`public_default_embedding_model` in
  public mode, otherwise `embedding_model`). It never constructs an embedding
  provider or probes Gemini/Nebius; optional local profiles do not gate it.
- Public quota reservations are atomic and idempotent per request id. A request
  is counted only after it obtains one of two ask slots; provider failures and
  client cancellation still count, while a capacity rejection does not.
  Defaults are 3 starts per rolling minute, 10 per privacy-safe client per UTC
  day, and 100 globally per UTC day. Replaying a completed request id returns
  409 `request_already_accepted` before capacity acquisition and never launches
  provider work; Firestore transaction retries may observe the reservation but
  cannot duplicate it.
- Public ask settings and query length have server-owned maxima. Public session
  ownership applies to ask, progress, cancellation, and liveness routes; a
  token for another client or session cannot operate them.
- Embedding providers and Chroma clients are process scoped. Local Hugging Face
  query encoding and construction are coordinated per profile, so a slow
  EmbeddingGemma preparation does not serialize unrelated Gemini/Nebius asks.
  Failed constructions are cached for ordinary asks, while an explicit
  preparation request retries that profile.
- Turnstile and Firestore read/write infrastructure failures use the stable 503
  `abuse_service_unavailable` envelope; none fall through as generic 500s.

Binding decisions: [DEC-036/017](../decisions.md#dec-036017--thin-local-web-app-with-process-scoped-models),
[DEC-035/020](../decisions.md#dec-035020--strict-packet-only-answers-and-citations),
[DEC-034](../decisions.md#dec-034--persisted-evidence-boundary),
and [DEC-041](../decisions.md#dec-041--bounded-web-adjustable-retrieval-settings),
[DEC-042](../decisions.md#dec-042--explicit-public-demo-boundary-and-deployment-contract).
