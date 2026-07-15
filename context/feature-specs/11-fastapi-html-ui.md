# Feature Spec 11: FastAPI HTML UI

## Purpose

Expose the answer workflow through a FastAPI backend and a simple HTML/JS frontend that displays answers, citations, references, and search coverage. Operational workflows such as ingestion, indexing, and evaluation remain CLI-first.

## Depends On

- [08-query-routing-and-hybrid-retrieval.md](08-query-routing-and-hybrid-retrieval.md)
- [09-evidence-packets-and-coverage.md](09-evidence-packets-and-coverage.md)
- [10-answering-and-citations.md](10-answering-and-citations.md)
- DEC-017, DEC-020, DEC-021

## In Scope

- FastAPI app setup.
- Health/config-safe endpoint.
- Ask endpoint that runs retrieve, evidence packet, and answer generation.
- Endpoint to fetch evidence packet/search coverage by ID.
- Simple static HTML/JS interface for entering questions and viewing results.
- Citation, references, limitations, and searched/found/missing display.
- Basic request validation and error responses.

## Out of Scope

- Polished visual design.
- Authentication.
- Browser-based ingestion/indexing controls.
- File upload.
- Editing or deleting `Courses` files.
- Real-time streaming unless trivially supported.

## Public Interfaces

Commands:

```powershell
uv run -m uni_rag_agent app serve
uv run -m uni_rag_agent app serve --host 127.0.0.1 --port 8000
```

HTTP endpoints:

```text
GET /health
GET /config
POST /api/ask
GET /api/search-runs/{search_run_id}/coverage
GET /api/evidence-packets/{evidence_packet_id}
GET /api/answers/{answer_id}
GET /
```

Notebook:

No UI-specific EDA notebook is required for the MVP. Inspect the underlying persisted traces through `notebooks/retrieval_eda.ipynb` and `notebooks/answering_eda.ipynb`; verify UI behavior through API/UI tests.

`POST /api/ask` request:

```json
{
  "query": "Explain MapReduce from my courses",
  "session_id": "optional-session-id"
}
```

`POST /api/ask` response:

```json
{
  "answer_id": 1,
  "search_run_id": 1,
  "evidence_packet_id": 1,
  "answer_text": "...",
  "citations": [],
  "references": [],
  "limitations": [],
  "coverage": {}
}
```

`references` is a deduplicated first-appearance-ordered projection of the
authoritative structured citations. Each item contains exactly
`citation_id`, `course`, `file_path`, `source_type`, and `location_label`.
Coverage and evidence-packet lookup endpoints return the existing safe model
dictionary directly, without a generic `data` wrapper. Answer lookup returns
the same public answer shape as `POST /api/ask`, including ids, references, and
coverage.

## HTTP, Session, and Timeout Contract

- `app serve` defaults to `127.0.0.1:8000`; callers may explicitly choose a
  different host or valid port. Reload and multi-worker controls are not part
  of Feature 11.
- A missing `session_id` performs one stateless ask. A supplied id must match
  `[A-Za-z0-9_-]{1,128}` and reuses the existing custom planner-only
  `AnswerSession` in process. The registry keeps at most 20 sessions, expires
  sessions after two hours of inactivity, evicts least-recently-used entries,
  serializes requests within one session, and permits different sessions to
  execute concurrently.
- `UNI_RAG_ASK_TIMEOUT_SECONDS` is a positive integer and defaults to `120`.
  Timeout returns HTTP 504. Synchronous provider work may finish after the HTTP
  timeout, but the timed-out request must not append an answer afterward; an
  evidence packet persisted before answer timeout remains inspectable.
- Request queries are trimmed, must be nonblank, and may contain at most 10,000
  characters. Unknown request fields and invalid session ids are rejected.
  Provider/model overrides are not accepted through HTTP.
- Error responses have the exact envelope
  `{"error":{"code":"...","message":"..."}}`. Validation uses 422,
  missing ids use 404, invalid or missing production configuration uses 503,
  planner/retrieval/embedding/answer-provider failures use 502, timeout uses
  504, and storage corruption or unexpected failures use 500. Responses never
  include traces, prompts, credentials, or sensitive filesystem details.
- `/health` is liveness-only and returns `{"status":"ok"}` without loading
  providers or requiring initialized storage. `/config` returns operational
  non-secret settings and path-existence booleans, but not absolute paths.

## Storage and Schema Impact

No new tables. The API reads and writes through retrieval, evidence, and answering services:

- `search_runs`
- `search_results`
- `evidence_packets`
- `answers`

Session memory may be in-process for MVP and should not require cross-session persistence.

## Workflow

1. User opens `/`.
2. Frontend sends query to `POST /api/ask`.
3. Backend routes and retrieves evidence.
4. Backend builds and stores an evidence packet.
5. Backend generates and stores an answer.
6. Frontend renders answer, inline citations, references, limitations, and coverage.
7. User can inspect evidence packet or coverage details by ID.

The first screen is the question-answering interface. It renders structured
citation/reference cards, limitations, searched/found/missing coverage, and an
expandable evidence view with persisted-resource inspection links. It exposes
no ingestion, indexing, evaluation, upload, or source-mutation controls.

Static assets are package-owned files under
`src/uni_rag_agent/app/static/` (`index.html`, `app.js`, and `styles.css`) and
do not require a template engine.

## Failure and Safety Rules

- API errors must return structured JSON with a human-readable message.
- `/config` must not expose secrets or API keys.
- The UI must not offer controls that mutate `Courses`.
- Ingestion/index/eval operations stay CLI-first for MVP.
- Empty retrieval should return a valid insufficient-evidence answer, not a server error.
- Long-running ask requests should report clear timeout or partial failure messages.
- Do not add browser-driven notebooks for MVP UI validation; use tests and the retrieval/answering notebooks for trace analysis.

## Tests

- Automated FastAPI tests with test client and injected retrieval/answering doubles.
- Verify `/health` works without initialized real providers.
- Verify `/config` hides secrets.
- Verify `/api/ask` returns answer IDs, citations, references, and coverage.
- Verify insufficient-evidence responses render as successful API responses.
- Verify static UI file loads.
- Optional smoke: start the app with explicitly configured production models and ask against a tiny fixture database.

## Acceptance Criteria

- `uv run -m uni_rag_agent app serve` starts the local app.
- The first screen is a usable question-answering interface, not a marketing page.
- API responses expose answer, citations, references, limitations, and coverage.
- Operational ingestion/index/eval workflows remain documented as CLI commands.
