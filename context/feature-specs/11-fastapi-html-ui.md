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

## Failure and Safety Rules

- API errors must return structured JSON with a human-readable message.
- `/config` must not expose secrets or API keys.
- The UI must not offer controls that mutate `Courses`.
- Ingestion/index/eval operations stay CLI-first for MVP.
- Empty retrieval should return a valid insufficient-evidence answer, not a server error.
- Long-running ask requests should report clear timeout or partial failure messages.

## Tests

- Automated FastAPI tests with test client and fake retrieval/answering adapters.
- Verify `/health` works without initialized real providers.
- Verify `/config` hides secrets.
- Verify `/api/ask` returns answer IDs, citations, references, and coverage.
- Verify insufficient-evidence responses render as successful API responses.
- Verify static UI file loads.
- Optional smoke: start app with fake adapters and ask against a tiny fixture database.

## Acceptance Criteria

- `uv run -m uni_rag_agent app serve` starts the local app.
- The first screen is a usable question-answering interface, not a marketing page.
- API responses expose answer, citations, references, limitations, and coverage.
- Operational ingestion/index/eval workflows remain documented as CLI commands.
