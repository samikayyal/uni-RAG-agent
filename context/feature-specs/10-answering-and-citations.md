# Feature Spec 10: Answering and Citations

## Purpose

Generate final user-facing answers strictly from evidence packets, with inline citations, a references section, and clear insufficient-evidence behavior.

## Depends On

- [09-evidence-packets-and-coverage.md](09-evidence-packets-and-coverage.md)
- `context/architecture.md` tables: `answers`, `evidence_packets`
- DEC-004, DEC-010, DEC-018, DEC-020, DEC-021

## In Scope

- Load an evidence packet and generate an answer using only packet evidence.
- Use LangChain chat model abstraction with configurable provider/model.
- Use an injected test-only chat model for deterministic answer-generation tests.
- Enforce inline citation format.
- Include a references section listing cited files and locations.
- Refuse or qualify answers when evidence is insufficient.
- Store final answer traces.
- Support per-session conversation context for follow-up query planning while keeping each packet self-contained.
- Add or update the answering EDA notebook for persisted answer and citation traces.

## Out of Scope

- Retrieval.
- UI rendering.
- Direct source-file inspection.
- Claims based on model memory or unstored conversation context.
- Cross-session persistent memory.

## Public Interfaces

Command:

```powershell
uv run -m uni_rag_agent answer --evidence-packet-id 1
uv run -m uni_rag_agent ask "Explain MapReduce from my courses"
```

Notebook:

```text
notebooks/answering_eda.ipynb
```

Create this notebook when answering is implemented. It should inspect `answers`, joined `evidence_packets`, citation JSON, limitations, model traces, injected-test behavior, and insufficient-evidence handling.

Internal interfaces:

```python
generate_answer(packet: EvidencePacket, conversation_context: list[dict] | None = None) -> AnswerResult
store_answer(evidence_packet_id: int, answer: AnswerResult) -> int
format_citation(evidence_item: EvidenceItem) -> str
validate_answer_citations(answer: AnswerResult, packet: EvidencePacket) -> CitationValidationResult
```

Answer result fields:

```text
answer_text
citations
limitations
model_name
```

Required answer format:

```text
<answer with inline citations>

References:
- <course> - <file path> - <location>

Limitations:
- <weakness or insufficient evidence note, when relevant>
```

## Storage and Schema Impact

Read:

- `evidence_packets`

Populate:

- `answers`

`answers.citations_json` must contain structured citations mapped to evidence items, not only rendered strings.

## Workflow

1. Load evidence packet by ID or receive packet from the ask pipeline.
2. If evidence is empty or weak, produce an insufficient-evidence answer using searched/found/missing coverage.
3. Build a prompt that includes only packet evidence and answer constraints.
4. Generate answer via the configured LangChain chat model.
5. Validate that every citation maps to packet evidence.
6. Add or repair references section when possible.
7. Store answer text, citations, limitations, and model name.
8. Keep `notebooks/answering_eda.ipynb` aligned with answer fields, citation JSON shape, limitation semantics, model trace fields, and injected-test behavior.

## Failure and Safety Rules

- The answer generator must not cite files absent from the packet.
- The answer generator must not answer from model memory when evidence is missing.
- If citation validation fails, return a safe insufficient-evidence response or retry with stricter instructions.
- Do not include API keys or internal prompts in stored answers.
- Conversation memory may help interpret follow-up queries, but evidence packets remain self-contained.
- The EDA notebook must read generated app data only and must not mutate SQLite, evidence packets, answers, or `Courses`.
- Notebook outputs and execution counts should be cleared before commit.

## Tests

- Automated tests inject a deterministic test-only chat model for cited answer, insufficient evidence, weak retrieval, and invalid citation repair/refusal.
- Verify every inline citation maps to a packet evidence item.
- Verify references section includes full file paths and locations.
- Verify answers with empty evidence do not invent facts.
- Verify answer-generation tests do not require production model credentials.
- Verify `notebooks/answering_eda.ipynb`, once created, is valid notebook JSON, imports pandas successfully, and documents its read-only safety boundary.
- Optional smoke: run `uv run -m uni_rag_agent ask "query"` against a tiny fixture index with explicitly configured production models.

## Acceptance Criteria

- `uv run -m uni_rag_agent answer --evidence-packet-id ...` writes an evidence-grounded answer.
- Answers cite course, file, and location where available.
- Insufficient evidence is explicit and useful.
- Stored answer traces are auditable back to the exact packet.
- `notebooks/answering_eda.ipynb` exists once this feature lands and can inspect answer/citation traces without mutating generated or source data.
