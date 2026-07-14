# Feature Spec 10: Answering and Citations

## Purpose

Generate final user-facing answers strictly from immutable evidence packets,
with stable inline citation markers, a deterministic references section, and
clear insufficient-evidence behavior.

## Depends On

- [09-evidence-packets-and-coverage.md](09-evidence-packets-and-coverage.md)
- `context/architecture.md` tables: `answers`, `evidence_packets`
- DEC-004, DEC-010, DEC-018, DEC-020, DEC-021, DEC-035

## In Scope

- Load an evidence packet and answer only from packet evidence.
- Use a separately configured LangChain chat model and injected deterministic
  test doubles.
- Enforce strict JSON, inline citation, references, and limitation contracts.
- Persist append-only answer traces and expose `answer` and one-shot `ask`.
- Bound in-process planner-only conversation context.
- Add the read-only `notebooks/answering_eda.ipynb` notebook.

## Out of Scope

- Retrieval, UI rendering, direct source-file inspection, claims from model
  memory, and cross-session persistent memory.

## Public Interfaces

Commands:

```powershell
uv run -m uni_rag_agent answer --evidence-packet-id 1
uv run -m uni_rag_agent ask "Explain MapReduce from my courses" --model BAAI/bge-m3
```

Notebook:

```text
notebooks/answering_eda.ipynb
```

Internal interfaces:

```python
generate_answer(packet: EvidencePacket, conversation_context: list[dict] | None = None) -> AnswerResult
store_answer(evidence_packet_id: int, answer: AnswerResult) -> int
format_citation(evidence_item: EvidenceItem) -> str
validate_answer_citations(answer: AnswerResult, packet: EvidencePacket) -> CitationValidationResult
```

`conversation_context` remains in `generate_answer` for signature
compatibility, is validated/ignored, and never reaches the answer prompt or
storage. `AnswerSession` passes prior complete user/assistant turns only to the
planner and evicts the oldest complete turns when the positive
`UNI_RAG_ANSWER_SESSION_MESSAGE_LIMIT` bound is reached.

The model must return exactly one JSON object with exactly these fields:

```json
{"answer_paragraphs":[{"text":"nonblank prose without citation markers","citation_ids":["E1"]}],"limitations":[]}
```

Stable ids `E1`, `E2`, ... are assigned by packet evidence position (1-based).
Validation accepts only the unambiguous `chunk:<chunk_id>` compatibility alias
in addition to canonical ids; bare chunk ids and `E<chunk_id>` are invalid.
The application always canonicalizes aliases and renders/stores positional
markers. Every non-empty
packet paragraph must be nonblank and cite at least one known evidence item.
Structured stored citations contain `citation_id`, `evidence_index`, `course`,
`file_id`, `chunk_id`, `file_path`, `source_type`, and
`location_type`/`location_value`/`location_label`; only cited evidence appears.

Required rendered answer format:

```text
<answer paragraph with [E1] markers>

References:
- <course> - <file path> - <location>

Limitations:
- <weakness or insufficient-evidence note, when relevant>
```

## Configuration

`UNI_RAG_ANSWER_LLM_PROVIDER` and `UNI_RAG_ANSWER_LLM_MODEL` are nullable as a
pair during general validation and use the same provider allow-list as the
planner. They are mandatory only when non-empty evidence reaches answer
generation; planner settings are never a fallback. `UNI_RAG_ANSWER_MAX_RETRIES`
defaults to `1` and is nonnegative (`0` means no retry). The session message
limit defaults to `20` and is positive.
`UNI_RAG_ANSWER_PROMPT_MAX_TOKENS` defaults to `16,000`, is positive, and
bounds the complete whitespace-estimated answer-model input including query,
schema, metadata, weaknesses, constraints, rules, retry diagnostics, and
evidence. Stored `answers.model_name` is the safe `provider:model` value.

## Workflow

1. Load an evidence packet by id or receive one from `ask`.
2. For empty evidence, produce a deterministic useful insufficient-evidence
   answer from searched/found/missing coverage without invoking an answer model.
3. For non-empty evidence, select complete evidence items in packet/fused-rank
   order while the complete prompt fits the answer prompt budget. Omitted items
   do not renumber retained ids and produce a deterministic limitation. If no
   complete item fits, return and store a
   deterministic insufficient-budget answer without invoking the model.
   Otherwise prompt with selected evidence, allowed canonical ids plus
   `chunk:<id>` aliases, and constraints only; packet weaknesses are
   deduplicated into limitations.
4. Invoke the configured answer provider and validate strict JSON, paragraph
   rules, citation ids, and absence of markers in model prose.
5. Retry with the same evidence and validation errors only, up to the configured
   retry count. After exhaustion, return a deterministic safe refusal with no
   citations and a validation-failure limitation. Do not persist invalid model
   responses or prompts.
6. Provider construction/invocation failure creates no answer row. Validated,
   deterministic insufficient, and safe-refusal outcomes are append-only rows.
7. Before insertion, `store_answer` reloads the immutable packet and rejects
   mismatched evidence indexes/chunk ids, altered authoritative citation fields,
   inconsistent rendered markers/references, missing packet weaknesses,
   unqualified model identities, and forged deterministic refusals.
8. `ask` persists the Feature 09 packet before answer generation, so an answer
   failure leaves the search/evidence trace available.

## Failure and Safety Rules

- Never cite files absent from the packet, evidence omitted from the answer
  prompt, or answer from model memory.
- Model prose rejects Markdown-decorated `References:`/`Limitations:` sections,
  bracketed numeric or malformed `E...` citation lookalikes, and Markdown links.
- `ANSWER_ERROR=9` is reserved for answer provider/validation boundaries;
  existing configuration, storage, retrieval, and evidence domains remain.
- Telemetry contains counts, ids, provider/model labels, and statuses only; it
  never includes prompts, keys, or raw invalid output.
- The EDA notebook opens SQLite with `mode=ro` and `PRAGMA query_only=ON`,
  handles invalid JSON safely, and has cleared outputs/execution counts.

## Tests

- Inject deterministic chat doubles for cited answers, empty evidence, weak
  packet limitations, retries (including zero), unknown ids, safe refusal, and
  provider failure/no-row behavior.
- Verify canonical ids and `chunk:<id>` alias normalization, per-paragraph
  citation, Markdown-decorated section/marker/link rejection, stable references,
  append-only rows, separate `provider:model`, and packet persistence when
  `ask` answer generation fails.
- Cover a packet near the 12,000-token evidence limit, complete prompt overhead,
  a lower answer budget, stable original ids after whole-item omission, retry
  prompt bounds, and the deterministic no-evidence-fits outcome.
- Verify persistence rejects packet/citation/weakness/rendering/model mismatches
  and forged refusals without inserting an answer row.
- Cover CLI answer/ask JSON/text/error codes, config validation, bounded
  planner-only `AnswerSession`, sanitized failure telemetry, and read-only
  notebook validity.

## Acceptance Criteria

- `answer --evidence-packet-id ...` writes an evidence-grounded answer.
- `ask QUERY --model EMBEDDING_MODEL` runs planner/retrieval and separate answer
  models in one shot while preserving the packet on answer failure.
- Answers cite course, file, and location where available; insufficient
  evidence and safe refusals never invent facts or citations.
- Stored answer traces are auditable back to their exact packet and the
  deterministic prompt-budget selection applied to that packet.
