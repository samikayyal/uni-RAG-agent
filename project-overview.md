# Uni RAG Agent Project Overview

## Goal

Build a local course-archive assistant for `D:\Projects\Uni RAG Agent\Courses`.

The user recently graduated and wants an agentic RAG system that can answer questions using their university course files. The system should be pragmatic, source-grounded, and auditable. It should not behave like a generic chatbot over a vector database.

The intended product is better described as a **course archive intelligence system**:

- answer questions from course materials;
- identify relevant courses and source files;
- inspect code, notebooks, and data summaries when useful;
- report what was searched, what was found, and what was missing;
- cite exact files and locations whenever possible.

## Development Constraint

This is a Python-oriented project. Use `uv` for Python dependency and run workflows:

```powershell
uv add package_name
uv run -m uni_rag_agent ...
```

Do not document or use non-`uv` package installation or direct interpreter commands for normal project work.

## Current Data Profile

The `Courses` folder was profiled with PowerShell commands. Approximate totals:

- Files: `27,978`
- Size: `24.4 GB`

Top file extensions:

```text
18,633 .png
 7,840 .jpg
   435 .pdf
   210 .ipynb
   112 .docx
    92 .jpeg
    91 .pptx
    65 .txt
    57 .py
    56 .mp4
    47 .tif
    46 .csv
    45 .r
    21 .cpp
    16 .pfl
    15 .ppt
    15 .doc
    14 .xpt
    11 .xlsx
    11 .ini
    10 .mst
     9 .mlx
     8 .zip
     7 .rhistory
     7 .json
     5 .joblib
     5 .m
     5 .m4a
     5 .md
     4 .h
     4 .rdata
     4 .toml
     3 .wav
     3 .vtt
     3 .db
     3 .drawio
     3 no-extension files
```

Other observed extensions include archives, binaries, installers, models, and data artifacts:

```text
.7z, .rar, .bin, .weights, .pt, .pkl, .tflite, .exe, .msi, .cab,
.sqlite, .sas7bdat, .sas, .rds, .ttl, .jsonl, .mov, .mkv, .avi
```

Largest top-level course folders by size:

```text
NLP                                      5.19 GB,     49 files
Graduation Project                       3.54 GB, 26,427 files
Algorithms Design and Analysis           3.31 GB,     86 files
Operating Systems                        2.01 GB,     73 files
Database                                 1.97 GB,    126 files
Computer and Society                     1.90 GB,     54 files
Object Oriented Programming              1.41 GB,     48 files
Pattern Recognition                      0.90 GB,    119 files
Data Structures and Intro to Algorithms  0.64 GB,     43 files
Entrepreneurship                         0.32 GB,     11 files
```

Largest observed files include:

```text
NLP\Assignments\GoogleNews-vectors-negative300.bin                         3.39 GB
Graduation Project\Body Comp\models\checkpoints\rf\model_rf.joblib         2.39 GB
Database\Lab\Lab 0\OracleXE213_Win64\DB.cab                                1.82 GB
NLP\Assignments\GoogleNews-vectors-negative300.bin.zip                     1.64 GB
Object Oriented Programming\...\Recordings .h & .cpp files.7z              1.28 GB
Several .mp4/.mov lecture or project videos                                0.2-0.6 GB each
Pattern Recognition\Assignments\Ass 6\yolov3.weights                       0.23 GB
```

Courses with the most text-like files, based on an extension filter for documents, slides, notebooks, code, and structured data:

```text
Database                                  96
Data Eng                                  82
Stats for Data Science                    75
Artificial Intelligence                   62
Algorithms Design and Analysis            56
Pattern Recognition                       56
Graduation Project                        55
Operating Systems                         53
Machine Learning                          40
Data Visualization                        38
Calculus 2                                37
Data Structures and Intro to Algorithms   35
Object Oriented Programming               34
Data Mining                               30
NLP                                       27
Special Topics 1 (Knowledge Graphs)       25
High Preformance Computing for Big Data   25
Linear Algebra                            24
ISS                                       23
Intro to DS                               23
```

## Important Product Decisions

### Ignore Images for RAG

The user confirmed that almost all images are data and not useful course knowledge.

Image files should be:

- kept in the metadata inventory;
- excluded from semantic/text RAG;
- not OCR'd by default;
- searchable only as file/folder metadata.

This applies primarily to:

```text
.png, .jpg, .jpeg, .tif, .jfif
```

The system may still answer metadata questions such as:

- which course folders contain image datasets;
- where image-heavy folders are located;
- which projects used image data.

### Do Not Embed Everything

A naive "embed the whole Courses folder" approach is explicitly rejected.

Reasons:

- most files by count are images and likely datasets;
- several huge files are model artifacts, installers, archives, or binaries;
- blindly indexing everything would pollute retrieval;
- cost, runtime, and storage would be wasted;
- citations and provenance would be harder to trust.

### Keep Binaries and Heavy Artifacts as Metadata Only

These should not be loaded, embedded, or executed by default:

```text
.bin, .joblib, .cab, .weights, .tflite, .pt, .pkl,
.exe, .msi, .zip, .rar, .7z
```

Treat pickle/joblib/model artifacts as untrusted unless the user explicitly asks to inspect them and accepts the risk.

### Video and Audio Are Deferred

Videos/audio should initially be metadata only. Existing `.vtt` transcripts can be indexed.

Do not transcribe all video/audio by default. Transcription should be an opt-in tool later.

## Intended Workflow

The user's proposed high-level workflow:

1. User asks a query.
2. An agent decides which courses and indexes are relevant for RAG search.
3. If retrieval is weak, the system says exactly what was searched and what was missing.
4. The agent has tools such as `read_file()`, `python_repl()`, `keyword_search()`, and others.
5. A second answer step writes based on the gathered information.

Recommended framing:

```text
Researcher / retrieval agent builds a structured evidence packet.
Answer agent writes only from that packet.
```

This is not mainly about giving the second agent "fresh context." The value is:

- separation of concerns;
- auditable retrieval;
- source-grounded answering;
- explicit weak-retrieval reporting;
- less hallucination;
- easier debugging.

## Recommended Architecture

```text
User Query
   |
   v
Query Router
   - classify query type
   - identify candidate courses
   - identify candidate indexes
   - decide whether keyword/file/code/data tools are needed
   |
   v
Retrieval / Research Orchestrator
   - semantic search
   - keyword search
   - metadata search
   - read extracted chunks
   - inspect source files
   - inspect notebooks
   - summarize data schemas
   - optionally use Python for safe analysis
   |
   v
Evidence Packet
   - interpreted intent
   - searched courses
   - searched indexes
   - keyword terms
   - semantic queries
   - evidence chunks
   - source citations
   - weaknesses / missing coverage
   |
   v
Answer Generator
   - answer only from evidence packet
   - cite files and locations
   - distinguish direct evidence from inference
   - report limitations when retrieval is weak
```

## Evidence Packet Contract

The evidence packet is the key interface between retrieval and answering.

It should be structured, not a vague summary.

Example shape:

```json
{
  "query": "Explain MapReduce from my courses",
  "interpreted_intent": "concept_explanation",
  "searched": {
    "courses": [
      "Distributed Systems",
      "High Preformance Computing for Big Data",
      "Data Eng"
    ],
    "indexes": [
      "slides",
      "documents",
      "notebooks",
      "code"
    ],
    "keyword_terms": [
      "mapreduce",
      "map reduce",
      "hadoop"
    ],
    "semantic_queries": [
      "MapReduce programming model",
      "Hadoop distributed computation",
      "map and reduce phases"
    ]
  },
  "evidence": [
    {
      "course": "High Preformance Computing for Big Data",
      "file": "path/to/slides.pptx",
      "source_type": "pptx",
      "location": "slide 14",
      "text": "relevant extracted chunk",
      "score": 0.82
    }
  ],
  "weaknesses": [
    "No transcript search was performed because videos are not indexed.",
    "No exact keyword match was found in Distributed Systems."
  ],
  "answer_constraints": [
    "Answer only from evidence.",
    "Cite course and file.",
    "If evidence is insufficient, say so."
  ]
}
```

## Query Types to Support

Start with explicit query types. Each type should route to different indexes and tools.

```text
concept_explanation
course_summary
cross_course_comparison
find_file
assignment_or_project_lookup
code_question
data_question
study_quiz
portfolio_resume
unknown_or_unsupported
```

Expected behavior:

- `concept_explanation`: search slides, documents, notebooks; answer with citations.
- `course_summary`: summarize course-level extracted metadata and high-quality files.
- `cross_course_comparison`: search multiple candidate courses and compare evidence.
- `find_file`: prioritize metadata, file names, folder names, and keywords over embeddings.
- `assignment_or_project_lookup`: search project/assignment folders, notebooks, reports, and code.
- `code_question`: search code and notebooks, then inspect top files.
- `data_question`: inspect schema summaries, file metadata, and small safe samples.
- `study_quiz`: generate questions only from cited materials.
- `portfolio_resume`: produce evidence-backed bullets from projects, notebooks, code, reports.
- `unknown_or_unsupported`: say what was searched and what was unavailable.

## Indexes

Use separate indexes rather than one mixed index.

### Metadata Index

Inventory every file, including skipped files.

Fields:

```text
path
course
extension
size
modified time
guessed category
is_indexed
reason_not_indexed
```

### Document Index

For:

```text
.pdf, .docx, .doc, .txt, .md
```

Extract text, split into chunks, preserve page or section information where possible.

### Slides Index

For:

```text
.pptx, .ppt
```

Extract slide text and notes if possible. Preserve slide numbers.

### Notebook Index

For:

```text
.ipynb
```

Index markdown cells and code cells separately. Preserve notebook path and cell number.

### Code Index

For:

```text
.py, .r, .cpp, .h, .m
```

Index functions, comments, imports, and surrounding code context.

### Data Schema Index

For:

```text
.csv, .xlsx, .json, .jsonl, .sqlite, .db
```

Do not embed entire datasets by default. Store summaries:

```text
row count if cheap
column names
column types
small sample
sheet names
table names
file path
course
```

### Transcript Index

For:

```text
.vtt, existing text transcripts
```

Preserve timestamps if available.

## Search Strategy

Use hybrid retrieval:

- semantic/vector search;
- BM25 or keyword search;
- metadata filters;
- course-name matching;
- file-name matching;
- heading/title matching.

Do not rely only on an LLM router. Combine:

- course folder names;
- file names;
- extracted headings;
- keyword hits;
- embeddings over course summaries and chunks;
- LLM classification.

Recommended retrieval flow:

```text
1. Select candidate courses.
2. Select candidate indexes.
3. Search candidate files.
4. Retrieve chunks within candidate files.
5. Merge keyword and semantic results with Reciprocal Rank Fusion (RRF).
6. Read exact source chunks/files for top evidence.
7. Build evidence packet.
```

Reranking is not part of the MVP. Add a reranker later only if evaluation shows RRF is not good enough.

## Tools

Initial tool candidates:

```python
list_courses()
search_metadata(query, filters=None)
keyword_search(query, course=None, extensions=None, top_k=20)
semantic_search(query, course=None, index=None, top_k=20)
read_file(path, max_chars=None)
read_extracted_chunk(chunk_id)
inspect_notebook(path)
summarize_csv(path)
summarize_xlsx(path)
summarize_sqlite(path)
python_repl(code)
explain_search_coverage(search_run_id)
```

`python_repl()` should be constrained.

Acceptable uses:

- counting files;
- parsing notebooks;
- inspecting CSV/XLSX schemas;
- summarizing safe data samples;
- testing small isolated snippets.

Avoid automatic use for:

- running old course scripts;
- executing notebooks;
- loading pickle/joblib files;
- installing dependencies;
- opening random large databases;
- executing archived project code.

## Weak Retrieval Reporting

When retrieval is weak, the answer should explicitly say:

```text
I searched:
- Courses: ...
- Indexes: ...
- Keywords: ...
- Semantic queries: ...

I found:
- ...

Missing or not searched:
- ...
```

Examples of useful limitations:

- videos were not searched because they have no indexed transcript;
- images are metadata-only and intentionally skipped;
- no exact keyword hits were found;
- only code was found, not lecture material;
- a course was selected by semantic similarity but had weak evidence.

## Answering Rules

The final answer step must:

- use only the evidence packet;
- cite course and file;
- cite page, slide, cell, row, or timestamp when available;
- separate direct evidence from inference;
- say when evidence is insufficient;
- never invent course coverage;
- never cite a file that is not present in the evidence packet.

## MVP Scope

Build first:

1. File inventory into SQLite.
2. File classification: indexed vs skipped with reason.
3. Text extraction for PDFs, PPTX, DOCX, TXT, MD, notebooks, code, and existing VTT.
4. Separate indexes by source type.
5. Keyword search over extracted text.
6. Vector search over ChromaDB collections by logical index.
7. Simple query router.
8. Evidence packet generation.
9. Final source-grounded answer generation.
10. FastAPI answer API with a simple HTML/JS frontend and CLI commands for ingestion/index/eval operations.

Defer:

- OCR;
- image captioning;
- full video transcription;
- knowledge graph;
- automatic old-code execution;
- unrestricted Python execution;
- loading model/pickle/joblib artifacts;
- fancy UI.

## Accepted Implementation Stack

Use this stack for MVP implementation:

```text
Python + uv
SQLite for metadata
SQLite FTS5 for keyword search
ChromaDB for vector storage
LangChain for LLM, embedding, retriever, tool, and memory abstractions
FastAPI with plain HTML/JS for the answer UI
CLI commands through uv run -m uni_rag_agent for ingestion, indexing, smoke checks, and evaluation
PyMuPDF for PDFs
python-pptx for PPTX
python-docx for DOCX
nbformat for notebooks
pandas/openpyxl for CSV/XLSX summaries
```

The LLM and embedding provider/model are configuration values, not hardcoded project choices. Tests should use deterministic fake adapters so they do not require API keys.

The `context/` folder and `context/feature-specs/` are the implementation source of truth. This root overview remains the fuller narrative summary.

## Design Position

The system should be boring, strict, and inspectable:

```text
Ignore images.
Index text-like files.
Keep binaries/media as metadata.
Use hybrid search.
Let the researcher inspect files.
Produce a structured evidence packet.
Answer only from evidence.
Report searched/found/missing when retrieval is weak.
```

The "agentic" part should be about routing, tool use, validation, and coverage reporting, not about free-form autonomous behavior.
