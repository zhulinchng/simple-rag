# RAG Pipeline

## Architecture

This is a **staged RAG pipeline** with a strict 11-step state machine that enforces execution order. All intermediate outputs are persisted as JSON artifacts to the working directory.

### Stage flow

```
INIT → INPUTS_LOADED → DOCUMENTS_CHUNKED → INDEX_BUILT → RETRIEVAL_COMPLETE
     → DRAFT_ANSWERS_GENERATED → HUMAN_REVIEW_COMPLETE → ANSWERS_AUDITED
     → FINAL_REPORT_GENERATED → VALIDATION_COMPLETE → RESULTS_FINALISED
```

`pipeline/state.py` — `PipelineStateMachine.advance()` enforces strictly ordered transitions and raises `InvalidTransition` if a stage is skipped. `run_pipeline.py` is the orchestrator that wires all stages together.

### Inputs (read from disk, replaceable by evaluator)

| File | Purpose |
|------|---------|
| `documents/*.txt` | Corpus — read in sorted order for determinism |
| `queries.json` | `{"queries": [{"query_id", "question"}]}` |
| `policy.json` | Controls chunking params, retrieval weights, LLM model, answer policy |

### Key pipeline modules

**`pipeline/chunker.py`** — deterministic sliding-window character chunker. Chunk IDs are `{doc_stem}_{start_char:06d}`. No LLM calls. Called before any LLM stage; `validate.py` checks mtime ordering.

**`pipeline/retriever.py`** — `HybridRetriever` combines BM25 (`rank-bm25`) and sentence-transformer cosine similarity. Both score arrays are min-max normalized to [0,1], then blended: `final = bm25_weight * bm25_norm + embedding_weight * emb_norm`. Ties broken lexicographically by `chunk_id`. Setting `bm25_weight=1.0, embedding_weight=0.0` in `policy.json` gives pure BM25 (no model download). `index_metadata.json` records the mode (`hybrid`/`keyword`/`embedding`).

**`pipeline/generator.py`** — one `client.beta.chat.completions.parse()` call per query (OpenAI structured output with Pydantic `DraftAnswerResponse`). Citations are validated post-call to ensure they are a subset of the retrieved chunk IDs.

**`pipeline/auditor.py`** — one structured LLM call per query, never batched. Reads `review_overrides.json` to determine final context: overridden queries use the override chunk IDs; others use the original top-k. Stores `final_context_chunk_ids` in every audit record.

**`pipeline/reviewer.py`** — interactive terminal checkpoint between generation and audit. Accepts `query_id chunk_id1,chunk_id2,...` lines; validates chunk IDs against `chunks.json`; writes `review_overrides.json`.

**`pipeline/stretch.py`** — three optional features: revised answers for `audit_label=fail` or `hallucination_risk=high` queries; retrieval metrics (`hit@k`, `recall@k`) if `queries.json` contains `expected_evidence` fields; retrieval error analysis that classifies failures as `corpus_gap`, `ranking`, `chunking`, or `ambiguity`.

**`pipeline/models.py`** — single source of truth for all Pydantic schemas (artifacts + LLM response types). Every artifact is a `BaseModel` with `.model_dump_json(indent=2)`.

**`pipeline/logger.py`** — appends one JSON line to `llm_calls.jsonl` per LLM call, including `stage`, `query_id`, ISO-8601 timestamp, `prompt_hash` (SHA-256 prefix), and `input_artifacts`/`output_artifact` paths.

### Output artifacts

Required: `chunks.json`, `index_metadata.json`, `retrieval_results.json`, `draft_answers.json`, `review_overrides.json`, `answer_audit.json`, `final_report.md`, `llm_calls.jsonl`

Optional (written unconditionally as empty if unused): `retrieval_metrics.json`, `revised_answers.json`, `retrieval_error_analysis.json`

### Validation (`validate.py`)

12 checks covering: artifact existence, JSON validity, mtime ordering (chunks before LLM calls), unique chunk IDs, retrieval completeness, allowed labels, citation subset correctness, per-query LLM call records in `llm_calls.jsonl`, audit timestamps after draft timestamps, override chunk IDs reflected in `final_context_chunk_ids`, and report non-emptiness. Exit 0 = pass, 1 = fail.

## Retrieval modes

Controlled by `policy.json` `retrieval` block:

```json
"bm25_weight": 0.5,
"embedding_weight": 0.5,
"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
```

Pure BM25 (offline, no model download): set `bm25_weight=1.0, embedding_weight=0.0`.
