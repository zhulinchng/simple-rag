# Runbook — InsightBoard RAG Pipeline

Operational guide for configuring, running, and troubleshooting the pipeline.

---

## Switching LLM Providers

### OpenAI

```json
"llm": {
  "provider": "openai",
  "model": "gpt-4o-mini",
  "base_url": null,
  "temperature": 0.0,
  "max_tokens": 1024
}
```

Requires `OPENAI_API_KEY` in the environment:
```bash
export OPENAI_API_KEY=sk-...
```

### Local LLM (LM Studio or compatible server)

```json
"llm": {
  "provider": "local",
  "model": "qwen/qwen3-4b-2507",
  "base_url": "http://localhost:1234",
  "temperature": 0.0,
  "max_tokens": 1024
}
```

The pipeline posts to `{base_url}/api/v1/chat` with:
```json
{"model": "...", "system_prompt": "...", "input": "..."}
```

Smoke-test the endpoint before running the pipeline:
```bash
curl http://localhost:1234/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3-4b-2507",
    "system_prompt": "You answer only in rhymes.",
    "input": "What is your favorite color?"
  }'
```

---

## Changing the Document Corpus

1. Add or replace `.txt` files in `documents/`
2. Clean existing artifacts and rerun:
   ```bash
   make clean && make run
   ```

Chunk IDs are deterministic: `{filename_stem}_{start_char:06d}`. Adding a document changes chunk IDs for nothing else; renaming or modifying a file changes IDs for that document only.

---

## Tuning Retrieval

All retrieval parameters are in the `"retrieval"` block of `config.json`.

| Parameter | Effect |
|-----------|--------|
| `top_k` | Number of chunks returned per query |
| `chunk_size_chars` | Characters per chunk (smaller = more precise, more chunks) |
| `chunk_overlap_chars` | Overlap between consecutive chunks (reduces boundary misses) |
| `bm25_weight` | Weight for BM25 keyword signal (set to 0 for embedding-only) |
| `embedding_weight` | Weight for semantic embedding signal (set to 0 for BM25-only) |
| `embedding_model` | Sentence-transformer model for embeddings |

**Pure BM25:** `bm25_weight: 1.0, embedding_weight: 0.0`
**Pure embedding:** `bm25_weight: 0.0, embedding_weight: 1.0`
**Hybrid (default):** both > 0

---

## Enabling CrossEncoder Reranking

Set in `config.json`:
```json
"reranker_enabled": true,
"reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
"reranker_expansion_factor": 3
```

Reranking fetches `top_k * reranker_expansion_factor` candidates from hybrid retrieval, scores each `(query, chunk)` pair with the CrossEncoder, then returns the top `top_k`. Improves precision at the cost of latency.

---

## Human Review Checkpoint

The pipeline pauses after retrieval and before auditing to allow manual override of the top-k chunks for any query.

**Prompt format:**
```
Enter overrides (blank to skip): QUERY_ID CHUNK_ID1,CHUNK_ID2,...
```

**Example:**
```
Enter overrides: Q2 product_overview_000300,security_000000
```

Overrides are saved to `review_overrides.json` and applied during the audit stage. The `final_context_chunk_ids` in each audit record reflects the final state after overrides.

---

## Running the Evaluation Dataset

```bash
bash eval/run_eval.sh
```

This copies `eval/eval_queries.json` (4 queries with `expected_evidence` annotations) as `queries.json`, runs the full pipeline, and prints `retrieval_metrics.json` at the end.

**Reading metrics:**
```json
{
  "top_k": 3,
  "overall_hit_at_k": 1.0,
  "overall_recall_at_k": 1.0,
  "per_query": [...]
}
```

- `hit_at_k`: 1 if at least one expected chunk was retrieved, 0 otherwise
- `recall_at_k`: fraction of expected chunks that were retrieved

---

## Artifact Reference

| File | Stage produced | Description |
|------|---------------|-------------|
| `chunks.json` | DOCUMENTS_CHUNKED | All chunks with `chunk_id`, `start_char`, `end_char`, `text` |
| `index_metadata.json` | INDEX_BUILT | Config snapshot: mode, weights, model, chunk count |
| `retrieval_results.json` | RETRIEVAL_COMPLETE | Top-k `RetrievedChunk` per query with scores and ranks |
| `draft_answers.json` | DRAFT_ANSWERS_GENERATED | Answer, label, citations, reasoning per query |
| `review_overrides.json` | HUMAN_REVIEW_COMPLETE | Reviewer-specified chunk ID replacements |
| `answer_audit.json` | ANSWERS_AUDITED | Grounding check, hallucination risk, final context IDs |
| `revised_answers.json` | (stretch) | Re-generated answers for fail/high-risk audits |
| `retrieval_metrics.json` | (stretch) | hit@k and recall@k, or `{"skipped": true}` |
| `retrieval_error_analysis.json` | (stretch) | Per-query failure classification |
| `final_report.md` | FINAL_REPORT_GENERATED | Human-readable 6-section summary |
| `llm_calls.jsonl` | Continuous | One JSON line per LLM call: stage, query_id, prompt hash, timestamps |

---

## Troubleshooting

### `OPENAI_API_KEY` not set
```
openai.AuthenticationError: No API key provided.
```
Fix: `export OPENAI_API_KEY=sk-...`

### Local LLM connection refused
```
requests.exceptions.ConnectionError: ... Connection refused
```
Fix: Ensure your local server is running on the configured `base_url`. Smoke-test with `curl` (see above).

### Local LLM returns invalid JSON
The local provider injects the Pydantic schema into the system prompt and attempts to parse the response. If parsing fails with `ValidationError`, the model may be wrapping the JSON in markdown code fences or adding extra text. Try:
- Lowering `temperature` to `0.0`
- Using a larger / instruction-tuned model
- Adding explicit JSON instructions to your prompt

### Validation failures after a run
```bash
make validate
```
Common causes:
- `policy.json exists` — pipeline loaded `config.json`, but `validate.py` also checks `policy.json` is present on disk. Both files are kept.
- `citation not in retrieved` — LLM cited a chunk ID that was not in the top-k. Citations are automatically filtered in `generator.py`; this check enforces the filtered list.
- `Only N bytes` for `final_report.md` — report must exceed 100 bytes. A normal run always exceeds this; if not, check `reporter.py`.

### Re-running from a specific stage
The pipeline does not support partial replay. Run `make clean` to delete all artifacts, then `make run` to start from scratch.
