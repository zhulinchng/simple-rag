# InsightBoard RAG Pipeline

A deterministic, replayable Retrieval-Augmented Generation (RAG) pipeline that ingests a document corpus and a query set, performs hybrid retrieval, generates grounded answers with citations, runs a second-stage audit, and produces a structured evaluation report.

## Features

- **Hybrid retrieval** — BayesianBM25 + sentence-transformer embeddings with configurable weights
- **Optional CrossEncoder reranking** — post-retrieval reranking via `cross-encoder/ms-marco-MiniLM-L6-v2`
- **Two-stage LLM architecture** — Stage 1: draft generation, Stage 2: hallucination audit
- **Human review checkpoint** — interactive mid-pipeline override of retrieval results
- **Multi-provider LLM support** — OpenAI or any local endpoint (e.g. LM Studio)
- **Deterministic artifacts** — every pipeline run writes versioned JSON/JSONL/Markdown files
- **Retrieval metrics** — hit@k and recall@k when queries have `expected_evidence` annotations

## Prerequisites

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- `OPENAI_API_KEY` in environment (when using the OpenAI provider)

## Quickstart

```bash
# 1. Install dependencies
uv pip install -r requirements.txt

# 2. Configure (see config.json)
#    Default: OpenAI gpt-4o-mini, hybrid BM25+embedding retrieval

# 3. Run
make run

# 4. Validate all output artifacts
make validate

# 5. Run the test suite
make test
```

## Configuration

All pipeline parameters live in **`config.json`**:

```json
{
  "retrieval": {
    "top_k": 3,
    "chunk_size_chars": 350,
    "chunk_overlap_chars": 50,
    "bm25_weight": 0.5,
    "embedding_weight": 0.5,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "reranker_enabled": false,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "reranker_expansion_factor": 3
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "base_url": null,
    "temperature": 0.0,
    "max_tokens": 1024
  },
  "answer_policy": { ... }
}
```

### Switching to a local LLM

Change three fields in `config.json`:

```json
"llm": {
  "provider": "local",
  "model": "qwen/qwen3-4b-2507",
  "base_url": "http://localhost:1234"
}
```

The local provider posts to `{base_url}/api/v1/chat` with `{model, system_prompt, input}`.

## Outputs

| File | Contents |
|------|----------|
| `chunks.json` | Tokenised document chunks |
| `index_metadata.json` | Retrieval configuration snapshot |
| `retrieval_results.json` | Top-k chunks per query |
| `draft_answers.json` | LLM-generated answers with citations |
| `review_overrides.json` | Human reviewer changes to retrieval |
| `answer_audit.json` | Second-stage audit results |
| `final_report.md` | Human-readable summary report |
| `llm_calls.jsonl` | Full LLM call log with prompt hashes |
| `retrieval_metrics.json` | hit@k / recall@k (if annotations present) |

## Project Layout

```
documents/          Corpus .txt files
queries.json        Queries (add expected_evidence for eval)
config.json         All pipeline parameters
pipeline/           Pipeline package
  llm_client.py     Provider abstraction (openai / local)
  retriever.py      BayesianBM25 + embedding hybrid retriever
  generator.py      Stage 1: draft answer generation
  auditor.py        Stage 2: hallucination audit
  stretch.py        Revised answers, retrieval metrics, error analysis
tests/              pytest suite (27 tests, no API key required)
eval/               Annotated eval dataset + run_eval.sh
```

## Evaluation Dataset

```bash
bash eval/run_eval.sh
```

Copies `eval/eval_queries.json` (with `expected_evidence` annotations) as `queries.json`, runs a full pipeline, and prints `retrieval_metrics.json`.
