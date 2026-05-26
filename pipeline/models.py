from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

# ── chunks.json ─────────────────────────────────────────────────────────────


class Chunk(BaseModel):
    chunk_id: str
    document_name: str
    start_char: int
    end_char: int
    text: str


class ChunkList(BaseModel):
    chunks: list[Chunk]


# ── index_metadata.json ──────────────────────────────────────────────────────


class IndexMetadata(BaseModel):
    retrieval_mode: Literal["hybrid", "keyword", "embedding"]
    bm25_weight: float
    embedding_weight: float
    embedding_model: str
    chunk_count: int
    document_names: list[str]
    built_at: str  # ISO-8601
    reranker_model: Optional[str] = None
    reranker_enabled: bool = False


# ── retrieval_results.json ───────────────────────────────────────────────────


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_name: str
    rank: int
    retrieval_score: float


class QueryRetrieval(BaseModel):
    query_id: str
    question: str
    retrieved_chunks: list[RetrievedChunk]


class RetrievalResults(BaseModel):
    results: list[QueryRetrieval]


# ── draft_answers.json ───────────────────────────────────────────────────────


class DraftAnswer(BaseModel):
    query_id: str
    answer: str
    label: Literal["supported", "insufficient_support", "not_in_corpus"]
    citations: list[str]
    reasoning_summary: str


class DraftAnswers(BaseModel):
    answers: list[DraftAnswer]


# ── review_overrides.json ────────────────────────────────────────────────────


class ReviewOverride(BaseModel):
    query_id: str
    override_chunk_ids: list[str]


class ReviewOverrides(BaseModel):
    overrides: list[ReviewOverride]


# ── answer_audit.json ────────────────────────────────────────────────────────


class AuditResult(BaseModel):
    query_id: str
    audit_label: Literal["pass", "fail"]
    support_assessment: str
    citation_check: str
    hallucination_risk: Literal["low", "medium", "high"]
    recommended_fix: str
    final_context_chunk_ids: list[str]


class AnswerAudit(BaseModel):
    audits: list[AuditResult]


# ── llm_calls.jsonl ──────────────────────────────────────────────────────────


class LLMCallRecord(BaseModel):
    stage: str
    query_id: Optional[str]
    timestamp: str  # ISO-8601
    provider: str
    model: str
    prompt_hash: str
    input_artifacts: list[str]
    output_artifact: str


# ── retrieval_metrics.json ───────────────────────────────────────────────────


class QueryMetric(BaseModel):
    query_id: str
    expected_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    hit_at_k: int
    recall_at_k: float


class RetrievalMetrics(BaseModel):
    top_k: int
    overall_hit_at_k: float
    overall_recall_at_k: float
    per_query: list[QueryMetric]


# ── revised_answers.json ─────────────────────────────────────────────────────


class RevisedAnswer(BaseModel):
    query_id: str
    original_audit_label: Literal["pass", "fail"]
    original_hallucination_risk: Literal["low", "medium", "high"]
    answer: str
    label: Literal["supported", "insufficient_support", "not_in_corpus"]
    citations: list[str]
    reasoning_summary: str


class RevisedAnswers(BaseModel):
    answers: list[RevisedAnswer]


# ── retrieval_error_analysis.json ────────────────────────────────────────────


class RetrievalError(BaseModel):
    query_id: str
    failure_type: Literal["ranking", "chunking", "ambiguity", "corpus_gap", "none"]
    description: str


class RetrievalErrorAnalysis(BaseModel):
    analyses: list[RetrievalError]


# ── LLM response schemas (used by generator/auditor) ────────────────────────


class DraftAnswerResponse(BaseModel):
    answer: str
    label: Literal["supported", "insufficient_support", "not_in_corpus"]
    citations: list[str]
    reasoning_summary: str


class AuditResponse(BaseModel):
    audit_label: Literal["pass", "fail"]
    support_assessment: str
    citation_check: str
    hallucination_risk: Literal["low", "medium", "high"]
    recommended_fix: str


class RevisedAnswerResponse(BaseModel):
    answer: str
    label: Literal["supported", "insufficient_support", "not_in_corpus"]
    citations: list[str]
    reasoning_summary: str
