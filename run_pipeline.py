#!/usr/bin/env python3
"""
Main RAG pipeline orchestrator.

Stages:
  INIT -> INPUTS_LOADED -> DOCUMENTS_CHUNKED -> INDEX_BUILT ->
  RETRIEVAL_COMPLETE -> DRAFT_ANSWERS_GENERATED -> HUMAN_REVIEW_COMPLETE ->
  ANSWERS_AUDITED -> FINAL_REPORT_GENERATED -> VALIDATION_COMPLETE -> RESULTS_FINALISED
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pipeline.auditor import audit_answers
from pipeline.chunker import load_and_chunk
from pipeline.generator import generate_draft_answers
from pipeline.logger import clear_log
from pipeline.models import (
    AnswerAudit,
    ChunkList,
    DraftAnswers,
    IndexMetadata,
    RetrievalResults,
    ReviewOverrides,
)
from pipeline.reporter import generate_report
from pipeline.retriever import HybridRetriever
from pipeline.reviewer import run_human_review
from pipeline.state import PipelineStage, PipelineStateMachine
from pipeline.stretch import (
    analyze_retrieval_errors,
    compute_retrieval_metrics,
    generate_revised_answers,
)

DOCUMENTS_DIR = Path("documents")
QUERIES_FILE = Path("queries.json")
POLICY_FILE = Path("policy.json")

CHUNKS_FILE = Path("chunks.json")
INDEX_META_FILE = Path("index_metadata.json")
RETRIEVAL_FILE = Path("retrieval_results.json")
DRAFT_FILE = Path("draft_answers.json")
OVERRIDES_FILE = Path("review_overrides.json")
AUDIT_FILE = Path("answer_audit.json")
REPORT_FILE = Path("final_report.md")
METRICS_FILE = Path("retrieval_metrics.json")
REVISED_FILE = Path("revised_answers.json")
ERROR_ANALYSIS_FILE = Path("retrieval_error_analysis.json")


def _save_json(data, path: Path) -> None:
    path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
    print(f"[SAVED] {path}")


def main() -> None:
    sm = PipelineStateMachine()
    clear_log()

    # ── INIT → INPUTS_LOADED ─────────────────────────────────────────────────
    if not DOCUMENTS_DIR.exists():
        sys.exit(f"ERROR: {DOCUMENTS_DIR}/ not found")
    if not QUERIES_FILE.exists():
        sys.exit(f"ERROR: {QUERIES_FILE} not found")
    if not POLICY_FILE.exists():
        sys.exit(f"ERROR: {POLICY_FILE} not found")

    config_path = Path("config.json") if Path("config.json").exists() else POLICY_FILE
    queries_data = json.loads(QUERIES_FILE.read_text())
    policy = json.loads(config_path.read_text())
    queries: list[dict] = queries_data["queries"]
    retrieval_cfg = policy["retrieval"]
    answer_policy = policy["answer_policy"]
    llm_cfg = policy.get("llm", {"provider": "openai", "model": "gpt-4o-mini"})

    sm.advance(PipelineStage.INPUTS_LOADED)
    print(
        f"  Loaded {len(queries)} queries, {len(list(DOCUMENTS_DIR.glob('*.txt')))} documents"
    )

    # ── INPUTS_LOADED → DOCUMENTS_CHUNKED ────────────────────────────────────
    chunk_list: ChunkList = load_and_chunk(
        DOCUMENTS_DIR,
        chunk_size=retrieval_cfg["chunk_size_chars"],
        overlap=retrieval_cfg["chunk_overlap_chars"],
    )
    _save_json(chunk_list, CHUNKS_FILE)
    sm.advance(PipelineStage.DOCUMENTS_CHUNKED)
    print(f"  {len(chunk_list.chunks)} chunks created")

    # ── DOCUMENTS_CHUNKED → INDEX_BUILT ──────────────────────────────────────
    retriever = HybridRetriever(
        chunk_list=chunk_list,
        bm25_weight=retrieval_cfg.get("bm25_weight", 0.5),
        embedding_weight=retrieval_cfg.get("embedding_weight", 0.5),
        embedding_model=retrieval_cfg.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        top_k=retrieval_cfg["top_k"],
        reranker_model=retrieval_cfg.get(
            "reranker_model", "cross-encoder/ms-marco-MiniLM-L6-v2"
        ),
        reranker_enabled=retrieval_cfg.get("reranker_enabled", False),
        reranker_expansion_factor=retrieval_cfg.get("reranker_expansion_factor", 3),
    )
    index_metadata: IndexMetadata = retriever.build_metadata()
    _save_json(index_metadata, INDEX_META_FILE)
    sm.advance(PipelineStage.INDEX_BUILT)

    # ── INDEX_BUILT → RETRIEVAL_COMPLETE ─────────────────────────────────────
    retrieval_results: RetrievalResults = retriever.retrieve_all(queries)
    _save_json(retrieval_results, RETRIEVAL_FILE)
    sm.advance(PipelineStage.RETRIEVAL_COMPLETE)

    # ── RETRIEVAL_COMPLETE → DRAFT_ANSWERS_GENERATED ──────────────────────────
    print("\n[LLM] Generating draft answers (Stage 1) ...")
    draft_answers: DraftAnswers = generate_draft_answers(
        retrieval_results=retrieval_results,
        chunk_list=chunk_list,
        answer_policy=answer_policy,
        llm_cfg=llm_cfg,
    )
    _save_json(draft_answers, DRAFT_FILE)
    sm.advance(PipelineStage.DRAFT_ANSWERS_GENERATED)

    # ── DRAFT_ANSWERS_GENERATED → HUMAN_REVIEW_COMPLETE ──────────────────────
    review_overrides: ReviewOverrides = run_human_review(
        retrieval_results=retrieval_results,
        draft_answers=draft_answers,
        chunk_list=chunk_list,
    )
    _save_json(review_overrides, OVERRIDES_FILE)
    sm.advance(PipelineStage.HUMAN_REVIEW_COMPLETE)

    # ── HUMAN_REVIEW_COMPLETE → ANSWERS_AUDITED ───────────────────────────────
    print("\n[LLM] Auditing answers (Stage 2) ...")
    answer_audit: AnswerAudit = audit_answers(
        draft_answers=draft_answers,
        retrieval_results=retrieval_results,
        review_overrides=review_overrides,
        chunk_list=chunk_list,
        answer_policy=answer_policy,
        llm_cfg=llm_cfg,
    )
    _save_json(answer_audit, AUDIT_FILE)
    sm.advance(PipelineStage.ANSWERS_AUDITED)

    # ── SHOULD: Revised answers for failed/high-risk queries ─────────────────
    print("\n[LLM] Checking for answers requiring revision ...")
    needs_revision = [
        a
        for a in answer_audit.audits
        if a.audit_label == "fail" or a.hallucination_risk == "high"
    ]
    if needs_revision:
        print(f"  {len(needs_revision)} answer(s) flagged for revision.")
        revised = generate_revised_answers(
            answer_audit=answer_audit,
            draft_answers=draft_answers,
            retrieval_results=retrieval_results,
            review_overrides=review_overrides,
            chunk_list=chunk_list,
            answer_policy=answer_policy,
            llm_cfg=llm_cfg,
        )
        _save_json(revised, REVISED_FILE)
    else:
        print("  No revisions needed.")
        from pipeline.models import RevisedAnswers

        _save_json(RevisedAnswers(answers=[]), REVISED_FILE)

    # ── SHOULD: Retrieval metrics (skip if no annotations) ───────────────────
    metrics = compute_retrieval_metrics(
        queries, retrieval_results, retrieval_cfg["top_k"]
    )
    if metrics:
        _save_json(metrics, METRICS_FILE)
        print(
            f"[METRICS] hit@{retrieval_cfg['top_k']}={metrics.overall_hit_at_k:.2f} recall@{retrieval_cfg['top_k']}={metrics.overall_recall_at_k:.2f}"
        )
    else:
        print(
            "[METRICS] No expected_evidence annotations — skipping retrieval metrics."
        )
        METRICS_FILE.write_text(
            '{"skipped": true, "reason": "no expected_evidence annotations in queries.json"}'
        )

    # ── STRETCH: Retrieval error analysis ─────────────────────────────────────
    error_analysis = analyze_retrieval_errors(
        answer_audit=answer_audit,
        draft_answers=draft_answers,
        retrieval_results=retrieval_results,
        chunk_list=chunk_list,
    )
    _save_json(error_analysis, ERROR_ANALYSIS_FILE)

    # ── ANSWERS_AUDITED → FINAL_REPORT_GENERATED ──────────────────────────────
    generate_report(
        retrieval_results=retrieval_results,
        draft_answers=draft_answers,
        review_overrides=review_overrides,
        answer_audit=answer_audit,
        index_metadata=index_metadata,
        output_path=REPORT_FILE,
    )
    sm.advance(PipelineStage.FINAL_REPORT_GENERATED)

    # ── FINAL_REPORT_GENERATED → VALIDATION_COMPLETE ─────────────────────────
    import subprocess

    result = subprocess.run([sys.executable, "validate.py"], capture_output=False)
    if result.returncode == 0:
        sm.advance(PipelineStage.VALIDATION_COMPLETE)
    else:
        print("[WARN] Validation reported issues — check output above.")
        sm.advance(PipelineStage.VALIDATION_COMPLETE)

    # ── VALIDATION_COMPLETE → RESULTS_FINALISED ───────────────────────────────
    sm.advance(PipelineStage.RESULTS_FINALISED)
    print("\n[DONE] Pipeline complete. All artifacts written to disk.")


if __name__ == "__main__":
    main()
