"""Stretch and SHOULD features: revised answers, retrieval metrics, error analysis."""

from __future__ import annotations

from .llm_client import call_llm
from .logger import log_llm_call
from .models import (
    AnswerAudit,
    Chunk,
    ChunkList,
    DraftAnswers,
    QueryMetric,
    RetrievalError,
    RetrievalErrorAnalysis,
    RetrievalMetrics,
    RetrievalResults,
    ReviewOverrides,
    RevisedAnswer,
    RevisedAnswerResponse,
    RevisedAnswers,
)

# ── Revised Answers ───────────────────────────────────────────────────────────


def _format_chunks(chunks: list[Chunk]) -> str:
    parts = [f"[{c.chunk_id}] ({c.document_name})\n{c.text}" for c in chunks]
    return "\n\n---\n\n".join(parts)


def generate_revised_answers(
    answer_audit: AnswerAudit,
    draft_answers: DraftAnswers,
    retrieval_results: RetrievalResults,
    review_overrides: ReviewOverrides,
    chunk_list: ChunkList,
    answer_policy: dict,
    llm_cfg: dict,
) -> RevisedAnswers:
    chunk_map = {c.chunk_id: c for c in chunk_list.chunks}

    override_map = {
        o.query_id: o.override_chunk_ids for o in review_overrides.overrides
    }
    question_map = {r.query_id: r.question for r in retrieval_results.results}
    retrieval_map = {r.query_id: r for r in retrieval_results.results}
    draft_map = {d.query_id: d for d in draft_answers.answers}

    revised: list[RevisedAnswer] = []

    for audit in answer_audit.audits:
        if audit.audit_label != "fail" and audit.hallucination_risk != "high":
            continue

        qid = audit.query_id
        question = question_map[qid]

        if qid in override_map:
            final_ids = override_map[qid]
        else:
            final_ids = [rc.chunk_id for rc in retrieval_map[qid].retrieved_chunks]

        final_chunks = [chunk_map[cid] for cid in final_ids if cid in chunk_map]
        allowed = ", ".join(answer_policy["allowed_labels"])
        forbidden = "\n".join(f"  - {b}" for b in answer_policy["forbidden_behaviours"])

        system_prompt = f"""You are a conservative question-answering assistant performing a REVISION.
A previous answer was flagged for hallucination risk or grounding failure.
You must now produce a more conservative answer using ONLY the provided context.

Rules:
- Only state what is explicitly supported by the context chunks.
- If the evidence is absent, say so clearly.
- Do not infer, speculate, or use outside knowledge.
- Label: {allowed}
- Citations must reference ONLY the chunk_ids listed in the user message.
- Maximum {answer_policy['max_citations_per_answer']} citations.

FORBIDDEN:
{forbidden}"""

        user_prompt = f"""Question: {question}

Audit feedback: {audit.recommended_fix}

Final context chunks:

{_format_chunks(final_chunks)}

Provide a revised, conservative answer grounded strictly in the above chunks."""

        full_prompt = system_prompt + "\n\n" + user_prompt

        parsed: RevisedAnswerResponse = call_llm(
            system_prompt, user_prompt, RevisedAnswerResponse, llm_cfg
        )

        valid_ids = set(final_ids)
        safe_citations = [cid for cid in parsed.citations if cid in valid_ids]

        revised.append(
            RevisedAnswer(
                query_id=qid,
                original_audit_label=audit.audit_label,
                original_hallucination_risk=audit.hallucination_risk,
                answer=parsed.answer,
                label=parsed.label,
                citations=safe_citations,
                reasoning_summary=parsed.reasoning_summary,
            )
        )

        log_llm_call(
            stage="revised_answer",
            query_id=qid,
            provider=llm_cfg.get("provider", "openai"),
            model=llm_cfg["model"],
            prompt=full_prompt,
            input_artifacts=[
                "answer_audit.json",
                "chunks.json",
                "review_overrides.json",
            ],
            output_artifact="revised_answers.json",
        )

    return RevisedAnswers(answers=revised)


# ── Retrieval Metrics ─────────────────────────────────────────────────────────


def compute_retrieval_metrics(
    queries: list[dict],
    retrieval_results: RetrievalResults,
    top_k: int,
) -> RetrievalMetrics | None:
    """Compute hit@k and recall@k if queries have expected_evidence annotations."""
    if not any("expected_evidence" in q for q in queries):
        return None

    retrieval_map = {r.query_id: r for r in retrieval_results.results}
    per_query: list[QueryMetric] = []

    for q in queries:
        if "expected_evidence" not in q:
            continue
        expected = q["expected_evidence"]
        qid = q["query_id"]
        retrieved_ids = [rc.chunk_id for rc in retrieval_map[qid].retrieved_chunks]
        hits = sum(1 for eid in expected if eid in retrieved_ids)
        per_query.append(
            QueryMetric(
                query_id=qid,
                expected_chunk_ids=expected,
                retrieved_chunk_ids=retrieved_ids,
                hit_at_k=hits,
                recall_at_k=hits / len(expected) if expected else 0.0,
            )
        )

    if not per_query:
        return None

    overall_hit = sum(1 for m in per_query if m.hit_at_k > 0) / len(per_query)
    overall_recall = sum(m.recall_at_k for m in per_query) / len(per_query)

    return RetrievalMetrics(
        top_k=top_k,
        overall_hit_at_k=overall_hit,
        overall_recall_at_k=overall_recall,
        per_query=per_query,
    )


# ── Retrieval Error Analysis ──────────────────────────────────────────────────


def analyze_retrieval_errors(
    answer_audit: AnswerAudit,
    draft_answers: DraftAnswers,
    retrieval_results: RetrievalResults,
    chunk_list: ChunkList,
) -> RetrievalErrorAnalysis:
    chunk_map = {c.chunk_id: c for c in chunk_list.chunks}
    draft_map = {d.query_id: d for d in draft_answers.answers}
    retrieval_map = {r.query_id: r for r in retrieval_results.results}

    analyses: list[RetrievalError] = []

    for audit in answer_audit.audits:
        qid = audit.query_id
        draft = draft_map.get(qid)
        qr = retrieval_map.get(qid)

        # corpus_gap: topic entirely absent
        if draft and draft.label == "not_in_corpus":
            analyses.append(
                RetrievalError(
                    query_id=qid,
                    failure_type="corpus_gap",
                    description="Question topic is not present in any retrieved or indexed chunk.",
                )
            )
            continue

        # Check for chunking boundary issue: cited chunk text overlaps closely with neighbour
        chunking_issue = False
        if draft and draft.citations:
            for cid in draft.citations:
                c = chunk_map.get(cid)
                if c is None:
                    continue
                # Look for a neighbour chunk from same doc with adjacent start/end
                for other in chunk_list.chunks:
                    if (
                        other.document_name == c.document_name
                        and other.chunk_id != c.chunk_id
                    ):
                        if (
                            abs(other.start_char - c.end_char) <= 10
                            or abs(c.start_char - other.end_char) <= 10
                        ):
                            chunking_issue = True
                            break
                if chunking_issue:
                    break

        if chunking_issue:
            analyses.append(
                RetrievalError(
                    query_id=qid,
                    failure_type="chunking",
                    description="Relevant evidence likely spans a chunk boundary, splitting the answer context.",
                )
            )
            continue

        # ambiguity: top chunks from multiple different documents with similar scores
        if qr:
            doc_names = [
                chunk_map[rc.chunk_id].document_name
                for rc in qr.retrieved_chunks
                if rc.chunk_id in chunk_map
            ]
            if len(set(doc_names)) >= 2:
                scores = [rc.retrieval_score for rc in qr.retrieved_chunks]
                if max(scores) - min(scores) < 0.1:
                    analyses.append(
                        RetrievalError(
                            query_id=qid,
                            failure_type="ambiguity",
                            description="Top chunks are drawn from multiple documents with similar retrieval scores, suggesting query ambiguity.",
                        )
                    )
                    continue

        # ranking: audit passed but answer was graded insufficient_support
        if audit.audit_label == "fail" or (
            draft and draft.label == "insufficient_support"
        ):
            analyses.append(
                RetrievalError(
                    query_id=qid,
                    failure_type="ranking",
                    description="Relevant evidence may exist in lower-ranked chunks that were not retrieved at top-k.",
                )
            )
            continue

        analyses.append(
            RetrievalError(
                query_id=qid,
                failure_type="none",
                description="No retrieval failure detected for this query.",
            )
        )

    return RetrievalErrorAnalysis(analyses=analyses)
