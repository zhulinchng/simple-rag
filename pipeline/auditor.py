from __future__ import annotations

from .llm_client import call_llm
from .logger import log_llm_call
from .models import (
    AnswerAudit,
    AuditResponse,
    AuditResult,
    Chunk,
    ChunkList,
    DraftAnswer,
    DraftAnswers,
    RetrievalResults,
    ReviewOverrides,
)


def _get_final_context(
    query_id: str,
    retrieval_results: RetrievalResults,
    review_overrides: ReviewOverrides,
    chunk_map: dict[str, Chunk],
) -> tuple[list[str], list[Chunk]]:
    """Return (final_chunk_ids, final_chunks) after applying any overrides."""
    override_map = {
        o.query_id: o.override_chunk_ids for o in review_overrides.overrides
    }

    if query_id in override_map:
        ids = override_map[query_id]
    else:
        qr = next(r for r in retrieval_results.results if r.query_id == query_id)
        ids = [rc.chunk_id for rc in qr.retrieved_chunks]

    chunks = [chunk_map[cid] for cid in ids if cid in chunk_map]
    return ids, chunks


def _format_chunks(chunks: list[Chunk]) -> str:
    parts = [f"[{c.chunk_id}] ({c.document_name})\n{c.text}" for c in chunks]
    return "\n\n---\n\n".join(parts)


def _build_system_prompt(answer_policy: dict) -> str:
    forbidden = "\n".join(f"  - {b}" for b in answer_policy["forbidden_behaviours"])
    return f"""You are a strict answer auditor. Your job is to verify whether a draft answer
is properly grounded in the provided document chunks.

You must check:
1. Is the answer actually supported by the final context chunks?
2. Are citations appropriate and present in the context?
3. Does the answer overclaim beyond what the corpus states?

FORBIDDEN BEHAVIOURS to check for:
{forbidden}

Assign audit_label = "pass" only if the answer is fully grounded and citations are valid.
Assign audit_label = "fail" if the answer fabricates, overclaims, or lacks proper citation.
Assign hallucination_risk: low (fully grounded), medium (minor unsupported claim), high (fabrication or clear overclaim)."""


def _build_user_prompt(
    question: str,
    draft: DraftAnswer,
    final_chunks: list[Chunk],
) -> str:
    return f"""Question: {question}

Draft answer:
  Label: {draft.label}
  Answer: {draft.answer}
  Citations: {', '.join(draft.citations) if draft.citations else 'none'}
  Reasoning: {draft.reasoning_summary}

Final context chunks (after any human review overrides):

{_format_chunks(final_chunks)}

Audit the draft answer against the final context. Check grounding, citations, and overclaiming."""


def audit_answers(
    draft_answers: DraftAnswers,
    retrieval_results: RetrievalResults,
    review_overrides: ReviewOverrides,
    chunk_list: ChunkList,
    answer_policy: dict,
    llm_cfg: dict,
) -> AnswerAudit:
    chunk_map = {c.chunk_id: c for c in chunk_list.chunks}
    system_prompt = _build_system_prompt(answer_policy)

    audits: list[AuditResult] = []

    # Build question map from retrieval results
    question_map = {r.query_id: r.question for r in retrieval_results.results}

    for draft in draft_answers.answers:
        final_ids, final_chunks = _get_final_context(
            draft.query_id, retrieval_results, review_overrides, chunk_map
        )
        question = question_map[draft.query_id]
        user_prompt = _build_user_prompt(question, draft, final_chunks)
        full_prompt = system_prompt + "\n\n" + user_prompt

        parsed: AuditResponse = call_llm(
            system_prompt, user_prompt, AuditResponse, llm_cfg
        )

        audits.append(
            AuditResult(
                query_id=draft.query_id,
                audit_label=parsed.audit_label,
                support_assessment=parsed.support_assessment,
                citation_check=parsed.citation_check,
                hallucination_risk=parsed.hallucination_risk,
                recommended_fix=parsed.recommended_fix,
                final_context_chunk_ids=final_ids,
            )
        )

        log_llm_call(
            stage="audit",
            query_id=draft.query_id,
            provider=llm_cfg.get("provider", "openai"),
            model=llm_cfg["model"],
            prompt=full_prompt,
            input_artifacts=[
                "draft_answers.json",
                "retrieval_results.json",
                "review_overrides.json",
                "chunks.json",
            ],
            output_artifact="answer_audit.json",
        )

    return AnswerAudit(audits=audits)
