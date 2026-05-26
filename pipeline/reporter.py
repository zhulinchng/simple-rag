from __future__ import annotations

from pathlib import Path

from .models import (
    AnswerAudit,
    DraftAnswers,
    IndexMetadata,
    RetrievalResults,
    ReviewOverrides,
)


def _label_emoji(label: str) -> str:
    return {"supported": "✓", "insufficient_support": "~", "not_in_corpus": "✗"}.get(
        label, "?"
    )


def _risk_marker(risk: str) -> str:
    return {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}.get(risk, risk.upper())


def generate_report(
    retrieval_results: RetrievalResults,
    draft_answers: DraftAnswers,
    review_overrides: ReviewOverrides,
    answer_audit: AnswerAudit,
    index_metadata: IndexMetadata,
    output_path: Path = Path("final_report.md"),
) -> None:
    draft_map = {d.query_id: d for d in draft_answers.answers}
    audit_map = {a.query_id: a for a in answer_audit.audits}
    override_map = {
        o.query_id: o.override_chunk_ids for o in review_overrides.overrides
    }
    retrieval_map = {r.query_id: r for r in retrieval_results.results}

    lines: list[str] = []
    lines.append("# RAG Pipeline Evaluation Report\n")

    # ── 1. Retrieval Summary ──────────────────────────────────────────────────
    lines.append("## 1. Retrieval Summary\n")
    lines.append(f"- **Mode**: {index_metadata.retrieval_mode}")
    lines.append(f"- **BM25 weight**: {index_metadata.bm25_weight}")
    lines.append(f"- **Embedding weight**: {index_metadata.embedding_weight}")
    lines.append(f"- **Embedding model**: {index_metadata.embedding_model}")
    lines.append(
        f"- **Top-k**: {len(next(iter(retrieval_results.results)).retrieved_chunks)}"
    )
    lines.append(f"- **Total chunks indexed**: {index_metadata.chunk_count}")
    lines.append(f"- **Documents indexed**: {', '.join(index_metadata.document_names)}")
    lines.append(f"- **Index built at**: {index_metadata.built_at}")
    lines.append("")

    # ── 2. Query-by-Query Results ─────────────────────────────────────────────
    lines.append("## 2. Query-by-Query Results\n")
    for qr in retrieval_results.results:
        qid = qr.query_id
        draft = draft_map.get(qid)
        audit = audit_map.get(qid)
        final_ids = (
            audit.final_context_chunk_ids
            if audit
            else [rc.chunk_id for rc in qr.retrieved_chunks]
        )
        overridden = qid in override_map

        lines.append(f"### {qid}")
        lines.append(f"**Question**: {qr.question}")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| Final context chunk IDs | {', '.join(final_ids)} |")
        lines.append(f"| Overridden | {'Yes' if overridden else 'No'} |")
        lines.append(
            f"| Draft label | `{draft.label if draft else 'N/A'}` {_label_emoji(draft.label) if draft else ''} |"
        )
        lines.append(
            f"| Draft citations | {', '.join(draft.citations) if draft and draft.citations else 'none'} |"
        )
        lines.append(f"| Audit label | `{audit.audit_label if audit else 'N/A'}` |")
        lines.append(
            f"| Hallucination risk | **{_risk_marker(audit.hallucination_risk) if audit else 'N/A'}** |"
        )
        lines.append("")
        if draft:
            lines.append(f"**Draft answer**: {draft.answer}")
            lines.append("")
        if audit:
            rec = audit.recommended_fix or "None"
            lines.append(
                f"**Final recommendation**: {rec if audit.audit_label == 'fail' else 'Answer approved.'}"
            )
            lines.append("")

    # ── 3. Reviewed Overrides ─────────────────────────────────────────────────
    lines.append("## 3. Reviewed Overrides\n")
    if review_overrides.overrides:
        for o in review_overrides.overrides:
            lines.append(
                f"- **{o.query_id}**: forced context → `{', '.join(o.override_chunk_ids)}`"
            )
    else:
        lines.append("_No overrides applied. Audit used original retrieval results._")
    lines.append("")

    # ── 4. Audit Findings ─────────────────────────────────────────────────────
    lines.append("## 4. Audit Findings\n")
    lines.append("| Query | Audit | Hallucination Risk | Citation Check |")
    lines.append("|-------|-------|--------------------|----------------|")
    for audit in answer_audit.audits:
        lines.append(
            f"| {audit.query_id} | `{audit.audit_label}` | {_risk_marker(audit.hallucination_risk)} "
            f"| {audit.citation_check[:80]} |"
        )
    lines.append("")

    pass_count = sum(1 for a in answer_audit.audits if a.audit_label == "pass")
    fail_count = len(answer_audit.audits) - pass_count
    lines.append(
        f"**Summary**: {pass_count} passed, {fail_count} failed out of {len(answer_audit.audits)} queries.\n"
    )

    # ── 5. Failure Modes Observed ─────────────────────────────────────────────
    lines.append("## 5. Failure Modes Observed\n")
    failure_modes: list[str] = []

    for audit in answer_audit.audits:
        draft = draft_map.get(audit.query_id)
        if audit.audit_label == "fail":
            failure_modes.append(
                f"- **{audit.query_id}** (audit FAIL): {audit.support_assessment}"
            )
        if audit.hallucination_risk == "high":
            failure_modes.append(
                f"- **{audit.query_id}** (HIGH hallucination risk): {audit.recommended_fix}"
            )
        if draft and draft.label == "not_in_corpus":
            failure_modes.append(
                f"- **{audit.query_id}** (corpus gap): question not answerable from corpus"
            )

    if failure_modes:
        lines.extend(failure_modes)
    else:
        lines.append("_No significant failure modes detected._")
    lines.append("")

    # ── 6. Recommended Improvements ──────────────────────────────────────────
    lines.append("## 6. Recommended Improvements\n")
    recommendations: list[str] = []

    high_risk = [a for a in answer_audit.audits if a.hallucination_risk == "high"]
    if high_risk:
        recommendations.append(
            f"- {len(high_risk)} answer(s) had HIGH hallucination risk. "
            "Consider adding more specific documents to the corpus or refining prompts."
        )

    corpus_gaps = [d for d in draft_answers.answers if d.label == "not_in_corpus"]
    if corpus_gaps:
        ids = ", ".join(d.query_id for d in corpus_gaps)
        recommendations.append(
            f"- Queries {ids} could not be answered from the corpus. "
            "Add relevant documentation to fill these gaps."
        )

    insufficient = [
        d for d in draft_answers.answers if d.label == "insufficient_support"
    ]
    if insufficient:
        ids = ", ".join(d.query_id for d in insufficient)
        recommendations.append(
            f"- Queries {ids} had insufficient support. "
            "Consider reducing chunk size or increasing top-k to surface more relevant context."
        )

    failed_audits = [a for a in answer_audit.audits if a.audit_label == "fail"]
    for fa in failed_audits:
        if fa.recommended_fix:
            recommendations.append(f"- [{fa.query_id}]: {fa.recommended_fix}")

    if not recommendations:
        recommendations.append(
            "- Pipeline performed well. No major improvements required."
        )

    lines.extend(recommendations)
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[REPORT] Written to {output_path}")
