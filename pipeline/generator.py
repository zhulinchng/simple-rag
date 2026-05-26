from __future__ import annotations

from .llm_client import call_llm
from .logger import log_llm_call
from .models import (
    Chunk,
    ChunkList,
    DraftAnswer,
    DraftAnswerResponse,
    DraftAnswers,
    RetrievalResults,
)


def _format_chunks(chunks: list[Chunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c.chunk_id}] ({c.document_name})\n{c.text}")
    return "\n\n---\n\n".join(parts)


def _build_system_prompt(answer_policy: dict) -> str:
    allowed = ", ".join(answer_policy["allowed_labels"])
    forbidden = "\n".join(f"  - {b}" for b in answer_policy["forbidden_behaviours"])
    return f"""You are a grounded question-answering assistant. Your job is to answer questions
using ONLY the provided document chunks. Do not use any outside knowledge.

ANSWER POLICY:
- You must assign one label from: {allowed}
- Use "supported" only when the corpus explicitly answers the question.
- Use "insufficient_support" when the corpus has partial but incomplete evidence.
- Use "not_in_corpus" when the topic is absent from the provided chunks.
- Citations must reference ONLY the chunk_ids provided in the user message.
- Maximum {answer_policy['max_citations_per_answer']} citations per answer.

FORBIDDEN BEHAVIOURS:
{forbidden}

If evidence is weak or missing, you must say so explicitly in the answer.
Do not present outside knowledge as corpus-grounded evidence."""


def _build_user_prompt(question: str, context_chunks: list[Chunk]) -> str:
    return f"""Question: {question}

Retrieved document chunks:

{_format_chunks(context_chunks)}

Using ONLY the chunks above, answer the question. Cite only the chunk_ids listed above."""


def generate_draft_answers(
    retrieval_results: RetrievalResults,
    chunk_list: ChunkList,
    answer_policy: dict,
    llm_cfg: dict,
) -> DraftAnswers:
    chunk_map = {c.chunk_id: c for c in chunk_list.chunks}

    system_prompt = _build_system_prompt(answer_policy)
    answers: list[DraftAnswer] = []

    for qr in retrieval_results.results:
        context_chunks = [
            chunk_map[rc.chunk_id]
            for rc in qr.retrieved_chunks
            if rc.chunk_id in chunk_map
        ]
        user_prompt = _build_user_prompt(qr.question, context_chunks)
        full_prompt = system_prompt + "\n\n" + user_prompt

        parsed: DraftAnswerResponse = call_llm(
            system_prompt, user_prompt, DraftAnswerResponse, llm_cfg
        )

        # Validate citations are in the retrieved set
        valid_ids = {rc.chunk_id for rc in qr.retrieved_chunks}
        safe_citations = [cid for cid in parsed.citations if cid in valid_ids]

        answers.append(
            DraftAnswer(
                query_id=qr.query_id,
                answer=parsed.answer,
                label=parsed.label,
                citations=safe_citations,
                reasoning_summary=parsed.reasoning_summary,
            )
        )

        log_llm_call(
            stage="draft_answer",
            query_id=qr.query_id,
            provider=llm_cfg.get("provider", "openai"),
            model=llm_cfg["model"],
            prompt=full_prompt,
            input_artifacts=["retrieval_results.json", "chunks.json"],
            output_artifact="draft_answers.json",
        )

    return DraftAnswers(answers=answers)
