from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline.models import (
    AuditResult,
    Chunk,
    DraftAnswer,
    IndexMetadata,
    LLMCallRecord,
)


def test_chunk_roundtrip():
    c = Chunk(
        chunk_id="doc_000000",
        document_name="doc.txt",
        start_char=0,
        end_char=50,
        text="Sample text for testing roundtrip serialization.",
    )
    restored = Chunk.model_validate_json(c.model_dump_json())
    assert restored == c


def test_draft_answer_invalid_label():
    with pytest.raises(ValidationError):
        DraftAnswer(
            query_id="Q1",
            answer="Some answer",
            label="unknown",
            citations=[],
            reasoning_summary="reasoning",
        )


def test_audit_invalid_risk():
    with pytest.raises(ValidationError):
        AuditResult(
            query_id="Q1",
            audit_label="pass",
            support_assessment="good",
            citation_check="ok",
            hallucination_risk="critical",
            recommended_fix="none",
            final_context_chunk_ids=[],
        )


def test_index_metadata_invalid_mode():
    with pytest.raises(ValidationError):
        IndexMetadata(
            retrieval_mode="fuzzy",
            bm25_weight=0.5,
            embedding_weight=0.5,
            embedding_model="all-MiniLM-L6-v2",
            chunk_count=10,
            document_names=["doc.txt"],
            built_at="2026-01-01T00:00:00+00:00",
        )


def test_llm_call_record_roundtrip():
    record = LLMCallRecord(
        stage="generation",
        query_id="Q1",
        timestamp="2026-01-01T00:00:00+00:00",
        provider="openai",
        model="gpt-4o-mini",
        prompt_hash="abc123def456",
        input_artifacts=["chunks.json", "retrieval_results.json"],
        output_artifact="draft_answers.json",
    )
    restored = LLMCallRecord.model_validate_json(record.model_dump_json())
    assert restored == record
