from __future__ import annotations

import json
import sys
from pathlib import Path


def _write_valid_artifacts(base: Path) -> None:
    """Write a minimal but fully valid set of pipeline artifacts."""
    # validator checks for policy.json, queries.json, and documents/
    (base / "policy.json").write_text(
        json.dumps(
            {
                "retrieval": {"top_k": 3},
                "answer_policy": {
                    "allowed_labels": [
                        "supported",
                        "insufficient_support",
                        "not_in_corpus",
                    ]
                },
            }
        )
    )
    (base / "queries.json").write_text(
        '{"queries": [{"query_id": "Q1", "question": "What is in the doc?"}]}'
    )
    docs = base / "documents"
    docs.mkdir()
    (docs / "doc.txt").write_text("Sample document text.")

    chunks = {
        "chunks": [
            {
                "chunk_id": "doc_000000",
                "document_name": "doc.txt",
                "start_char": 0,
                "end_char": 50,
                "text": "Sample text for testing.",
            }
        ]
    }
    index_meta = {
        "retrieval_mode": "keyword",
        "bm25_weight": 1.0,
        "embedding_weight": 0.0,
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_count": 1,
        "document_names": ["doc.txt"],
        "built_at": "2026-01-01T00:00:00+00:00",
        "reranker_model": None,
        "reranker_enabled": False,
    }
    retrieval = {
        "results": [
            {
                "query_id": "Q1",
                "question": "What is in the doc?",
                "retrieved_chunks": [
                    {
                        "chunk_id": "doc_000000",
                        "document_name": "doc.txt",
                        "rank": 1,
                        "retrieval_score": 0.9,
                    }
                ],
            }
        ]
    }
    draft_answers = {
        "answers": [
            {
                "query_id": "Q1",
                "answer": "The doc contains sample text.",
                "label": "supported",
                "citations": ["doc_000000"],
                "reasoning_summary": "Directly stated.",
            }
        ]
    }
    review_overrides = {"overrides": []}
    answer_audit = {
        "audits": [
            {
                "query_id": "Q1",
                "audit_label": "pass",
                "support_assessment": "Supported by chunk.",
                "citation_check": "Citation matches retrieved chunk.",
                "hallucination_risk": "low",
                "recommended_fix": "None needed.",
                "final_context_chunk_ids": ["doc_000000"],
            }
        ]
    }

    (base / "chunks.json").write_text(json.dumps(chunks))
    (base / "index_metadata.json").write_text(json.dumps(index_meta))
    (base / "retrieval_results.json").write_text(json.dumps(retrieval))
    (base / "draft_answers.json").write_text(json.dumps(draft_answers))
    (base / "review_overrides.json").write_text(json.dumps(review_overrides))
    (base / "answer_audit.json").write_text(json.dumps(answer_audit))
    (base / "final_report.md").write_text(
        "# Final Report\n\n" + "Report content. " * 10
    )
    (base / "llm_calls.jsonl").write_text(
        json.dumps(
            {
                "stage": "draft_answer",
                "query_id": "Q1",
                "timestamp": "2026-01-01T00:00:01+00:00",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "prompt_hash": "abc123",
                "input_artifacts": ["chunks.json"],
                "output_artifact": "draft_answers.json",
            }
        )
        + "\n"
        + json.dumps(
            {
                "stage": "audit",
                "query_id": "Q1",
                "timestamp": "2026-01-01T00:00:02+00:00",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "prompt_hash": "def456",
                "input_artifacts": ["draft_answers.json"],
                "output_artifact": "answer_audit.json",
            }
        )
        + "\n"
    )


def test_passes_with_all_valid_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_valid_artifacts(tmp_path)

    validate_path = Path(__file__).parent.parent / "validate.py"
    monkeypatch.syspath_prepend(str(Path(__file__).parent.parent))
    import importlib.util

    spec = importlib.util.spec_from_file_location("validate", validate_path)
    mod = importlib.util.load_from_spec = None  # reset

    import subprocess

    result = subprocess.run(
        [sys.executable, str(validate_path)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_fails_missing_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_valid_artifacts(tmp_path)
    (tmp_path / "answer_audit.json").unlink()

    import subprocess

    validate_path = Path(__file__).parent.parent / "validate.py"
    result = subprocess.run(
        [sys.executable, str(validate_path)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 1


def test_fails_invalid_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_valid_artifacts(tmp_path)
    (tmp_path / "chunks.json").write_text("{invalid json{{")

    import subprocess

    validate_path = Path(__file__).parent.parent / "validate.py"
    result = subprocess.run(
        [sys.executable, str(validate_path)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 1


def test_fails_bad_draft_label(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_valid_artifacts(tmp_path)
    draft = json.loads((tmp_path / "draft_answers.json").read_text())
    draft["answers"][0]["label"] = "wrong_label"
    (tmp_path / "draft_answers.json").write_text(json.dumps(draft))

    import subprocess

    validate_path = Path(__file__).parent.parent / "validate.py"
    result = subprocess.run(
        [sys.executable, str(validate_path)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 1


def test_fails_citation_not_in_retrieved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_valid_artifacts(tmp_path)
    draft = json.loads((tmp_path / "draft_answers.json").read_text())
    draft["answers"][0]["citations"] = ["nonexistent_chunk_id"]
    (tmp_path / "draft_answers.json").write_text(json.dumps(draft))

    import subprocess

    validate_path = Path(__file__).parent.parent / "validate.py"
    result = subprocess.run(
        [sys.executable, str(validate_path)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 1
