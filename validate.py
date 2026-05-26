#!/usr/bin/env python3
"""
Validation command: python validate.py
Checks all pipeline artifacts for correctness and stage ordering.
Exit 0 = all checks pass; exit 1 = one or more checks failed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_ARTIFACTS = [
    "chunks.json",
    "index_metadata.json",
    "retrieval_results.json",
    "draft_answers.json",
    "review_overrides.json",
    "answer_audit.json",
    "final_report.md",
    "llm_calls.jsonl",
]

OPTIONAL_ARTIFACTS = [
    "retrieval_metrics.json",
    "revised_answers.json",
    "retrieval_error_analysis.json",
]

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
SKIP = "\033[33m[SKIP]\033[0m"


def _load_json(path: str) -> Any | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return "INVALID"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    msg = f"  {status} {label}"
    if not condition and detail:
        msg += f"\n         → {detail}"
    print(msg)
    return condition


def main() -> int:
    failures = 0
    print("\n=== RAG Pipeline Validation ===\n")

    # ── 1. Required artifacts exist ──────────────────────────────────────────
    print("1. Required artifacts exist")
    for artifact in REQUIRED_ARTIFACTS:
        ok = Path(artifact).exists()
        if not check(artifact, ok, f"File not found: {artifact}"):
            failures += 1

    # ── 2. JSON files are valid ───────────────────────────────────────────────
    print("\n2. JSON files are valid")
    json_files = [a for a in REQUIRED_ARTIFACTS if a.endswith(".json")]
    # Also validate optional JSON files if present
    for artifact in OPTIONAL_ARTIFACTS:
        if Path(artifact).exists() and artifact.endswith(".json"):
            json_files.append(artifact)

    for jf in json_files:
        data = _load_json(jf)
        if data is None:
            check(jf, False, "File missing (skipped JSON validation)")
            continue
        ok = data != "INVALID"
        if not check(jf, ok, "File contains invalid JSON"):
            failures += 1

    # ── 3. Load core data ─────────────────────────────────────────────────────
    policy_data = _load_json("policy.json")
    queries_data = _load_json("queries.json")
    chunks_data = _load_json("chunks.json")
    index_meta = _load_json("index_metadata.json")
    retrieval_data = _load_json("retrieval_results.json")
    draft_data = _load_json("draft_answers.json")
    overrides_data = _load_json("review_overrides.json")
    audit_data = _load_json("answer_audit.json")

    # ── 3. Input files readable from disk ────────────────────────────────────
    print("\n3. Input files readable from disk")
    if not check("policy.json exists", policy_data not in (None, "INVALID")):
        failures += 1
    if not check("queries.json exists", queries_data not in (None, "INVALID")):
        failures += 1
    if not check("documents/ directory exists", Path("documents").exists()):
        failures += 1
    doc_files = (
        list(Path("documents").glob("*.txt")) if Path("documents").exists() else []
    )
    if not check(
        "documents/*.txt files present",
        len(doc_files) > 0,
        "No .txt files found in documents/",
    ):
        failures += 1

    # ── 4. Chunking before LLM calls ─────────────────────────────────────────
    print("\n4. Chunking happened before any LLM call")
    llm_log = Path("llm_calls.jsonl")
    chunks_path = Path("chunks.json")
    if llm_log.exists() and chunks_path.exists():
        chunk_mtime = chunks_path.stat().st_mtime
        llm_mtime = llm_log.stat().st_mtime
        ok = chunk_mtime <= llm_mtime
        if not check(
            "chunks.json mtime ≤ llm_calls.jsonl mtime",
            ok,
            f"chunks mtime={chunk_mtime:.0f}, llm_calls mtime={llm_mtime:.0f}",
        ):
            failures += 1
    elif not llm_log.exists():
        print(f"  {SKIP} llm_calls.jsonl missing — skipping mtime check")
    elif not chunks_path.exists():
        print(f"  {SKIP} chunks.json missing — skipping mtime check")

    # ── 5. Chunk IDs are unique ───────────────────────────────────────────────
    print("\n5. Chunk IDs are unique")
    if chunks_data and chunks_data != "INVALID":
        chunk_ids = [c["chunk_id"] for c in chunks_data.get("chunks", [])]
        ok = len(chunk_ids) == len(set(chunk_ids))
        if not check("No duplicate chunk_ids", ok, "Duplicate IDs found"):
            failures += 1
        if not check(
            "At least one chunk", len(chunk_ids) > 0, "chunks.json has no chunks"
        ):
            failures += 1

    # ── 6. Index metadata retrieval mode ─────────────────────────────────────
    print("\n6. Index metadata is valid")
    if index_meta and index_meta != "INVALID":
        valid_modes = {"hybrid", "keyword", "embedding"}
        mode = index_meta.get("retrieval_mode", "")
        if not check(
            f"retrieval_mode in {valid_modes}", mode in valid_modes, f"Got: '{mode}'"
        ):
            failures += 1

    # ── 7. Every query has retrieval results ──────────────────────────────────
    print("\n7. Every query has retrieval results")
    if (
        queries_data
        and retrieval_data
        and queries_data != "INVALID"
        and retrieval_data != "INVALID"
    ):
        query_ids = {q["query_id"] for q in queries_data.get("queries", [])}
        retrieval_ids = {r["query_id"] for r in retrieval_data.get("results", [])}
        missing = query_ids - retrieval_ids
        if not check(
            "All queries have retrieval results", not missing, f"Missing: {missing}"
        ):
            failures += 1

        for result in retrieval_data.get("results", []):
            qid = result["query_id"]
            has_chunks = len(result.get("retrieved_chunks", [])) > 0
            if not check(f"{qid} has at least one retrieved chunk", has_chunks):
                failures += 1

    # ── 8. Draft answer labels are from allowed set ───────────────────────────
    print("\n8. Draft answer labels use only allowed labels")
    if (
        policy_data
        and draft_data
        and policy_data != "INVALID"
        and draft_data != "INVALID"
    ):
        allowed = set(policy_data.get("answer_policy", {}).get("allowed_labels", []))
        for ans in draft_data.get("answers", []):
            label = ans.get("label", "")
            qid = ans.get("query_id", "?")
            if not check(
                f"{qid} label '{label}' is allowed",
                label in allowed,
                f"Allowed: {allowed}",
            ):
                failures += 1

    # ── 9. Citations refer only to retrieved chunk IDs ────────────────────────
    print("\n9. Citations refer only to retrieved or override chunk IDs")
    if (
        retrieval_data
        and draft_data
        and overrides_data
        and all(d != "INVALID" for d in [retrieval_data, draft_data, overrides_data])
    ):

        retrieval_map: dict[str, set[str]] = {}
        for r in retrieval_data.get("results", []):
            retrieval_map[r["query_id"]] = {
                rc["chunk_id"] for rc in r.get("retrieved_chunks", [])
            }

        override_map: dict[str, set[str]] = {
            o["query_id"]: set(o["override_chunk_ids"])
            for o in overrides_data.get("overrides", [])
        }

        for ans in draft_data.get("answers", []):
            qid = ans.get("query_id", "?")
            citations = ans.get("citations", [])
            valid_ids = override_map.get(qid, retrieval_map.get(qid, set()))
            bad = [c for c in citations if c not in valid_ids]
            if not check(
                f"{qid} citations are valid", not bad, f"Invalid citations: {bad}"
            ):
                failures += 1

    # ── 10. LLM calls have separate draft_answer and audit records ────────────
    print("\n10. llm_calls.jsonl has required records")
    if llm_log.exists() and queries_data and queries_data != "INVALID":
        records = []
        with llm_log.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        query_ids = {q["query_id"] for q in queries_data.get("queries", [])}

        draft_records = {
            r["query_id"] for r in records if r.get("stage") == "draft_answer"
        }
        audit_records = {r["query_id"] for r in records if r.get("stage") == "audit"}

        missing_draft = query_ids - draft_records
        if not check(
            "draft_answer record per query",
            not missing_draft,
            f"Missing: {missing_draft}",
        ):
            failures += 1

        missing_audit = query_ids - audit_records
        if not check(
            "audit record per query", not missing_audit, f"Missing: {missing_audit}"
        ):
            failures += 1

        # Audit timestamps must be after draft timestamps
        draft_ts = {
            r["query_id"]: r["timestamp"]
            for r in records
            if r.get("stage") == "draft_answer"
        }
        audit_ts = {
            r["query_id"]: r["timestamp"] for r in records if r.get("stage") == "audit"
        }
        for qid in query_ids & draft_ts.keys() & audit_ts.keys():
            ok = audit_ts[qid] >= draft_ts[qid]
            if not check(
                f"{qid} audit after draft",
                ok,
                f"draft={draft_ts[qid]} audit={audit_ts[qid]}",
            ):
                failures += 1

    # ── 11. Overrides are applied to audit inputs ─────────────────────────────
    print("\n11. Overrides applied to audit final_context_chunk_ids")
    if (
        overrides_data
        and audit_data
        and all(d != "INVALID" for d in [overrides_data, audit_data])
    ):
        override_map2 = {
            o["query_id"]: set(o["override_chunk_ids"])
            for o in overrides_data.get("overrides", [])
        }
        audit_map = {a["query_id"]: a for a in audit_data.get("audits", [])}

        for qid, expected_ids in override_map2.items():
            audit = audit_map.get(qid)
            if audit is None:
                check(
                    f"{qid} override reflected in audit", False, "No audit record found"
                )
                failures += 1
                continue
            actual = set(audit.get("final_context_chunk_ids", []))
            ok = expected_ids == actual
            if not check(
                f"{qid} override chunk IDs match audit final_context",
                ok,
                f"Expected {expected_ids}, got {actual}",
            ):
                failures += 1

    # ── 12. Final report is non-empty ─────────────────────────────────────────
    print("\n12. final_report.md is non-empty")
    report_path = Path("final_report.md")
    if report_path.exists():
        size = report_path.stat().st_size
        if not check("final_report.md has content", size > 100, f"Only {size} bytes"):
            failures += 1
    else:
        check("final_report.md exists", False)
        failures += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 40}")
    if failures == 0:
        print("\033[32m✓ All checks passed.\033[0m\n")
        return 0
    else:
        print(f"\033[31m✗ {failures} check(s) failed.\033[0m\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
