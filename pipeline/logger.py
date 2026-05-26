from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import LLMCallRecord

LOGFILE = Path("llm_calls.jsonl")


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def log_llm_call(
    *,
    stage: str,
    query_id: Optional[str],
    provider: str,
    model: str,
    prompt: str,
    input_artifacts: list[str],
    output_artifact: str,
) -> None:
    record = LLMCallRecord(
        stage=stage,
        query_id=query_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        provider=provider,
        model=model,
        prompt_hash=_hash_prompt(prompt),
        input_artifacts=input_artifacts,
        output_artifact=output_artifact,
    )
    with LOGFILE.open("a") as f:
        f.write(record.model_dump_json() + "\n")


def read_llm_calls() -> list[LLMCallRecord]:
    if not LOGFILE.exists():
        return []
    records = []
    with LOGFILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(LLMCallRecord.model_validate_json(line))
    return records


def clear_log() -> None:
    if LOGFILE.exists():
        LOGFILE.unlink()
