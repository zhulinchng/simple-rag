from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.models import Chunk, ChunkList


@pytest.fixture
def sample_chunks() -> ChunkList:
    chunks = [
        Chunk(
            chunk_id="alpha_000000",
            document_name="alpha.txt",
            start_char=0,
            end_char=100,
            text="The standard plan retains event data for 90 days. Longer retention is available on enterprise.",
        ),
        Chunk(
            chunk_id="alpha_000050",
            document_name="alpha.txt",
            start_char=50,
            end_char=150,
            text="Longer retention is available on enterprise plans with custom SLAs negotiated upfront.",
        ),
        Chunk(
            chunk_id="beta_000000",
            document_name="beta.txt",
            start_char=0,
            end_char=100,
            text="SCIM provisioning is supported for enterprise customers using SSO with Okta or Azure AD.",
        ),
        Chunk(
            chunk_id="beta_000050",
            document_name="beta.txt",
            start_char=50,
            end_char=150,
            text="Customers on the standard plan can request refunds within 14 days of billing via support.",
        ),
    ]
    return ChunkList(chunks=chunks)


@pytest.fixture
def sample_policy() -> dict:
    policy_path = Path(__file__).parent.parent / "policy.json"
    return json.loads(policy_path.read_text())
