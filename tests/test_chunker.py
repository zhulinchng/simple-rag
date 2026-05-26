from __future__ import annotations

import re
import tempfile
from pathlib import Path

from pipeline.chunker import chunk_document, load_and_chunk


def test_determinism():
    text = "Hello world this is a test document with enough text to create multiple chunks."
    chunks1 = chunk_document(text, "doc.txt", chunk_size=30, overlap=5)
    chunks2 = chunk_document(text, "doc.txt", chunk_size=30, overlap=5)
    assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]
    assert [c.text for c in chunks1] == [c.text for c in chunks2]


def test_chunk_id_format():
    text = "A" * 200
    chunks = chunk_document(text, "my-doc.txt", chunk_size=50, overlap=10)
    pattern = re.compile(r"^[a-zA-Z0-9_]+_\d{6}$")
    for c in chunks:
        assert pattern.match(c.chunk_id), f"Bad chunk_id: {c.chunk_id}"


def test_overlap_boundary():
    text = "X" * 500
    overlap = 50
    chunk_size = 200
    chunks = chunk_document(text, "doc.txt", chunk_size=chunk_size, overlap=overlap)
    for i in range(len(chunks) - 1):
        expected_next_start = chunks[i].end_char - overlap
        assert chunks[i + 1].start_char == expected_next_start


def test_single_chunk_short_doc():
    text = "Short text."
    chunks = chunk_document(text, "tiny.txt", chunk_size=350, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(text)


def test_multi_doc_sorted_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        (d / "beta.txt").write_text("Beta document content here.")
        (d / "alpha.txt").write_text("Alpha document content here.")
        chunk_list = load_and_chunk(d, chunk_size=100, overlap=10)
        doc_names = [c.document_name for c in chunk_list.chunks]
        # alpha.txt must appear before beta.txt
        alpha_idx = next(i for i, n in enumerate(doc_names) if "alpha" in n)
        beta_idx = next(i for i, n in enumerate(doc_names) if "beta" in n)
        assert alpha_idx < beta_idx
