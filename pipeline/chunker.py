from __future__ import annotations

from pathlib import Path

from .models import Chunk, ChunkList


def _doc_stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("-", "_")


def chunk_document(
    text: str, doc_name: str, chunk_size: int, overlap: int
) -> list[Chunk]:
    """Deterministic sliding-window character chunker. No LLM calls."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    stem = _doc_stem(Path(doc_name))
    chunks: list[Chunk] = []
    step = chunk_size - overlap
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunk_id = f"{stem}_{start:06d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_name=doc_name,
                start_char=start,
                end_char=end,
                text=chunk_text,
            )
        )
        if end == len(text):
            break
        start += step

    return chunks


def load_and_chunk(documents_dir: Path, chunk_size: int, overlap: int) -> ChunkList:
    """Read all .txt files (sorted for determinism) and chunk them."""
    all_chunks: list[Chunk] = []
    for txt_path in sorted(documents_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8")
        doc_chunks = chunk_document(text, txt_path.name, chunk_size, overlap)
        all_chunks.extend(doc_chunks)
    return ChunkList(chunks=all_chunks)
