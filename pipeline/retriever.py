from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from bayesian_bm25 import (
    BayesianBM25Scorer,
    cosine_to_probability,
    log_odds_conjunction,
)

from .models import (
    ChunkList,
    IndexMetadata,
    QueryRetrieval,
    RetrievalResults,
    RetrievedChunk,
)


def _tokenize(text: str) -> list[str]:
    """Simple deterministic tokenizer: lowercase, split on non-alphanumeric."""
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    """
    Hybrid BM25 + embedding retriever with optional CrossEncoder reranking.

    bm25_weight=1.0, embedding_weight=0.0  → pure BM25
    bm25_weight=0.0, embedding_weight=1.0  → pure embedding
    Default 0.5 / 0.5 → probabilistic log-odds fusion via bayesian-bm25.
    """

    def __init__(
        self,
        chunk_list: ChunkList,
        bm25_weight: float,
        embedding_weight: float,
        embedding_model: str,
        top_k: int,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        reranker_enabled: bool = False,
        reranker_expansion_factor: int = 3,
    ) -> None:
        self._chunks = chunk_list.chunks
        self._bm25_weight = bm25_weight
        self._embedding_weight = embedding_weight
        self._top_k = top_k
        self._embedding_model_name = embedding_model
        self._reranker_enabled = reranker_enabled
        self._reranker_expansion = reranker_expansion_factor
        self._reranker_model_name = reranker_model

        # Build BayesianBM25 index
        tokenized = [_tokenize(c.text) for c in self._chunks]
        self._bm25_scorer = BayesianBM25Scorer(
            k1=1.2, b=0.75, method="lucene", base_rate="auto"
        )
        self._bm25_scorer.index(tokenized, show_progress=False)

        # Build embedding index (skip if weight == 0)
        self._embeddings: Optional[np.ndarray] = None
        if embedding_weight > 0.0:
            self._embeddings = self._build_embeddings(embedding_model)

        # Load CrossEncoder reranker (skip if disabled)
        self._reranker = None
        if reranker_enabled:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(reranker_model)

    def _build_embeddings(self, model_name: str) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        texts = [c.text for c in self._chunks]
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def _cosine_scores(self, query: str) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self._embedding_model_name)
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[
            0
        ]
        # dot product of L2-normalized vectors == cosine similarity in [-1, 1]
        return self._embeddings @ q_emb

    def _hybrid_scores(self, question: str) -> np.ndarray:
        """Return combined relevance scores for all chunks."""
        n = len(self._chunks)

        if self._bm25_weight > 0:
            doc_ids, bm25_partial = self._bm25_scorer.retrieve(
                [_tokenize(question)], k=n
            )
            bm25_probs = np.zeros(n, dtype=float)
            bm25_probs[doc_ids[0]] = bm25_partial[0]
        else:
            bm25_probs = np.zeros(n, dtype=float)

        if self._embedding_weight > 0.0 and self._embeddings is not None:
            cosine_sims = self._cosine_scores(question)
            emb_probs = cosine_to_probability(cosine_sims)
        else:
            emb_probs = np.zeros(n, dtype=float)

        if self._bm25_weight > 0 and self._embedding_weight > 0:
            stacked = np.stack([bm25_probs, emb_probs], axis=-1)
            return log_odds_conjunction(
                stacked,
                weights=np.array([self._bm25_weight, self._embedding_weight]),
            )
        elif self._bm25_weight > 0:
            return bm25_probs
        else:
            return emb_probs

    def retrieve(self, query_id: str, question: str) -> QueryRetrieval:
        combined = self._hybrid_scores(question)

        # Stable sort: descending score, then ascending chunk_id for ties
        chunk_ids = [c.chunk_id for c in self._chunks]
        indexed = sorted(
            enumerate(combined),
            key=lambda x: (-x[1], chunk_ids[x[0]]),
        )

        if self._reranker_enabled and self._reranker is not None:
            # Expand candidate set, then rerank
            fetch_k = min(self._top_k * self._reranker_expansion, len(self._chunks))
            candidate_indices = [i for i, _ in indexed[:fetch_k]]
            candidates = [self._chunks[i] for i in candidate_indices]

            pairs = [(question, c.text) for c in candidates]
            rerank_scores = self._reranker.predict(pairs)

            reranked = sorted(
                zip(candidates, rerank_scores),
                key=lambda x: -x[1],
            )
            top_chunks = [c for c, _ in reranked[: self._top_k]]
            top_scores = [float(s) for _, s in reranked[: self._top_k]]
        else:
            top_indices = [i for i, _ in indexed[: self._top_k]]
            top_chunks = [self._chunks[i] for i in top_indices]
            top_scores = [float(combined[i]) for i in top_indices]

        retrieved: list[RetrievedChunk] = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_name=chunk.document_name,
                rank=rank,
                retrieval_score=round(score, 6),
            )
            for rank, (chunk, score) in enumerate(zip(top_chunks, top_scores), start=1)
        ]

        return QueryRetrieval(
            query_id=query_id,
            question=question,
            retrieved_chunks=retrieved,
        )

    def retrieve_all(self, queries: list[dict]) -> RetrievalResults:
        results = [self.retrieve(q["query_id"], q["question"]) for q in queries]
        return RetrievalResults(results=results)

    def build_metadata(self) -> IndexMetadata:
        if self._bm25_weight > 0 and self._embedding_weight > 0:
            mode = "hybrid"
        elif self._bm25_weight > 0:
            mode = "keyword"
        else:
            mode = "embedding"

        doc_names = sorted({c.document_name for c in self._chunks})
        return IndexMetadata(
            retrieval_mode=mode,
            bm25_weight=self._bm25_weight,
            embedding_weight=self._embedding_weight,
            embedding_model=self._embedding_model_name,
            chunk_count=len(self._chunks),
            document_names=doc_names,
            built_at=datetime.now(timezone.utc).isoformat(),
            reranker_model=(
                self._reranker_model_name if self._reranker_enabled else None
            ),
            reranker_enabled=self._reranker_enabled,
        )
