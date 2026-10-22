"""Metadata-constrained dense retrieval followed by Cross-Encoder reranking."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .. import config
from ..embedder import BgeEmbedder, LsaEmbedder
from ..vectorstore import VectorStore
from .query import QueryRewriter


class QueryEncoder(Protocol):
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Cross-Encoder adapter. Model weights are loaded only when this object is created."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 batch_size: int = 32):
        import os

        os.environ.setdefault("USE_TF", "0")
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [(query, document) for document in documents]
        logits = np.asarray(self.model.predict(pairs, batch_size=self.batch_size), dtype="float64").reshape(-1)
        if not np.isfinite(logits).all():
            raise ValueError("Cross-Encoder returned non-finite scores")
        return logits.tolist()


def load_query_encoder(embedder: str | None = None) -> QueryEncoder:
    """Load an encoder matching the embedding kind recorded for the current index."""
    import json

    manifest_path = config.ART / "index_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Vector index manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    indexed_kind = manifest.get("embedder", {}).get("kind")
    if indexed_kind not in {"lsa", "bge"}:
        raise ValueError(f"Unsupported embedder in vector index manifest: {indexed_kind!r}")
    if embedder is not None and embedder != indexed_kind:
        raise ValueError(f"Requested embedder {embedder!r} does not match index embedder {indexed_kind!r}")
    embedder = indexed_kind
    if embedder == "lsa":
        path = config.ART / "embedder_lsa.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Saved LSA embedder not found: {path}")
        return LsaEmbedder.load(path)
    if embedder == "bge":
        return BgeEmbedder()
    raise ValueError(f"Unsupported embedder: {embedder}")


class AdaptiveRetriever:
    def __init__(self, store: VectorStore, encoder: QueryEncoder,
                 reranker: Reranker, query_rewriter: QueryRewriter | None = None):
        self.store = store
        self.encoder = encoder
        self.reranker = reranker
        self.query_rewriter = query_rewriter or QueryRewriter()

    def retrieve(self, query: str, *, target_subject: str, target_grade: str,
                 target_skill: str | None = None, top_k: int = 5,
                 candidate_k: int = 50) -> list[dict]:
        if top_k < 1 or candidate_k < top_k:
            raise ValueError("Require candidate_k >= top_k >= 1")
        if not target_subject or not target_grade:
            raise ValueError("target_subject and target_grade are required")

        rewritten = self.query_rewriter.rewrite(query)
        if not rewritten:
            raise ValueError("query must not be empty")
        candidate_ids = self.store.filter_ids(
            skill_tag=target_skill,
            target_subject=target_subject,
            target_grade=target_grade,
        )
        if not len(candidate_ids):
            return []

        qvec = np.asarray(self.encoder.encode_queries([rewritten]), dtype="float32")
        if qvec.ndim != 2 or qvec.shape[0] != 1 or qvec.shape[1] != self.store.vectors.shape[1]:
            raise ValueError("Query encoder output shape does not match the vector index")
        if not np.isfinite(qvec).all():
            raise ValueError("Query encoder returned non-finite values")
        query_norm = float(np.linalg.norm(qvec[0]))
        if query_norm <= 1e-12:
            raise ValueError("Query encoder returned a zero vector")
        qvec[0] /= query_norm

        dense = self.store.search_ids(qvec[0], candidate_ids, candidate_k)
        documents = [self.store.meta[index]["text"] for index, _ in dense]
        rerank_scores = self.reranker.score(rewritten, documents)
        if len(rerank_scores) != len(dense):
            raise ValueError("Reranker must return one score per candidate document")

        results = []
        for (index, semantic_score), rerank_score in zip(dense, rerank_scores):
            metadata = self.store.meta[index]
            results.append({
                "chunk_id": metadata["chunk_id"],
                "doc_id": metadata["doc_id"],
                "text": metadata["text"],
                "subject": metadata["subject"],
                "grade": metadata["grade"],
                "skill_tag": metadata["skill_tag"],
                "difficulty": metadata["difficulty"],
                "semantic_score": semantic_score,
                "rerank_score": float(rerank_score),
            })
        results.sort(key=lambda result: result["rerank_score"], reverse=True)
        return results[:top_k]


def ranking_metrics(ranked_ids: list[str], relevant_ids: set[str], k: int = 5) -> dict[str, float]:
    """Compute Precision@K, Recall@K, reciprocal rank, and NDCG@K for one query."""
    if k < 1:
        raise ValueError("k must be >= 1")
    hits = [1.0 if item_id in relevant_ids else 0.0 for item_id in ranked_ids[:k]]
    relevant_count = len(relevant_ids)
    precision = sum(hits) / k
    recall = sum(hits) / relevant_count if relevant_count else 0.0
    reciprocal_rank = next((1.0 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    dcg = sum(hit / np.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal_count = min(relevant_count, k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"mrr@{k}": reciprocal_rank,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
    }
