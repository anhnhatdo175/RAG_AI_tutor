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
                 candidate_k: int = 50, profile_vector: np.ndarray | None = None,
                 profile_mastery: float | None = None,
                 personalized_scores_by_doc: dict[str, float] | None = None,
                 profile_vector_blend: float = config.RAG_PROFILE_VECTOR_BLEND,
                 personalization_confidence: float = 1.0,
                 alpha: float = config.RAG_SEMANTIC_WEIGHT,
                 beta: float = config.RAG_PROFILE_WEIGHT) -> list[dict]:
        if top_k < 1 or candidate_k < top_k:
            raise ValueError("Require candidate_k >= top_k >= 1")
        if not target_subject or not target_grade:
            raise ValueError("target_subject and target_grade are required")
        if (not np.isfinite(alpha) or not np.isfinite(beta) or alpha < 0 or beta < 0
                or not np.isclose(alpha + beta, 1.0)):
            raise ValueError("retrieval alpha and beta must be non-negative and sum to 1")

        profile = None
        if profile_vector is not None:
            profile = np.asarray(profile_vector, dtype="float32").reshape(-1)
            if profile.shape[0] != self.store.vectors.shape[1] or not np.isfinite(profile).all():
                raise ValueError("profile_vector must be finite and match the vector index dimension")
            profile_norm = float(np.linalg.norm(profile))
            if profile_norm <= 1e-12:
                profile = None
            else:
                profile /= profile_norm
        if profile_mastery is not None and (
            isinstance(profile_mastery, bool) or not np.isfinite(profile_mastery)
            or not 0 <= profile_mastery <= 1
        ):
            raise ValueError("profile_mastery must be in [0, 1]")
        if not np.isfinite(profile_vector_blend) or not 0 <= profile_vector_blend <= 1:
            raise ValueError("profile_vector_blend must be in [0, 1]")
        if (not np.isfinite(personalization_confidence)
                or not 0 <= personalization_confidence <= 1):
            raise ValueError("personalization_confidence must be in [0, 1]")
        personalized_scores_by_doc = personalized_scores_by_doc or {}
        if any(not np.isfinite(value) or not 0 <= value <= 1
               for value in personalized_scores_by_doc.values()):
            raise ValueError("personalized document scores must be finite and in [0, 1]")

        rewritten = self.query_rewriter.rewrite(query)
        if not rewritten:
            raise ValueError("query must not be empty")
        candidate_ids = self.store.filter_ids(
            skill_tag=target_skill,
            target_subject=target_subject,
            target_grade=target_grade,
            mapped_only=True,
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
        rerank_scores = np.asarray(rerank_scores, dtype="float64")
        if not np.isfinite(rerank_scores).all():
            raise ValueError("Reranker returned non-finite scores")
        low = float(rerank_scores.min()) if len(rerank_scores) else 0.0
        high = float(rerank_scores.max()) if len(rerank_scores) else 0.0
        semantic_scores = (
            (rerank_scores - low) / (high - low)
            if high - low > 1e-12 else np.full(len(rerank_scores), 0.5)
        )

        results = []
        for (index, dense_score), rerank_score, semantic_score in zip(
            dense, rerank_scores, semantic_scores
        ):
            metadata = self.store.meta[index]
            profile_score = None
            collaborative_target_score = personalized_scores_by_doc.get(str(metadata["doc_id"]))
            adaptive_score = float(semantic_score)
            if profile is not None or collaborative_target_score is not None:
                item_vector = self.store.vectors[index]
                profile_zpd = 0.5
                if profile is not None:
                    profile_zpd = float(np.clip((np.dot(profile, item_vector) + 1.0) / 2.0, 0.0, 1.0))
                    difficulty = metadata.get("difficulty")
                    if profile_mastery is not None and isinstance(difficulty, (int, float)):
                        from lmsrag.layer4_recommender import zpd_score
                        profile_zpd *= zpd_score(float(difficulty) - float(profile_mastery))
                if collaborative_target_score is None:
                    profile_score = profile_zpd
                else:
                    profile_score = (
                        profile_vector_blend * profile_zpd
                        + (1.0 - profile_vector_blend) * collaborative_target_score
                    )
                effective_beta = beta * personalization_confidence
                adaptive_score = (
                    (1.0 - effective_beta) * float(semantic_score)
                    + effective_beta * profile_score
                )
            results.append({
                "chunk_id": metadata["chunk_id"],
                "doc_id": metadata["doc_id"],
                "text": metadata["text"],
                "subject": metadata["subject"],
                "grade": metadata["grade"],
                "skill_tag": metadata["skill_tag"],
                "difficulty": metadata["difficulty"],
                "semantic_score": float(dense_score),
                "rerank_score": float(rerank_score),
                "semantic_rerank_score": float(semantic_score),
                "profile_score": profile_score,
                "collaborative_target_score": collaborative_target_score,
                "adaptive_score": adaptive_score,
            })
        results.sort(key=lambda result: result["adaptive_score"], reverse=True)
        return results[:top_k]


def ranking_metrics(ranked_ids: list[str], relevant_ids: set[str], k: int = 5) -> dict[str, float]:
    """Compute Precision@K, Recall@K, reciprocal rank, and NDCG@K for one query."""
    if k < 1:
        raise ValueError("k must be >= 1")
    hits = [1.0 if item_id in relevant_ids else 0.0 for item_id in ranked_ids[:k]]
    relevant_count = len(relevant_ids)
    precision = sum(hits) / k
    recall = sum(hits) / relevant_count if relevant_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    reciprocal_rank = next((1.0 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    dcg = sum(hit / np.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal_count = min(relevant_count, k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        f"hit@{k}": float(any(hits)),
        f"precision@{k}": precision,
        f"recall@{k}": recall,
        f"f1@{k}": f1,
        f"mrr@{k}": reciprocal_rank,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
    }
