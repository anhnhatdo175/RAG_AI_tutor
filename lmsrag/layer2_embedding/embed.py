"""
LAYER 2 / stage 4 - EMBEDDING.  v_d = f_embed(d) in R^dim with ||v_d||_2 = 1 (docx 2.3.2).

Wraps an embedder back-end, enforces the unit-norm contract and reports geometry diagnostics so a bad embedding space is
detected before it reaches the retriever.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..embedder import LsaEmbedder, make_embedder


def assert_unit_norm(V: np.ndarray, tol: float = 1e-4) -> dict:
    n = np.linalg.norm(V, axis=1)
    bad = int((~np.isfinite(V).all(1)).sum() + (np.abs(n - 1) > tol).sum())
    if bad:
        raise ValueError(f"{bad} vectors violate ||v||=1 (min={n.min():.4f}, max={n.max():.4f})")
    return {"norm_min": float(n.min()), "norm_max": float(n.max()), "dim": int(V.shape[1]), "n": int(V.shape[0])}


class EmbeddingLayer:
    def __init__(self, kind: str = "lsa"):
        self.kind = kind
        self.model = make_embedder(kind)

    def fit_encode_docs(self, texts: list[str]) -> np.ndarray:
        self.model.fit(texts)
        V = self.model.encode(texts)
        self.norm_report = assert_unit_norm(V)
        return V

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.model.encode_queries(texts)

    def fingerprint(self, texts: list[str]) -> str:
        return hashlib.sha1(f"{self.kind}|{len(texts)}|{texts[0][:200]}|{texts[-1][:200]}".encode()).hexdigest()[:12]

    def save(self, path):
        if isinstance(self.model, LsaEmbedder):
            self.model.save(path)


def geometry_report(V: np.ndarray, tags: np.ndarray, rng_seed: int = 0, n_pairs: int = 20000) -> dict:
    """Is the space usable?  random-pair cosine (anisotropy), intra- vs inter-skill cosine (separation), effective rank."""
    rng = np.random.default_rng(rng_seed)
    i, j = rng.integers(0, len(V), n_pairs), rng.integers(0, len(V), n_pairs)
    cos = (V[i] * V[j]).sum(1)
    same = (tags[i] == tags[j]) & (tags[i] != "unmapped") & (i != j)
    diff = (tags[i] != tags[j]) & (tags[i] != "unmapped") & (tags[j] != "unmapped")
    s = np.linalg.svd(V[rng.choice(len(V), min(3000, len(V)), replace=False)] - V.mean(0), compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    return {"random_pair_cos_mean": float(cos.mean()), "random_pair_cos_std": float(cos.std()),
            "intra_skill_cos_mean": float(cos[same].mean()) if same.any() else None,
            "inter_skill_cos_mean": float(cos[diff].mean()) if diff.any() else None,
            "effective_rank": float(np.exp(-(p * np.log(p + 1e-12)).sum()))}
