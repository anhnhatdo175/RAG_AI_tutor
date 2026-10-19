"""Embedding back-ends. All return L2-normalised float32 matrices (so inner product = cosine).

* ``bge``   : BAAI/bge-small-en-v1.5 via sentence-transformers (what the original repo used).
              Needs the HuggingFace hub, i.e. run it on your own machine / Colab.
* ``lsa``   : TF-IDF (1-2 grams) + truncated SVD. Fully offline, deterministic, no downloads.
              Used for the numbers produced inside the sandbox where HF is blocked.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class LsaEmbedder:
    name = "lsa"

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim, self.seed = dim, seed
        self.tfidf = self.svd = None

    def fit(self, texts: list[str]) -> "LsaEmbedder":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(token_pattern=r"(?u)\\?[A-Za-z]{2,}|\d+", ngram_range=(1, 2), min_df=3,
                                     max_features=80000, sublinear_tf=True, lowercase=True)
        x = self.tfidf.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=min(self.dim, x.shape[1] - 1), random_state=self.seed).fit(x)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return _l2(self.svd.transform(self.tfidf.transform(texts)))

    encode_queries = encode

    def save(self, path: Path):
        path.write_bytes(pickle.dumps((self.tfidf, self.svd, self.dim)))

    @classmethod
    def load(cls, path: Path) -> "LsaEmbedder":
        o = cls(); o.tfidf, o.svd, o.dim = pickle.loads(path.read_bytes()); return o


class BgeEmbedder:
    name = "bge"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)

    def fit(self, texts):  # stateless
        return self

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "      # bge retrieval convention

    def encode(self, texts: list[str]) -> np.ndarray:
        return _l2(self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False))

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode([self.QUERY_PREFIX + t for t in texts])


def make_embedder(kind: str):
    return BgeEmbedder() if kind == "bge" else LsaEmbedder()
