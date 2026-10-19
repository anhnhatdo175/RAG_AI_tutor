"""Tiny vector database: FAISS inner-product index + metadata + *pre-filtered* search."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, vectors: np.ndarray, meta: list[dict]):
        assert len(vectors) == len(meta)
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        self.meta = meta
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)
        self.chunk_pos = {m["chunk_id"]: i for i, m in enumerate(meta)}
        self.ids_by_tag: dict[str, np.ndarray] = {
            t: np.asarray(ix, dtype="int64") for t, ix in self._group("skill_tag").items()}

    def _group(self, key):
        g = defaultdict(list)
        for i, m in enumerate(self.meta):
            g[m[key]].append(i)
        return g

    @classmethod
    def load(cls, vec_path: Path, meta_path: Path) -> "VectorStore":
        meta = [json.loads(l) for l in meta_path.read_text(encoding="utf-8").split("\n") if l.strip()]
        return cls(np.load(vec_path), meta)

    def search(self, qvec: np.ndarray, k: int = 5, where_skill_tag: str | None = None):
        """Top-k by cosine. If ``where_skill_tag`` is given, ONLY chunks whose
        metadata skill_tag equals it are candidates (hard pre-filter, Step 4)."""
        q = np.ascontiguousarray(qvec.reshape(1, -1), dtype="float32")
        if where_skill_tag is None:
            D, I = self.index.search(q, k)
        else:
            ids = self.ids_by_tag.get(where_skill_tag)
            if ids is None or len(ids) == 0:
                return []
            params = faiss.SearchParameters()
            params.sel = faiss.IDSelectorBatch(ids)
            D, I = self.index.search(q, min(k, len(ids)), params=params)
        return [(int(i), float(d)) for d, i in zip(D[0], I[0]) if i != -1]

    # ---- Layer 3 helpers ---------------------------------------------------------------------------------
    def filter_ids(self, skill_tag: str, subjects=None, max_level: int | None = None, content_types=None) -> np.ndarray:
        """C_candidates = {d | skill_tag(d)=target  AND  subject(d) in subjects  AND  level(d) <= max_level}  (docx 2.4.2)."""
        ids = self.ids_by_tag.get(skill_tag, np.array([], dtype="int64"))
        keep = []
        for i in ids:
            m = self.meta[i]
            if subjects and m.get("subject") not in subjects:
                continue
            if max_level is not None and m.get("level") is not None and m["level"] > max_level:
                continue
            if content_types and m["content_type"] not in content_types:
                continue
            keep.append(i)
        return np.asarray(keep, dtype="int64")

    def search_ids(self, qvec: np.ndarray, ids: np.ndarray, k: int):
        if len(ids) == 0:
            return []
        params = faiss.SearchParameters()
        params.sel = faiss.IDSelectorBatch(ids)
        q = np.ascontiguousarray(qvec.reshape(1, -1), dtype="float32")
        D, I = self.index.search(q, min(k, len(ids)), params=params)
        return [(int(i), float(d)) for d, i in zip(D[0], I[0]) if i != -1]
