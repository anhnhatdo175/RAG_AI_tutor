"""LAYER 2 / stage 5 - INDEXING.  Persist FAISS index + metadata + manifest; verify on load."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter

import faiss
import numpy as np

from .. import config
from ..vectorstore import VectorStore

SCHEMA_VERSION = 3                       # 3: + `grade`
REQUIRED_META = ("chunk_id", "doc_id", "text", "skill_tag", "difficulty", "content_type", "source", "token_count", "grade")


def save_index(chunks: list[dict], V: np.ndarray, embedder_info: dict) -> dict:
    for c in chunks:
        missing = [k for k in REQUIRED_META if k not in c]
        if missing:
            raise ValueError(f"chunk {c.get('chunk_id')} misses metadata {missing}")
    assert len(chunks) == len(V)
    np.save(config.EMB_NPY, V.astype("float32"))
    body = "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks)
    config.CHUNKS_JSONL.write_text(body, encoding="utf-8")
    index = faiss.IndexFlatIP(V.shape[1]); index.add(np.ascontiguousarray(V, dtype="float32"))
    faiss.write_index(index, str(config.ART / "index.faiss"))
    manifest = {"schema_version": SCHEMA_VERSION, "created_unix": int(time.time()), "n_chunks": len(chunks), "dim": int(V.shape[1]),
                "embedder": embedder_info, "chunks_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "by_source": dict(Counter(c["source"] for c in chunks)),
                "by_content_type": dict(Counter(c["content_type"] for c in chunks)),
                "by_difficulty": {str(k): v for k, v in sorted(Counter(c["difficulty"] for c in chunks).items())},
                "by_grade": dict(Counter(c["grade"] for c in chunks)),
                "n_skill_tags": len({c["skill_tag"] for c in chunks} - {"unmapped"}),
                "n_unmapped": sum(c["skill_tag"] == "unmapped" for c in chunks)}
    (config.ART / "index_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_and_verify() -> tuple[VectorStore, dict]:
    man = json.loads((config.ART / "index_manifest.json").read_text())
    body = config.CHUNKS_JSONL.read_text(encoding="utf-8")
    if hashlib.sha256(body.encode()).hexdigest() != man["chunks_sha256"]:
        raise RuntimeError("chunks file changed since the index was built - rebuild the index")
    store = VectorStore.load(config.EMB_NPY, config.CHUNKS_JSONL)
    assert len(store.meta) == man["n_chunks"] and store.vectors.shape[1] == man["dim"]
    return store, man


def self_retrieval_check(store: VectorStore, n: int = 300, seed: int = 0) -> float:
    """Every chunk must be its own nearest neighbour (catches index/metadata misalignment). A different chunk is accepted
    only if its vector is identical (twin chunks that differ only in tokens the embedder ignores, e.g. rare numbers)."""
    rng = np.random.default_rng(seed)
    ok = 0
    ids = rng.choice(len(store.meta), n, replace=False)
    for i in ids:
        top = store.search(store.vectors[i], k=1)[0][0]
        ok += int(top == i or np.allclose(store.vectors[top], store.vectors[i], atol=1e-6))
    return ok / n
