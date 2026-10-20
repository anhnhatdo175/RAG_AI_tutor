import numpy as np
import pytest

from lmsrag.layer2_embedding import chunk, clean, embed, index, parse
from lmsrag.layer2_embedding.profile_vector import DecayedProfile, profile_vector_direct


def test_chunker_respects_budget_and_overlap():
    text = "\n\n".join(" ".join(f"w{p}_{i}" for i in range(250)) + "." for p in range(8))
    ch = chunk.chunk_text(text, header="[Problem: x ...]")
    assert len(ch) > 1 and all(chunk.n_tokens(c) <= chunk.MAX_TOKENS for c in ch)
    # overlap: the first words of chunk i+1 (after the header) come from the end of chunk i
    tail_prev = set(ch[0].split()[-20:])
    assert tail_prev & set(ch[1].split()[:80])
    assert chunk.chunk_text("short text") == ["short text"]


def test_clean_normalises_and_dedups():
    assert clean.normalize("a <<2+2=4>>4\u2028 b\u00a0c") == "a 4 b c"
    mk = lambda i, q: parse.ParsedRecord(f"d{i}", "T", None, None, q, "sol", "1", 0.3)
    recs = [mk(1, "What is the sum of two and three today?"), mk(2, "what is the SUM of two and three today?"),
            mk(3, "Find x. [asy] draw(); [/asy] as shown"), mk(4, "x"),
            mk(5, "A completely different question about triangles and angles.")]
    kept, rep = clean.clean_records(recs)
    assert [r.doc_id for r in kept] == ["d1", "d5"]
    assert rep["dropped_exact_duplicate"] == 1 and rep["dropped_figure_dependent"] == 1


def test_unit_norm_contract():
    V = np.random.default_rng(0).normal(size=(10, 8)).astype("float32")
    with pytest.raises(ValueError):
        embed.assert_unit_norm(V)
    embed.assert_unit_norm(V / np.linalg.norm(V, axis=1, keepdims=True))


def test_profile_vector_incremental_equals_direct():
    rng = np.random.default_rng(1)
    T, d = 40, 16
    vs = rng.normal(size=(T, d)); vs /= np.linalg.norm(vs, axis=1, keepdims=True)
    c = rng.integers(0, 2, T).astype(float); tau = rng.uniform(2000, 90000, T)
    direct = profile_vector_direct(vs, c, tau, tau_base=20000.0)
    dp = DecayedProfile(d)
    for t in range(T):
        dp.update(vs[t], c[t], tau[t], 20000.0)
    assert np.allclose(direct, dp.vector, atol=1e-9) and abs(np.linalg.norm(dp.vector) - 1) < 1e-9


def test_recency_decay_prefers_recent():
    a, b = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    p = profile_vector_direct(np.stack([a, b]), np.array([1.0, 1.0]), np.array([5000.0, 5000.0]), 5000.0)
    assert p[1] > p[0]


def test_index_alignment(tmp_path, monkeypatch):
    from lmsrag import config

    monkeypatch.setattr(config, "ART", tmp_path)
    monkeypatch.setattr(config, "EMB_NPY", tmp_path / "chunk_embeddings.npy")
    monkeypatch.setattr(config, "CHUNKS_JSONL", tmp_path / "corpus_chunks.jsonl")
    chunks = [
        {
            "chunk_id": f"c{i}", "doc_id": f"d{i}", "text": f"text {i}", "skill_tag": "fractions",
            "canonical_skill_id": "fractions", "mapping_status": "mapped", "difficulty": 0.3,
            "content_type": "worked_example", "source": "test", "subject": "Mathematics",
            "grade": "lower_secondary", "token_count": 2,
        }
        for i in range(3)
    ]
    vectors = np.eye(3, dtype="float32")
    index.save_index(chunks, vectors, {"kind": "test", "model": "test", "fingerprint": "test"})
    store, man = index.load_and_verify()
    assert man["schema_version"] == index.SCHEMA_VERSION
    assert index.self_retrieval_check(store, n=3) == 1.0
    assert max(m["token_count"] for m in store.meta) <= chunk.MAX_TOKENS


def test_grade_metadata_rules():
    from lmsrag import config
    from lmsrag.layer2_embedding.parse import ParsedRecord, infer_grade, infer_subject
    mk = lambda **kw: ParsedRecord("x", kw.pop("source", "MATH"), kw.pop("subject", None),
                                   kw.pop("level", None), "q", "s", "a", 0.6, **kw)
    assert infer_grade(mk(source="GSM8K", is_gsm8k=True)) == config.GRADE_GSM8K
    assert infer_grade(mk(level=1)) == "lower_secondary" and infer_grade(mk(level=5)) == "upper_secondary"
    assert infer_grade(mk(source="Khan")) == config.GRADE_DEFAULT
    assert infer_grade(mk(source="Khan", extra={"grade": "G7"})) == "lower_secondary"
    assert infer_grade(mk(source="Khan", extra={"grade": "Grade 10"})) == "upper_secondary"
    assert infer_subject(mk(source="GSM8K", subject="word_problem")) == "Mathematics"
    assert infer_subject(mk(source="Khan", subject="math")) == "Mathematics"
    assert infer_subject(mk(source="Khan_Academy_export", subject="khan:fractions")) == "Mathematics"
    assert "grade" in index.REQUIRED_META


def test_vectorstore_exact_subject_grade_filter():
    from lmsrag.vectorstore import VectorStore

    vectors = np.eye(3, dtype="float32")
    meta = [
        {"chunk_id": "a", "skill_tag": "fractions", "subject": "Mathematics", "grade": "lower_secondary",
         "content_type": "worked_example", "level": 1},
        {"chunk_id": "b", "skill_tag": "fractions", "subject": "Mathematics", "grade": "upper_secondary",
         "content_type": "worked_example", "level": 4},
        {"chunk_id": "c", "skill_tag": "algebra", "subject": "English", "grade": "lower_secondary",
         "content_type": "worked_example", "level": 1},
    ]
    store = VectorStore(vectors, meta)
    ids = store.filter_ids(target_subject="Mathematics", target_grade="lower_secondary")
    assert ids.tolist() == [0]
    ids = store.filter_ids("fractions", target_subject="Mathematics", target_grade="upper_secondary")
    assert ids.tolist() == [1]
