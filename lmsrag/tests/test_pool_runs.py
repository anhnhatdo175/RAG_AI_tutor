import numpy as np
import pandas as pd
import pytest

from lmsrag.retrieval import pool_runs as pr


def _corpus():
    rows = [("c1", ["S1"], "add two fractions with like denominators", 0.20, 1),
            ("c2", ["S1"], "add fractions unlike denominators common denominator", 0.55, 3),
            ("c3", ["S1", "S2"], "fractions word problem pizza slices", 0.80, 5),
            ("c4", ["S2"], "mean of a list of numbers", 0.40, 2)]
    return pd.DataFrame(rows, columns=["chunk_id", "skill_tags", "text", "difficulty", "difficulty_level"])


def _queries():
    return pd.DataFrame({"query_id": ["q1", "q2"], "skill_tag": ["S1", "S2"], "mastery": [0.2, 0.5],
                         "target_level": [2, 4], "query": ["fractions denominators", "mean numbers"]})


def test_candidates_follow_skill_tags():
    cand = pr.candidates(_queries(), _corpus())
    assert cand["q1"].tolist() == [0, 1, 2] and cand["q2"].tolist() == [2, 3]


def test_top_is_deterministic_on_ties():
    t = pr.top("q", np.array(["b", "a", "c"]), np.array([1.0, 1.0, 2.0]), 3)
    assert t["chunk_id"].tolist() == ["c", "a", "b"] and t["rank"].tolist() == [1, 2, 3]


def test_bm25_prefers_matching_terms_and_idf_is_non_negative():
    w, q = _corpus(), _queries()
    cand = pr.candidates(q, w)
    run = pr.run_bm25(q, w, cand, dict(zip(q.query_id, q["query"])), depth=10)
    pr.check_run(run, q, w, cand, 10)
    assert run[run.query_id == "q2"].iloc[0]["chunk_id"] == "c4"
    assert run[run.query_id == "q1"].iloc[-1]["chunk_id"] == "c3"  # no 'denominators'
    bm = pr.BM25([["a"], ["a"], ["a", "b"]])
    assert min(bm.idf.values()) >= 0


def test_adaptive_old_matches_old_formula():
    ce = np.array([0.0, 1.0, 2.0])
    cos = np.array([1.0, 1.0, 1.0])  # profile term reduces to zpd_score
    s = pr.adaptive_old_scores(ce, cos, np.array([0.30, 0.90, 0.45]), mastery=0.2, beta=0.65)
    # deltas 0.10 -> 1.0, 0.70 -> 0.2, 0.25 -> 1.0
    np.testing.assert_allclose(s, 0.35 * np.array([0, 0.5, 1]) + 0.65 * np.array([1.0, 0.2, 1.0]))


class _Enc:
    def encode(self, texts):
        v = np.array([[len(t), t.count("fraction") + 1.0, 1.0] for t in texts], dtype="float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    encode_queries = encode


class _CE:
    def score(self, query, docs):
        return [float(len(set(query.split()) & set(d.split()))) for d in docs]


def test_dense_systems_with_stub_models(monkeypatch):
    monkeypatch.setattr(pr, "load_models", lambda: (_Enc(), _CE()))
    w, q = _corpus(), _queries()
    cand = pr.candidates(q, w)
    runs = pr.run_dense(q, w, cand, dict(zip(q.query_id, q["query"])), 10, 50, ["bge_ce", "adaptive_old"])
    for r in runs.values():
        pr.check_run(r, q, w, cand, 10)
    assert set(runs) == {"bge_ce", "adaptive_old"}


def test_check_run_rejects_chunk_of_other_skill():
    w, q = _corpus(), _queries()
    cand = pr.candidates(q, w)
    bad = pd.DataFrame({"query_id": ["q1", "q2"], "chunk_id": ["c4", "c4"], "rank": [1, 1], "score": [1.0, 1.0]})
    with pytest.raises(AssertionError):
        pr.check_run(bad, q, w, cand, 1)
