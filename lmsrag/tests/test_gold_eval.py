import hashlib

import pandas as pd
import pytest

from lmsrag.retrieval import gold_eval as G


def test_query_metrics_perfect_and_condensed():
    judged = {"a": 2, "b": 1, "c": 0}
    m = G.query_metrics(["a", "b", "c"], judged)
    assert m["ndcg@5"] == pytest.approx(1.0) and m["judged@5"] == 1.0
    m = G.query_metrics(["x", "y", "a", "b"], judged)  # unjudged first: lower bound drops, condensed does not
    assert m["ndcg@5"] < 1.0 and m["condensed_ndcg@5"] == pytest.approx(1.0) and m["judged@5"] == 0.5


def test_load_gold_checks_hash(tmp_path):
    p = tmp_path / "gold_v1.csv"
    p.write_bytes(b"query_id,chunk_id,label_final\nq1,a,2\n")
    p.with_suffix(".sha256").write_text(hashlib.sha256(p.read_bytes()).hexdigest() + "  gold_v1.csv\n")
    assert len(G.load_gold(p)) == 1
    p.write_bytes(b"query_id,chunk_id,label_final\nq1,a,1\n")
    with pytest.raises(SystemExit):
        G.load_gold(p)


def test_paired_bootstrap_sign():
    rows = [{"view": "runs", "system": s, "query_id": f"q{i}", "ndcg@5": v + (0.1 if s == "a" else 0)}
            for i, v in enumerate([0.2, 0.5, 0.7, 0.4]) for s in ("a", "b")]
    out = G.paired(pd.DataFrame(rows), "runs", [("a", "b")], ["ndcg@5"], 200, 0)
    assert out.loc[0, "diff"] == pytest.approx(0.1) and out.loc[0, "wins"] == 4
