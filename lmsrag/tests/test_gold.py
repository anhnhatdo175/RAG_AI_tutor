import numpy as np
import pandas as pd
import pytest

from lmsrag.labels.gold_pool import build_pool, chunk_content, load_runs, make_pools, round_robin
from lmsrag.labels.gold_queries import TEMPLATE, choose_skills, target_level

G = {"systems": ["a", "b"], "top_k": 3, "random_same_skill": 2, "random_other_skill": 1, "pool_min": 6,
     "pool_max": 8, "fill_depth": 5}


def test_target_level_rule_of_rubric_v1():
    assert [target_level(m) for m in (0.2, 0.35, 0.5, 0.65, 0.8)] == [2, 3, 4, 4, 5]
    assert target_level(0.0) == 1 and target_level(1.0) == 5
    assert TEMPLATE.format(mastery=0.35, skill="Mean", level=3) == (
        "A student with mastery 0.35 on Mean needs a worked example at difficulty level 3 of 5.")


def test_choose_skills_needs_three_levels_and_ranks_by_2009_rows():
    docs = pd.DataFrame({"doc_id": list("abcdefgh"), "skill_tag": ["S1"] * 3 + ["S2"] * 2 + ["S3"] * 3,
                         "difficulty_level": [1, 2, 3, 1, 5, 2, 3, 4]})
    rows = {"assist2009": pd.Series({"S1": 10, "S2": 999}), "assist2012": pd.Series({"S3": 50})}
    got = choose_skills(docs, rows, n=2)
    assert got.index.tolist() == ["S1", "S3"]  # S2 has 2 levels only; S3 fills from 2012
    assert got["skill_source"].tolist() == ["assist2009", "assist2012"]
    with pytest.raises(SystemExit):
        choose_skills(docs, rows, n=3)


def _runs(n_a=5, n_b=5):
    rows = [("q1", f"a{i}", i, "a") for i in range(1, n_a + 1)] + [("q1", f"b{i}", i, "b") for i in range(1, n_b + 1)]
    rows.append(("q1", "shared", 2, "b"))  # b's rank-2 slot is taken by b2 already; a second rank-2 hit is allowed
    return pd.DataFrame(rows, columns=["query_id", "chunk_id", "rank", "system"])


def test_round_robin_goes_rank_by_rank():
    rr = round_robin(_runs(), ["a", "b"], 1, 2)
    assert [c for c, _, _ in rr] == ["a1", "b1", "a2", "b2", "shared"]


def test_pool_keeps_checks_caps_size_and_records_provenance():
    same = [f"s{i}" for i in range(10)]
    pool = build_pool(pd.Series({"query_id": "q1"}), _runs(), same, ["o1", "o2"], G, np.random.default_rng(0))
    ids = [e["chunk_id"] for e in pool]
    assert len(ids) == len(set(ids)) == G["pool_max"]
    origins = {e["chunk_id"]: e["origins"] for e in pool}
    assert sum("random_same_skill" in o for o in origins.values()) == 2
    assert sum("random_other_skill" in o for o in origins.values()) == 1
    found = {e["chunk_id"]: e["found_by"] for e in pool if e["found_by"]}
    assert found["a1"] == {"a": 1} and found["b1"] == {"b": 1}  # rank 1 of every system is in
    assert "a3" not in found and "b3" not in found  # the cap cut rank 3 (5 system slots after 3 checks)


def test_small_pool_is_filled_from_deeper_ranks_then_random():
    g = {**G, "pool_min": 12, "pool_max": 15}
    pool = build_pool(pd.Series({"query_id": "q1"}), _runs(n_a=4, n_b=3), [f"s{i}" for i in range(3)], ["o1"], g,
                      np.random.default_rng(0))
    origins = [o for e in pool for o in e["origins"]]
    assert "fill" in origins  # a4 comes from rank 4
    assert len(pool) == 3 + 1 + 4 + 4  # every candidate used, short of pool_min: the skill has no more chunks


def test_make_pools_is_deterministic():
    chunks = pd.DataFrame({"chunk_id": [f"s{i}" for i in range(6)] + ["o1", "o2"] + [f"a{i}" for i in range(1, 6)]
                           + [f"b{i}" for i in range(1, 6)] + ["shared"]})
    chunks["skill_tags"] = [["T"]] * 6 + [["U"]] * 2 + [["T"]] * 11
    q = pd.DataFrame({"query_id": ["q1"], "skill_tag": ["T"]})
    p1, prov = make_pools(q, _runs(), chunks, G, seed=2026)
    p2, _ = make_pools(q, _runs(), chunks, G, seed=2026)
    assert p1.equals(p2) and p1["position"].tolist() == list(range(1, len(p1) + 1))
    assert set(prov.columns) >= {"rank_a", "rank_b", "origins"}


def test_load_runs_rejects_unknown_ids(tmp_path):
    pd.DataFrame({"query_id": ["q1", "q9"], "chunk_id": ["c1", "zz"], "rank": [1, 2], "score": [1.0, 0.5]}).to_csv(
        tmp_path / "a.csv", index=False)
    with pytest.raises(SystemExit) as e:
        load_runs(tmp_path, ["a", "b"], {"q1"}, {"c1"})
    msg = str(e.value)
    assert "missing run" in msg and "1 query ids" in msg and "1 chunk ids" in msg


def test_chunk_content_restores_root_sign():
    assert chunk_content("Problem: $\\sqrt{147}$\nAnswer: 147", "147", "radical").endswith("Answer: √(147)")
    assert chunk_content("Problem: 1+1\nAnswer: 2", "2", "number") == "Problem: 1+1\nAnswer: 2"


def _stage():
    return pd.DataFrame({"label_1": [2.0, 1.0, 0.0, 2.0], "label_3": [2.0, 2.0, 0.0, 1.0], "note_1": "", "note_3": ""},
                        index=pd.Index(["g1", "g2", "g3", "g4"], name="item"))


def test_adjudication_finalize_and_freeze(tmp_path):
    from lmsrag.labels.gold_finalize import adjudication, finalize, freeze

    items = pd.DataFrame({"item": ["g1", "g2", "g3", "g4"], "query_id": ["q1", "q1", "q2", "q2"],
                          "chunk_id": ["c1", "c2", "c1", "c3"]})
    sheet = pd.DataFrame({"query_id": items["query_id"].tolist(), "skill": "S", "target_level": 3, "content": "x"},
                         index=pd.Index(items["item"], name="item"))
    old = pd.DataFrame({"item": ["g2"], "label_final": ["2"], "note_final": ["ok"]})
    adj = adjudication(_stage(), items, sheet, old)
    assert adj["item"].tolist() == ["g2", "g4"] and adj["label_final"].tolist() == ["2", ""]
    assert finalize(_stage(), items, adj) is None  # g4 not decided yet
    adj.loc[adj["item"] == "g4", "label_final"] = "1"
    gold = finalize(_stage(), items, adj)
    assert gold["label_final"].tolist() == [2, 2, 0, 1] and gold["adjudicated"].tolist() == [False, True, False, True]
    prereg = tmp_path / "prereg" / "hypotheses.md"
    digest = freeze(gold, tmp_path, prereg)
    assert digest in (tmp_path / "gold_v1.sha256").read_text() and digest in prereg.read_text(encoding="utf-8")
    assert (tmp_path / "gold_v1.sha256").read_bytes() == f"{digest}  gold_v1.csv\n".encode()  # LF: sha256sum -c
    assert freeze(gold, tmp_path, prereg) == digest  # same labels: allowed, prereg not duplicated
    assert prereg.read_text(encoding="utf-8").count(digest) == 1
    changed = gold.assign(label_final=[2, 2, 0, 2])
    with pytest.raises(SystemExit):
        freeze(changed, tmp_path, prereg)


def test_unlabelled_pairs_block_the_freeze():
    from lmsrag.labels.gold_finalize import finalize

    s = _stage()
    s.loc["g1", "label_3"] = np.nan
    items = pd.DataFrame({"item": s.index, "query_id": "q1", "chunk_id": ["c1", "c2", "c3", "c4"]})
    adj = pd.DataFrame({"item": ["g2", "g4"], "label_final": ["2", "1"]})
    assert finalize(s, items, adj) is None


def test_weighted_kappa_on_the_0_1_2_scale():
    from sklearn.metrics import cohen_kappa_score

    from lmsrag.labels.gold_finalize import wkappa

    a, b = [0, 2, 2, 0, 2], [0, 2, 0, 0, 2]  # nobody used 1
    assert wkappa(a, b) == pytest.approx(cohen_kappa_score(a, b, weights="quadratic", labels=[0, 1, 2]))
