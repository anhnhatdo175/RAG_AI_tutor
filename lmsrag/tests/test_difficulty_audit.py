import numpy as np
import pandas as pd
import pytest

from lmsrag.labels.agreement import cohen_kappa
from lmsrag.labels.difficulty_agreement import adjudication_sheet, audit, read_sheet
from lmsrag.labels.difficulty_sheets import ALLOCATION, sample


def test_radical_answers_show_their_root_sign():
    from lmsrag.labels.difficulty_sheets import display_answer

    assert display_answer("192", "cuberoot") == "∛(192)"  # = 4∛3 (d066)
    assert display_answer("147", "radical") == "√(147)"  # = 7√3
    assert display_answer("6", "number") == "6"


def test_allocation_matches_card_p1_06():
    assert all(sum(v.values()) == 20 for v in ALLOCATION.values())
    totals = pd.DataFrame(ALLOCATION).T.fillna(0).sum()
    assert totals.to_dict() == {"khan": 50, "gsm8k": 25, "math": 25}
    assert "gsm8k" not in ALLOCATION[4] and "math" not in ALLOCATION[3] and "math" not in ALLOCATION[5]


def _docs():
    rows = []
    for lvl in range(1, 6):
        for src in ("khan", "gsm8k", "math"):
            for j in range(30):
                slug = f"t{lvl}_{j // 2}" if src == "khan" else None  # two problems per template
                rows.append({"doc_id": f"{src}:{lvl}:{j:02d}", "source": src, "difficulty_level": lvl, "khan_slug": slug})
    return pd.DataFrame(rows)


def test_sample_is_deterministic_and_one_problem_per_template():
    a = sample(_docs(), np.random.default_rng(2026))
    b = sample(_docs(), np.random.default_rng(2026))
    assert a["doc_id"].tolist() == b["doc_id"].tolist()
    assert len(a) == 100 and a["doc_id"].is_unique
    assert a.loc[a["source"] == "khan", "khan_slug"].is_unique
    counts = a.groupby(["difficulty_level", "source"]).size().to_dict()
    assert counts == {(lvl, s): n for lvl, d in ALLOCATION.items() for s, n in d.items()}


def test_weighted_kappa_uses_the_full_scale():
    from sklearn.metrics import cohen_kappa_score

    a, b = [1, 2, 4, 4, 2, 1], [2, 2, 4, 2, 1, 1]  # nobody used 3 or 5
    full = cohen_kappa(a, b, "quadratic", [1, 2, 3, 4, 5])
    assert full == pytest.approx(cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3, 4, 5]))
    assert full != pytest.approx(cohen_kappa(a, b, "quadratic"))  # labels seen only: 2 and 4 look adjacent


def _sheet(path, rows):
    pd.DataFrame(rows, columns=["item", "rating", "figure_ok", "answer_ok", "note"]).to_csv(
        path, index=False, encoding="utf-8-sig")
    return path


def test_read_sheet_reports_bad_values(tmp_path):
    p = _sheet(tmp_path / "s.csv", [["d1", "3", "có", "Không", ""], ["d2", "6", "–", "–", ""],
                                    ["d3", "2", "maybe", "", ""], ["d4", "", "có", "–", ""]])
    s, errors = read_sheet(p, khan_items={"d1", "d3"})
    assert s.loc["d1", "figure_ok"] == 1 and s.loc["d1", "answer_ok"] == 0
    assert np.isnan(s.loc["d2", "rating"]) and np.isnan(s.loc["d4", "rating"])
    assert len(errors) == 3
    assert any("d2: rating '6'" in e for e in errors)
    assert any("d3: figure_ok 'maybe'" in e for e in errors)
    assert any("d4: figure_ok is only asked on Khan rows" in e for e in errors)


def _audit_inputs():
    key = pd.DataFrame({"item": [f"d{i}" for i in range(8)],
                        "source": ["khan"] * 4 + ["gsm8k"] * 4,
                        "difficulty_source": ["junyi_history"] * 4 + ["gsm8k_steps"] * 4,
                        "difficulty": [0.1, 0.4, 0.6, 0.9, 0.2, 0.3, 0.7, 0.8],
                        "difficulty_level": [1, 2, 4, 5, 1, 2, 4, 5]})
    idx = key["item"].tolist()
    s2 = pd.DataFrame({"rating": [1, 2, 4, 5, 1, 2, 3, 5], "figure_ok": [1, 1, 0, 1] + [np.nan] * 4,
                       "answer_ok": [1, 1, 1, 1] + [np.nan] * 4, "note": ""}, index=idx)
    s3 = pd.DataFrame({"rating": [1, 3, 4, 5, 2, 2, 3, 4], "figure_ok": [1, 1, 1, 1] + [np.nan] * 4,
                       "answer_ok": [1, 0, 1, 1] + [np.nan] * 4, "note": ""}, index=idx)
    return key, s2, s3


def test_audit_before_and_after_adjudication():
    key, s2, s3 = _audit_inputs()
    t = audit(key, s2, s3, {}, seed=2026).set_index(["level", "group", "metric"])["value"]
    assert t["source", "khan", "spearman_difficulty_vs_mean_rating"] == pytest.approx(1.0)
    assert t["all", "all", "exact_agreement"] == pytest.approx(5 / 8)  # items 0, 2, 3, 5, 6
    assert t["all", "all", "within_one_agreement"] == 1.0 and t["all", "all", "rating_gap_ge2"] == 0
    assert t["level_source", "L1 all", "mean_rating"] == pytest.approx((1 + 1 + 1 + 2) / 4)
    assert t["khan_filter", "figure_ok", "pairs_to_adjudicate"] == 1
    assert ("khan_filter", "figure_ok", "accuracy_after_adjudication") not in t.index

    final = {("d2", "figure_ok"): "không", ("d1", "answer_ok"): "có"}
    t = audit(key, s2, s3, final, seed=2026).set_index(["level", "group", "metric"])["value"]
    assert t["khan_filter", "figure_ok", "accuracy_after_adjudication"] == 0.75
    assert t["khan_filter", "answer_ok", "accuracy_after_adjudication"] == 1.0
    assert t["khan_filter", "both", "accuracy_after_adjudication"] == 0.75


def test_adjudication_sheet_keeps_earlier_decisions():
    key, s2, s3 = _audit_inputs()
    sheet = pd.DataFrame({"problem": "p", "answer": "a"}, index=key["item"])
    old = pd.DataFrame({"item": ["d2"], "check": ["figure_ok"], "final": ["không"], "note_final": ["cần hình"]})
    adj = adjudication_sheet(key, sheet, s2, s3, old)
    assert adj[["item", "check"]].values.tolist() == [["d1", "answer_ok"], ["d2", "figure_ok"]]
    assert adj.set_index("item").loc["d2", "final"] == "không"
    assert adj.set_index("item").loc["d1", "final"] == ""
