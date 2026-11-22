import json

import numpy as np
import pandas as pd
import pytest

from lmsrag.corpus import difficulty as D
from lmsrag.corpus.build_corpus import _py, exercise_bank, make_chunks


def junyi_rows(slug, key, n, correct, split):
    return pd.DataFrame({"item_id": [f"{slug}:{key}"] * n, "khan_slug": slug, "correct": correct, "split": split})


def test_p_correct_hist_uses_history_rows_only():
    inter = pd.concat([junyi_rows("frac", "0", 300, 1, "history"),
                       junyi_rows("frac", "0", 500, 0, "validation"),
                       junyi_rows("frac", "0", 500, 0, "test")])
    unit, template = D.junyi_history_counts(inter)
    assert int(unit["n"].sum()) == 300 and int(template["n"].sum()) == 300
    out = D.khan_history_difficulty(pd.DataFrame({"slug": ["frac"], "type_key": ["0"]}), unit, template, 20, 200)
    assert out["p_correct_hist"].iloc[0] == pytest.approx(1.0)
    assert out["difficulty_n"].iloc[0] == 300 and out["difficulty_unit"].iloc[0] == "problem_type"


def test_shrinkage_template_fallback_and_missing():
    inter = pd.concat([junyi_rows("t", "a", 200, [1] * 100 + [0] * 100, "history"),
                       junyi_rows("t", "b", 50, 1, "history"),
                       junyi_rows("u", "a", 120, 1, "history")])
    unit, template = D.junyi_history_counts(inter)
    probs = pd.DataFrame({"slug": ["t", "t", "u", "zzz"], "type_key": ["a", "b", "a", "0"]})
    out = D.khan_history_difficulty(probs, unit, template, m=20, min_n=200)
    p_t = 150 / 250
    assert out["p_correct_hist"].iloc[0] == pytest.approx((100 + 20 * p_t) / (200 + 20))   # unit, shrunk
    assert out["p_correct_hist"].iloc[1] == pytest.approx(p_t)                              # unit < 200 -> template
    assert out["difficulty_unit"].iloc[1] == "template"
    assert np.isnan(out["p_correct_hist"].iloc[2]) and out["difficulty_n"].iloc[2] == 0     # template < 200
    assert np.isnan(out["p_correct_hist"].iloc[3])                                          # no data


def test_percentile_within_source_and_quintiles():
    v = pd.Series([1, 2, 2, 3, 10, 20, 30, 40])
    g = pd.Series(["a", "a", "a", "a", "b", "b", "b", "b"])
    p = D.percentile_within(v, g)
    assert p.between(0, 1, inclusive="neither").all()
    assert p[1] == p[2]                                  # ties share a value
    assert p[0] == p[4] and p[3] == p[7]                 # lowest/highest of each source map to the same value
    levels = D.quintile_levels(pd.Series(np.linspace(0.01, 0.99, 100)))
    assert levels.value_counts().sort_index().tolist() == [20, 20, 20, 20, 20]


def test_gsm8k_operations_and_isotonic_is_monotone():
    assert D.gsm8k_operations("He has <<2*3=6>>6 and <<6+1=7>>7.\n#### 7") == 2
    model = D.fit_steps_model([1, 2, 3, 4, 5, 6], [0.3, 0.1, 0.2, 0.5, 0.4, 0.6])
    pred = model.predict(np.arange(1, 10, dtype=float))
    assert (np.diff(pred) >= 0).all()


def docs_frame():
    return pd.DataFrame([
        {"doc_id": "khan:t:0:37", "source": "khan", "khan_slug": "t", "problem_type": "0", "difficulty": 0.4,
         "difficulty_level": np.int64(2), "difficulty_source": "junyi_history", "difficulty_n": np.int64(300),
         "answer": "3", "answer_type": "number", "p_correct_hist": 0.61, "skill_tags": ["fractions", "ratio"],
         "question": "What is 1 + 2?", "solution": "Add.\nIt is 3.", "hints": ["Add.", "It is 3."]},
        {"doc_id": "gsm8k:test:4", "source": "gsm8k", "khan_slug": None, "problem_type": None, "difficulty": 0.9,
         "difficulty_level": np.int64(5), "difficulty_source": "gsm8k_steps", "difficulty_n": np.int64(0),
         "answer": "72", "answer_type": "number", "p_correct_hist": np.nan, "skill_tags": ["rate"],
         "question": " ".join(["word"] * 400), "solution": " ".join(["step."] * 400), "hints": []},
    ])


def test_make_chunks_ids_split_and_hint_steps():
    count = lambda text: len(text.split())  # word counter stands in for the tokenizer
    chunks, info = make_chunks(docs_frame(), count)
    ids = [c["chunk_id"] for c in chunks]
    assert ids[:3] == ["khan:t:0:37#0", "khan:t:0:37#h1", "khan:t:0:37#h2"]
    gsm = [c for c in chunks if c["doc_id"] == "gsm8k:test:4"]
    assert len(gsm) > 1 and all(c["token_count"] <= 512 for c in gsm) and info["worked_docs_split_over_512_tokens"] == 1
    assert ids == [c["chunk_id"] for c in make_chunks(docs_frame(), count)[0]]  # stable across builds
    k = chunks[0]
    assert k["skill_tag"] == "fractions" and k["skill_tags"] == ["fractions", "ratio"]
    assert k["content_type"] == "worked_example" and k["text"].startswith("Problem: What is 1 + 2?\nSolution:")
    assert chunks[1]["content_type"] == "hint_step" and "Hint 1 of 2: Add." in chunks[1]["text"]
    assert gsm[0]["p_correct_hist"] is None and gsm[0]["licence"] == "MIT"
    assert k["difficulty_reliable"] is True and gsm[0]["difficulty_reliable"] is False  # P1-06 choice (a)
    assert k["difficulty_tier"] == 1 and gsm[0]["difficulty_tier"] == 3
    for c in chunks:
        json.dumps(c)  # plain JSON, no numpy scalars or NaN
    long_hint = docs_frame().iloc[[0]].copy()
    long_hint.at[long_hint.index[0], "hints"] = ["short", " ".join(["x"] * 700)]
    hint_chunks = [c for c in make_chunks(long_hint, count)[0] if c["content_type"] == "hint_step"]
    assert [c["chunk_id"] for c in hint_chunks][:2] == ["khan:t:0:37#h1", "khan:t:0:37#h2.0"]
    assert all(c["token_count"] <= 512 for c in hint_chunks)


def test_exercise_bank_has_statement_only():
    bank = exercise_bank(docs_frame())
    assert [b["chunk_id"] for b in bank] == ["khan:t:0:37#q", "gsm8k:test:4#q"]
    assert bank[0]["text"] == "What is 1 + 2?" and bank[0]["answer"] == "3"
    assert [b["difficulty_reliable"] for b in bank] == [True, False]
    json.dumps(bank)


def test_py_converts_numpy_and_nan():
    assert _py(np.int64(3)) == 3 and type(_py(np.int64(3))) is int
    assert _py(float("nan")) is None and _py("x") == "x"


def test_history_counts_add_outside_sample_learners():
    inter = pd.DataFrame({"item_id": ["t:a", "t:a", "t:b", "t:a"], "khan_slug": ["t", "t", "t", "t"],
                          "correct": [1, 0, 1, 1], "split": ["history", "history", "history", "test"]})
    unit, template = D.junyi_history_counts(inter)
    assert unit.set_index("type_key").loc["a", "n"] == 2  # the test row is not counted
    outside = pd.DataFrame({"khan_slug": ["t", "u"], "type_key": ["a", "x"], "n": [10, 5], "k": [7, 5]})
    unit, template = D.junyi_history_counts(inter, outside)
    u = unit.set_index(["khan_slug", "type_key"])
    assert (u.loc[("t", "a"), "n"], u.loc[("t", "a"), "k"]) == (12, 8)
    assert template.set_index("khan_slug").loc["t", "n"] == 13 and ("u", "x") in u.index


def test_outside_sample_counts_exclude_sampled_learners():
    from lmsrag.datasets.junyi2015 import outside_sample_counts

    eligible = pd.DataFrame({"user_id": [1, 1, 2, 3], "exercise": ["t"] * 4, "problem_type": ["a", "a", "a", "b"],
                             "correct": [True, False, True, True]})
    counts = outside_sample_counts(eligible, eligible[eligible["user_id"] == 1])
    assert counts.set_index("type_key")[["n", "k"]].to_dict("index") == {"a": {"n": 1, "k": 1}, "b": {"n": 1, "k": 1}}
