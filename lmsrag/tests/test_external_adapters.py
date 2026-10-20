"""Junyi / EdNet adapters on SYNTHETIC files that follow the documented schemas (no real data is bundled).
They prove the code path (join to corpus, correctness rule, time order, split, difficulty buckets), not the real schemas."""
import json

import numpy as np
import pandas as pd

from lmsrag.layer1_data import ednet, junyi
from lmsrag.skill_mapping import SkillMapper


def _chunks(tmp_path):
    rows = []
    for ex, tag in [("rounding_decimals_1", "Rounding"), ("rounding_decimals_2", "Rounding"), ("rounding_numbers", "Rounding"), ("rounding_big", "Rounding"),
                    ("mean_1", "Mean")]:
        for _ in range(3):
            rows.append(dict(subject=f"khan:{ex}", skill_tag=tag))
    rows.append(dict(subject="MATH", skill_tag="Mean"))             # non-khan chunks must be ignored
    p = tmp_path / "chunks.jsonl"; p.write_text("\n".join(json.dumps(r) for r in rows)); return p


def test_junyi(tmp_path):
    rng = np.random.default_rng(0); ex = ["rounding_decimals_1", "rounding_decimals_2", "rounding_numbers", "rounding_big", "mean_1", "no_khan_twin"]
    rows = []
    for u in range(25):
        for i in range(40):
            e = ex[int(rng.integers(0, 6))]; ok = rng.random() < 0.6
            rows.append(dict(user_id=u, exercise=e, time_done=f"2020-01-{1 + i % 28:02d} 10:{i % 60:02d}:00 UTC", total_sec_taken=int(rng.integers(3, 90)),
                             total_attempt_cnt=1 if ok else 2, used_hint_cnt=0, is_correct=int(ok or rng.random() < .5)))
    pd.DataFrame(rows).to_csv(tmp_path / "junyi_ProblemLog_original.csv", index=False)
    m = SkillMapper(); ex2tag = junyi.exercise_to_tag(_chunks(tmp_path))
    assert ex2tag["mean_1"] == "Mean" and "no_khan_twin" not in ex2tag
    df = junyi.load_junyi(m, tmp_path, chunks_jsonl=_chunks(tmp_path))
    assert "no_khan_twin" not in set(df.problem_id)                        # unmapped exercises are dropped, not guessed
    assert set(df.tag.dropna()) <= {"Rounding", "Mean"} and df.tag.notna().all()
    assert set(df.split) == {"history", "val", "test"} and df.groupby("user_id").order_id.apply(lambda s: s.is_monotonic_increasing).all()
    assert df.correct.isin([0, 1]).all() and (df[df.correct == 1].shape[0] > 0)
    assert set(df.bucket.unique()) <= {0, 1, 2}


def test_junyi_correct_means_first_try_without_hint(tmp_path):
    pd.DataFrame([dict(user_id=1, exercise="mean_1", time_done="2020-01-01 10:00:00 UTC", total_sec_taken=5, total_attempt_cnt=2, used_hint_cnt=0, is_correct=1),
                  dict(user_id=1, exercise="mean_1", time_done="2020-01-01 10:01:00 UTC", total_sec_taken=5, total_attempt_cnt=1, used_hint_cnt=1, is_correct=1),
                  dict(user_id=1, exercise="mean_1", time_done="2020-01-01 10:02:00 UTC", total_sec_taken=5, total_attempt_cnt=1, used_hint_cnt=0, is_correct=1)]
                 ).to_csv(tmp_path / "junyi_ProblemLog_original.csv", index=False)
    raw = junyi.read_junyi(tmp_path / "junyi_ProblemLog_original.csv", {"mean_1": "Mean"}, SkillMapper())
    assert raw.sort_values("order_id").correct.tolist() == [0, 0, 1]


def test_ednet(tmp_path):
    rng = np.random.default_rng(1)
    pd.DataFrame(dict(question_id=[f"q{i}" for i in range(12)], bundle_id=range(12), explanation_id=range(12), correct_answer=rng.choice(list("abcd"), 12),
                      part=[5] * 12, tags=["1;2", "3", "1", "3;4"] * 3, deployed_at=0)).to_csv(tmp_path / "questions.csv", index=False)
    q = pd.read_csv(tmp_path / "questions.csv"); (tmp_path / "KT1").mkdir()
    for u in range(1, 6):
        n = 30
        d = pd.DataFrame(dict(timestamp=np.sort(rng.integers(1, 10**9, n)), solving_id=range(n), question_id=rng.choice(q.question_id, n),
                              user_answer=rng.choice(list("abcd"), n), elapsed_time=rng.integers(1000, 60000, n)))
        d.to_csv(tmp_path / "KT1" / f"u{u}.csv", index=False)
    df = ednet.load_ednet(SkillMapper(), tmp_path)
    assert df.tag.isna().all()                                            # no TOEIC corpus -> honest None
    assert df.correct.isin([0, 1]).all() and set(df.split) == {"history", "val", "test"}
    assert df.skill_name.str.startswith("ednet:").all()
