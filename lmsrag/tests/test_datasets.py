import numpy as np
import pandas as pd
import pytest

from lmsrag.datasets import schema
from lmsrag.datasets.assist2009 import clean
from lmsrag.datasets.cleaning import kept_vs_dropped
from lmsrag.datasets.skill_builder import skill_builder_table
from lmsrag.datasets.splits import assign_splits, sample_users, truncate_history

CFG = {"data_seed": 2026, "split": {"history": 0.70, "validation": 0.15, "test": 0.15, "min_user_interactions": 20},
       "rt_clip_ms": [500, 120000], "bootstrap": {"n_resamples": 200}}


def synthetic_log(n_users=30, seed=0, multi_row_events=True):
    """Common-schema rows; some events span two rows (same user and ts), as multi-skill ASSISTments questions do."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        n_events = int(rng.integers(20, 90))
        ts = np.sort(rng.choice(10_000, size=n_events, replace=False))
        for t in ts:
            for skill in (["a", "b"] if multi_row_events and rng.random() < 0.2 else ["a"]):
                rows.append({"dataset": "assist2009", "user_id": f"u{u}", "item_id": f"i{int(rng.integers(50))}",
                             "source_skill": skill, "skill_tag": skill, "khan_slug": None, "ts": int(t),
                             "correct": int(rng.random() < 0.6), "rt_ms": float(rng.integers(500, 9000)),
                             "attempts": 1, "hints": 0, "section_type": "MasterySection"})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed)  # shuffled on purpose


def test_split_is_chronological_per_learner():
    df = assign_splits(synthetic_log(), CFG)
    for _, g in df.groupby("user_id"):
        h, v, t = (g[g.split == s].ts for s in ("history", "validation", "test"))
        assert len(h) and len(v) and len(t)
        assert h.max() < v.min() and v.max() < t.min()


def test_split_shares_within_one_event_and_events_not_split():
    df = assign_splits(synthetic_log(), CFG)
    assert (df.groupby(["user_id", "ts"]).split.nunique() == 1).all()
    events = df.drop_duplicates(["user_id", "ts"])
    for _, g in events.groupby("user_id"):
        n = len(g)
        for part, share in (("history", 0.70), ("validation", 0.15), ("test", 0.15)):
            assert abs((g.split == part).sum() - share * n) <= 1


def test_split_is_deterministic_and_order_independent():
    log = synthetic_log()
    a = assign_splits(log, CFG)
    b = assign_splits(log.sample(frac=1.0, random_state=7), CFG)
    key = ["user_id", "ts", "source_skill"]
    pd.testing.assert_frame_equal(a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True))


def test_truncate_history_keeps_first_k_events_and_all_test_rows():
    df = assign_splits(synthetic_log(), CFG)
    for k in (0, 1, 3):
        cut = truncate_history(df, k)
        assert not (cut.split == "validation").any()
        hist = cut[cut.split == "history"]
        assert (hist.groupby("user_id").ts.nunique() <= k).all()
        full_hist = df[df.split == "history"]
        first = full_hist.groupby("user_id").ts.min()
        if k:
            assert (hist.groupby("user_id").ts.min() == first.loc[hist.user_id.unique()]).all()
        pd.testing.assert_frame_equal(cut[cut.split == "test"].reset_index(drop=True),
                                      df[df.split == "test"].reset_index(drop=True))
    with pytest.raises(ValueError):
        truncate_history(df, -1)


def test_schema_validate_accepts_coerced_and_rejects_bad_values():
    df = schema.coerce(assign_splits(synthetic_log(n_users=3), CFG))
    schema.validate(df)
    bad = df.copy()
    bad["correct"] = bad["correct"].where(bad.index != 0, 2).astype("int8")
    with pytest.raises(schema.SchemaError):
        schema.validate(bad)
    with pytest.raises(schema.SchemaError):
        schema.validate(df.drop(columns=["hints"]))
    nulls = df.copy()
    nulls.loc[0, "user_id"] = None
    with pytest.raises(schema.SchemaError):
        schema.validate(nulls)


def test_clean_rt_clips_and_blanks_non_positive():
    rt = schema.clean_rt(pd.Series([-5, 0, 100, 3000, 10**7]), (500, 120000))
    assert rt.isna().tolist() == [True, True, False, False, False]
    assert rt.dropna().tolist() == [500.0, 3000.0, 120000.0]


def test_sample_users_is_seeded():
    log = synthetic_log(n_users=40)
    a = set(sample_users(log, 10, 2026).user_id)
    assert a == set(sample_users(log, 10, 2026).user_id) and len(a) == 10
    assert a != set(sample_users(log, 10, 2027).user_id)
    assert sample_users(log, None, 1) is log


class FakeMapper:
    def to_tag(self, name):
        return None if name == "Venn Diagram" else name.lower()

    def unmapped_reason(self, name):
        return "visual"


def raw_2009(n_users=6, seed=1):
    rng = np.random.default_rng(seed)
    rows, order = [], 1000
    for u in range(n_users):
        n = 30 if u else 10  # learner 0 has too few rows
        for j in range(n):
            order += 1
            skills = [(1, "Addition"), (2, "Venn Diagram")] if j % 5 == 0 else [(1, "Addition")]
            for sid, sname in skills:
                rows.append({"order_id": order, "user_id": u, "problem_id": 100 + j, "original": 1,
                             "correct": int(rng.random() < 0.7), "skill_id": sid, "skill_name": sname,
                             "ms_first_response": 2000, "attempt_count": 1, "hint_count": 0, "type": "MasterySection"})
    raw = pd.DataFrame(rows)
    extra = raw.iloc[:3].copy()
    extra["original"] = 0                       # scaffolding rows
    dup = raw.iloc[[5]].copy()                  # exact duplicate (order_id, skill_id)
    missing = raw.iloc[[6]].copy()
    missing["skill_id"] = np.nan
    missing["order_id"] = 1
    return pd.concat([raw, extra, dup, missing], ignore_index=True)


def test_assist2009_clean_logs_every_step_and_validates():
    raw = raw_2009()
    res = clean(raw, CFG, mapper=FakeMapper())
    steps = res.tables["a1_cleaning_assist2009"]
    applied = steps[steps.applied]
    assert applied.rows_before.iloc[0] == len(raw)
    assert (applied.rows_before.iloc[1:].to_numpy() == applied.rows_after.iloc[:-1].to_numpy()).all()
    assert (applied.dropped == applied.rows_before - applied.rows_after).all()
    by_rule = dict(zip(applied.rule, applied.dropped))
    assert by_rule["drop original == 0 (scaffolding sub-questions)"] == 3
    assert by_rule["drop missing skill_id or skill_name"] == 1
    assert by_rule["drop duplicates on (order_id, skill_id)"] == 1
    assert by_rule["drop learners with < 20 rows"] == 12   # learner 0: 10 events, 2 of them double-skill
    assert res.facts["unmapped_rows"] == 5 * 6
    schema.validate(res.interactions)
    assert res.interactions.skill_tag.notna().all() and res.interactions.user_id.nunique() == 5
    # same input -> identical output
    pd.testing.assert_frame_equal(res.interactions, clean(raw, CFG, mapper=FakeMapper()).interactions)


def test_kept_vs_dropped_point_estimate_and_interval():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"user_id": rng.integers(0, 50, 4000), "correct": (rng.random(4000) < 0.6).astype(int),
                       "source_skill": rng.choice(["x", "y"], 4000)})
    mask = df.source_skill == "x"
    out = kept_vs_dropped(df, mask, 300, 0)
    row = out[out.metric == "accuracy_diff"].iloc[0]
    expected = df[mask].correct.mean() - df[~mask].correct.mean()
    assert row.value == pytest.approx(expected, abs=1e-4)
    assert row.ci_low <= row.value <= row.ci_high


def test_skill_builder_table_counts_sequences_ending_on_three_correct():
    df = pd.DataFrame({"user_id": ["u"] * 7, "source_skill": ["s"] * 4 + ["t"] * 3, "ts": range(7),
                       "correct": [0, 1, 1, 1, 1, 0, 1], "split": ["history"] * 5 + ["test"] * 2})
    m = skill_builder_table(df, "toy", "all").set_index("metric")["value"]
    assert m["sequences"] == 2
    assert m["share_sequences_ending_3_correct"] == 0.5
    assert m["accuracy_position_1"] == 0.5
