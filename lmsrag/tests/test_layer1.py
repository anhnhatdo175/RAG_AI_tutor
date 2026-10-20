import numpy as np
import pandas as pd
import pytest

from lmsrag import data, profile
from lmsrag.layer1_data import interaction_matrix as im
from lmsrag.layer1_data.learner_state import LearnerState, build_states
from lmsrag.layer1_data.state_store import StateStore
from lmsrag.skill_mapping import SkillMapper


@pytest.fixture(scope="module")
def df():
    return data.load_logs(SkillMapper())


def test_R_range_and_splits(df):
    R, users, items = im.build_R(df, ("history",))
    assert R.shape == (len(users), len(items)) and 0 <= R.data.min() and R.data.max() <= 1
    hist_users = set(df[df.split == "history"].user_id)
    assert set(users) <= hist_users                                   # nothing from val/test leaked in
    R2, *_ = im.build_R(df, ("history", "val", "test"))
    assert R2.nnz >= R.nnz


def test_state_matches_manual_ema():
    st = LearnerState(1)
    M = 0.6
    for y in [1, 0, 1, 1, 0, 1]:
        st.update(7, float(y), 10000, alpha=0.1, prior=0.6)
        M = 0.1 * y + 0.9 * M
    assert st.M["7"] == pytest.approx(M) and st.attempts["7"] == 6 and st.A == pytest.approx(4 / 6)


def test_weak_skills_and_pace():
    st = LearnerState(2)
    for i in range(60):                                               # improving learner on skill 1, failing skill 2
        st.update(1, float(i > 20), 8000, 0.2, 0.5)
        st.update(2, 0.0, 8000, 0.2, 0.5)
    assert st.pace > 0
    assert [k for k, _ in st.W] == [2]
    assert st.sigma > 0


def test_offline_equals_online(df):
    alpha = 0.1
    priors, _ = profile.skill_priors(df)
    hist = build_states(df, alpha, priors, ("history",))
    both = build_states(df, alpha, priors, ("history", "val"))
    u = next(iter(hist))
    st = LearnerState.from_json(hist[u].to_json())
    for r in df[(df.user_id == u) & (df.split == "val")].sort_values(["order_id", "skill_id"]).itertuples():
        st.update(r.skill_id, float(r.correct), float(r.ms), alpha, float(priors.get(r.skill_id, priors.mean())))
    assert st.M == pytest.approx(both[u].M) and st.n == both[u].n


def test_no_leakage_in_history_states(df):
    priors, _ = profile.skill_priors(df)
    states = build_states(df, 0.1, priors, ("history",))
    n = df[df.split == "history"].groupby("user_id").size()
    assert all(states[u].n == n[u] for u in list(states)[:300])


def test_state_store_roundtrip(tmp_path):
    s = StateStore(tmp_path / "s.sqlite")
    a = s.apply_event(5, 3, 1.0, 9000, 0.1, 0.6)
    b = s.apply_event(5, 3, 0.0, 9000, 0.1, 0.6)
    got = s.get(5)
    assert got.n == 2 and got.M == b.M and s.db.execute("SELECT version FROM learner_state").fetchone()[0] == 2
    assert s.get(999) is None


def test_canonical_event_updates_mastery_profile_and_persists(tmp_path):
    store = StateStore(tmp_path / "canonical-state.sqlite")

    state = store.apply_canonical_event(
        17, "Percent-Of", 0.0, 12_000, alpha=0.5, prior=0.8,
        bucket=1, tau_base=12_000, task_id="T-17",
    )
    assert state.canonical_M["Percent-Of"] == pytest.approx(0.4)
    assert state.canonical_attempts["Percent-Of"] == 1
    assert state.canonical_prof["Percent-Of"][1] == pytest.approx(0.3)
    assert state.seen == ["T-17"] and state.n == 1
    assert store.get(17).canonical_M == state.canonical_M
    assert store.db.execute("SELECT version FROM learner_state WHERE user_id=17").fetchone()[0] == 1

    for _ in range(4):
        state = store.apply_canonical_event(
            17, "Percent-Of", 0.0, 12_000, alpha=0.5, prior=0.8, bucket=1, tau_base=12_000,
        )
    assert state.canonical_W == [("Percent-Of", pytest.approx(state.canonical_M["Percent-Of"]))]
    assert state.summary()["n_canonical_weak"] == 1
    snapshot = state.context_snapshot()
    assert snapshot["canonical_mastery"]["Percent-Of"] == pytest.approx(state.canonical_M["Percent-Of"])
    assert snapshot["weak_canonical_skills"][0]["skill_tag"] == "Percent-Of"
    assert store.db.execute("SELECT version FROM learner_state WHERE user_id=17").fetchone()[0] == 5
    store.close()


def test_live_state_store_initializes_from_offline_snapshot_once(tmp_path):
    snapshot = StateStore(tmp_path / "snapshot.sqlite")
    snapshot.apply_event(12, 4, 1.0, 8_000, alpha=0.2, prior=0.5)
    snapshot.close()

    live = StateStore(tmp_path / "live.sqlite")
    assert live.initialize_from_snapshot(tmp_path / "snapshot.sqlite") == 1
    live.apply_canonical_event(12, "Mean", 0.0, 15_000, alpha=0.5, task_id="T-12")

    assert live.get(12).canonical_attempts["Mean"] == 1
    with pytest.raises(RuntimeError, match="already contains data"):
        live.initialize_from_snapshot(tmp_path / "snapshot.sqlite")
    unchanged_snapshot = StateStore(tmp_path / "snapshot.sqlite")
    assert unchanged_snapshot.get(12).canonical_M == {}
    unchanged_snapshot.close()
    live.close()


def test_legacy_learner_state_json_loads_with_canonical_defaults():
    state = LearnerState.from_json('{"user_id": 4, "M": {"1": 0.7}, "attempts": {"1": 3}}')

    assert state.canonical_M == {}
    assert state.canonical_attempts == {}
    assert state.canonical_prof == {}


def test_ST_is_strong_topic_list():
    """docx 1.2.1: ST_u = strong topics (mirror of W_u); study time is a separate property."""
    st = LearnerState(3)
    for _ in range(40):
        st.update(1, 1.0, 5000, 0.3, 0.5)          # skill 1 mastered
        st.update(2, 0.0, 5000, 0.3, 0.5)          # skill 2 weak
    for _ in range(3):
        st.update(3, 1.0, 5000, 0.3, 0.5)          # too few attempts -> neither strong nor weak
    assert [k for k, _ in st.ST] == [1] and [k for k, _ in st.W] == [2]
    assert st.summary()["n_strong"] == 1 and st.study_time["mean_rt_s"] > 0
