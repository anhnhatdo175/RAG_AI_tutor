from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lmsrag.sim.env import Observation, SimEnv
from lmsrag.sim.model import (PUBLIC, build_world, calibrate_mu, expected_rate, item_difficulty, junyi_targets,
                              learning_gain, logit, p_correct)

PARAMS = {"guess": {"radio": 0.20, "default": 0.02}, "slip": 0.05, "discrimination_log_sd": 0.3, "theta_sd": 1.0,
          "p_clip": [0.01, 0.99], "zpd": {"center": 0.7, "width": 0.15}}


def _bank(n_per_skill=12, skills=("S1", "S2", "S3")):
    rng = np.random.default_rng(7)
    rows = []
    for k in skills:
        for j in range(n_per_skill):
            has_hist = j % 3 != 0
            rows.append({"chunk_id": f"khan:{k}:{j}#q", "skill_tag": k, "source": "khan" if has_hist else "gsm8k",
                         "difficulty": rng.uniform(0.05, 0.95), "difficulty_level": 1 + j % 5,
                         "difficulty_tier": 1 if has_hist else 3, "difficulty_reliable": has_hist,
                         "answer_type": "radio" if j % 4 == 0 else "number",
                         "p_correct_hist": rng.uniform(0.3, 0.95) if has_hist else np.nan,
                         "khan_slug": k, "problem_type": "0"})
    return pd.DataFrame(rows)


def _targets(rates=(0.85, 0.6, 0.72)):
    return pd.DataFrame({"junyi_rate": rates, "junyi_n": [500] * len(rates)}, index=["S1", "S2", "S3"][:len(rates)])


@pytest.fixture(scope="module")
def world():
    return build_world(_bank(), _targets(), PARAMS, seed=2026)


def test_learning_rules():
    p = np.array([0.0, 0.5, 0.7, 1.0])
    assert np.allclose(learning_gain(p, "flat"), 1.0)
    assert np.allclose(learning_gain(p, "info"), [0.0, 1.0, 0.84, 0.0])
    assert learning_gain(0.7, "zpd") == pytest.approx(1.0)
    assert learning_gain(0.85, "zpd") == pytest.approx(np.exp(-0.5))
    with pytest.raises(ValueError):
        learning_gain(p, "fast")


def test_response_model_bounds():
    assert p_correct(-50.0, 1.0, 0.0, 0.2, 0.05) == pytest.approx(0.2)
    assert p_correct(50.0, 1.0, 0.0, 0.2, 0.05) == pytest.approx(0.95)
    assert p_correct(0.0, 1.3, 0.0, 0.0, 0.0) == pytest.approx(0.5)


def test_item_difficulty_from_history_else_monotone_map():
    bank = _bank()
    b = item_difficulty(bank)
    has = bank["p_correct_hist"].notna().to_numpy()
    assert np.allclose(b[has], logit(1 - bank["p_correct_hist"].to_numpy()[has]))
    # the mapped items keep the order of their difficulty and stay inside the range of the fitted b
    d = bank["difficulty"].to_numpy()[~has]
    order = np.argsort(d)
    assert np.all(np.diff(b[~has][order]) >= -1e-12)
    assert b[has].min() - 1e-12 <= b[~has].min() and b[~has].max() <= b[has].max() + 1e-12


def test_junyi_targets_use_history_only():
    j = pd.DataFrame({"skill_tag": ["A"] * 4 + ["B"] * 3, "correct": [1, 1, 0, 0, 1, 1, 1],
                      "split": ["history", "history", "history", "test", "history", "history", "validation"]})
    t = junyi_targets(j, min_responses=2)
    assert t.loc["A", "junyi_rate"] == pytest.approx(2 / 3) and t.loc["A", "junyi_n"] == 3
    assert t.loc["B", "junyi_n"] == 2
    assert junyi_targets(j, min_responses=3).index.tolist() == ["A"]


def test_calibration_hits_targets_and_flags_unreachable(world):
    assert np.allclose(world.skills["model_rate"], world.skills["junyi_rate"], atol=1e-6)
    it = world.items
    args = [it[c].to_numpy(dtype="float64") for c in ("a", "b", "g", "s")]
    mu, rate = calibrate_mu(np.array([0.99, 0.01, 0.7]), 1.0, *args, it["skill_idx"].to_numpy())
    assert rate[0] < 0.96 and rate[1] > 0.01  # above 1 - s and under the guess: the bound, the gap stays visible
    assert rate[2] == pytest.approx(0.7, abs=1e-6)
    assert expected_rate(mu, 1.0, *args, it["skill_idx"].to_numpy(), 3)[2] == pytest.approx(0.7, abs=1e-6)


def test_world_drops_skills_without_target_and_hides_parameters():
    w = build_world(_bank(skills=("S1", "S2", "S9")), _targets((0.8, 0.7)), PARAMS, seed=1)
    assert set(w.items["skill_tag"]) == {"S1", "S2"}
    assert list(w.catalog().columns) == [c for c in PUBLIC if c in w.items.columns]
    assert not {"a", "b", "g", "s", "skill_idx"} & set(w.catalog().columns)
    assert set(w.items.loc[w.items["answer_type"] == "radio", "g"]) == {0.20}
    assert set(w.items.loc[w.items["answer_type"] != "radio", "g"]) == {0.02}


def test_same_world_seed_same_world():
    a = build_world(_bank(), _targets(), PARAMS, seed=5)
    b = build_world(_bank(), _targets(), PARAMS, seed=5)
    c = build_world(_bank(), _targets(), PARAMS, seed=6)
    assert np.array_equal(a.items["a"], b.items["a"]) and not np.array_equal(a.items["a"], c.items["a"])


def _run(env, seed, plan):
    env.reset(seed)
    return [(u, item, env.step(u, item)) for u, item in plan]


def test_same_seed_same_sequence_whatever_the_visiting_order(world):
    ids = world.items["item_id"].tolist()
    rng = np.random.default_rng(3)
    plan = [(int(u), ids[int(i)]) for u, i in zip(rng.integers(0, 20, 400), rng.integers(0, len(ids), 400))]
    env = SimEnv(world, "zpd", n_learners=20)
    first, again = _run(env, 11, plan), _run(env, 11, plan)
    assert first == again
    assert _run(env, 12, plan) != first
    # a learner's answers depend on its own items only, not on when the policy visits the other learners
    by_u = {u: [c for v, _, c in first if v == u] for u in range(20)}
    reordered = sorted(plan, key=lambda x: -x[0])
    got = _run(env, 11, reordered)
    assert {u: [c for v, _, c in got if v == u] for u in range(20)} == by_u
    # learners do not depend on the learning rule
    other = SimEnv(world, "flat", n_learners=20)
    other.reset(11)
    env.reset(11)
    assert all(other.true_theta(u, k) == env.true_theta(u, k) for u in range(20) for k in env.skills)


def test_answers_match_true_p_over_10000_turns(world):
    env = SimEnv(world, "info", n_learners=200)
    env.reset(0)
    ids = world.items["item_id"].tolist()
    rng = np.random.default_rng(1)
    p, c = [], []
    for t in range(10_000):
        u, item = int(rng.integers(200)), ids[int(rng.integers(len(ids)))]
        p.append(env.true_p(u, item))
        c.append(env.step(u, item))
    p, c = np.array(p), np.array(c)
    se = np.sqrt(np.sum(p * (1 - p))) / len(p)
    assert abs(c.mean() - p.mean()) < 4 * se
    for lo, hi in [(0, 0.4), (0.4, 0.7), (0.7, 0.85), (0.85, 1)]:  # also within bins of P
        m = (p >= lo) & (p < hi)
        if m.sum() > 200:
            assert abs(c[m].mean() - p[m].mean()) < 4 * np.sqrt(np.sum(p[m] * (1 - p[m]))) / m.sum()


def test_step_learns_only_the_answered_skill(world):
    env = SimEnv(world, "zpd", n_learners=3, eta=(0.05, 0.05))
    env.reset(4)
    item = world.items.iloc[0]
    before = {k: env.true_theta(1, k) for k in env.skills}
    p = env.true_p(1, item["item_id"])
    env.step(1, item["item_id"])
    gain = 0.05 * learning_gain(p, "zpd")
    for k in env.skills:
        expect = before[k] + (gain if k == item["skill_tag"] else 0.0)
        assert env.true_theta(1, k) == pytest.approx(expect)


def test_observe_returns_answers_only(world):
    env = SimEnv(world, "flat", n_learners=2)
    with pytest.raises(RuntimeError):
        env.observe(0)
    env.reset(0)
    assert env.observe(0) == ()
    ids = world.items["item_id"].tolist()[:3]
    answers = [env.step(0, i) for i in ids]
    obs = env.observe(0)
    assert obs == tuple(Observation(t, i, c) for t, (i, c) in enumerate(zip(ids, answers)))
    assert Observation._fields == ("t", "item_id", "correct")
    assert all(type(o.correct) is int and o.correct in (0, 1) for o in obs)
    assert env.observe(1) == ()
    obs_before = env.observe(0)
    env.step(1, ids[0])
    assert env.observe(0) == obs_before  # another learner's turn changes nothing here
    with pytest.raises(IndexError):
        env.observe(2)
    with pytest.raises(KeyError):
        env.step(0, "nope")


def test_tune_eta_keeps_ratio_and_hits_target(world):
    from lmsrag.sim.calibrate import tune_eta, turns_to_unit
    cfg = {**PARAMS, "eta": [0.02, 0.08]}
    (lo, hi), log = tune_eta(world, cfg, target=25, n_learners=40, seed=0, max_turns=300, iters=8)
    assert hi / lo == pytest.approx(4.0)
    assert log["pooled_median_turns"].iloc[-1] > 0
    tuned = {**cfg, "eta": [lo, hi]}
    pooled = np.concatenate([turns_to_unit(world, tuned, r, 40, 0, 300) for r in ("flat", "info", "zpd")])
    assert np.median(pooled) <= 25
    # paired draws: a larger eta never needs more turns for the same learner under the flat rule
    slow = turns_to_unit(world, cfg, "flat", 40, 0, 300)
    fast = turns_to_unit(world, {**cfg, "eta": [0.04, 0.16]}, "flat", 40, 0, 300)
    assert np.all(fast <= slow)


REPO = Path(__file__).resolve().parents[2]
ART = REPO / "lmsrag" / "artifacts" / "v2"


@pytest.mark.skipif(not (ART / "exercise_bank_v2.jsonl").exists() or not (ART / "interactions_junyi2015.parquet").exists(),
                    reason="needs the P1-03 and P1-04 artifacts")
def test_real_world_initial_accuracy_matches_junyi():
    pytest.importorskip("pyarrow")
    from lmsrag.datasets.build import load_config
    from lmsrag.sim.model import load_world
    cfg = load_config(REPO / "configs" / "sim.yaml")
    w = load_world(cfg, REPO)
    assert (w.skills["model_rate"] - w.skills["junyi_rate"]).abs().max() < 0.05
