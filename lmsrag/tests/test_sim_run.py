import numpy as np
import pandas as pd
import pytest

from lmsrag.sim.baselines import FixedLevelPolicy, MostPopularPolicy, RandomPolicy, junyi_popularity
from lmsrag.sim.env import Observation, SimEnv
from lmsrag.sim.model import PUBLIC, build_world
from lmsrag.sim.run import assign_skills, bootstrap_index, make_policies, run, run_policy, summarise
from lmsrag.tests.test_sim import PARAMS, _bank, _targets

CFG = {**PARAMS, "world_seed": 2026, "eta": [0.02, 0.08], "n_learners": 30, "learning_rules": ["flat", "zpd"],
       "closed_loop": {"n_learners": 30, "turns": 12, "seeds": [0, 1], "fixed_level": 3, "zpd_band": [0.6, 0.8],
                       "mastery_threshold": 0.95, "n_resamples": 200, "compare_to": "fixed",
                       "proposed": "lmsrag.sim.no_such_module:ProposedPolicy"}}


@pytest.fixture(scope="module")
def world():
    return build_world(_bank(), _targets(), PARAMS, seed=2026)


def _cands(levels, skill="S1"):
    return pd.DataFrame({"item_id": [f"i{j}" for j in range(len(levels))], "skill_tag": skill,
                         "difficulty_level": levels})


def _hist(items):
    return tuple(Observation(t, i, 1) for t, i in enumerate(items))


def test_random_gives_every_candidate_before_any_repeat():
    c, pol = _cands([1, 2, 3, 4, 5]), RandomPolicy(seed=1)
    got = []
    for _ in range(10):
        got.append(pol.choose(_hist(got), c))
    assert sorted(got[:5]) == sorted(c["item_id"]) and sorted(got[5:]) == sorted(c["item_id"])


def test_fixed_uses_level_3_else_the_nearest_lower_level():
    pol = FixedLevelPolicy(3)
    assert {pol.choose((), _cands([1, 3, 3, 5])) for _ in range(20)} <= {"i1", "i2"}
    pol2 = FixedLevelPolicy(3)
    assert pol2.choose((), _cands([2, 4, 5], skill="S2")) == "i0"  # 2 and 4 are both one level away: the lower
    assert pol2.choose(_hist(["i0"]), _cands([2, 4, 5], skill="S2")) == "i0"  # the only level-2 item comes back


def test_most_popular_goes_down_the_popularity_list():
    pol = MostPopularPolicy({"i0": (5, 100), "i1": (50, 60), "i2": (5, 200)})
    c = _cands([1, 1, 1, 1])  # i3 has no Junyi count
    got = []
    for _ in range(5):
        got.append(pol.choose(_hist(got), c))
    assert got == ["i1", "i2", "i0", "i3", "i1"]


def test_junyi_popularity_counts_both_spellings_of_a_problem_type():
    junyi = pd.DataFrame({"item_id": ["dec:0", "dec:0", "dec:tenths", "dec:1", "add:0", "add:0"],
                          "split": ["history", "history", "history", "history", "history", "test"]})
    types = pd.DataFrame({"slug": ["dec", "dec"], "problem_index": ["0", "1"], "problem_id": ["tenths", "quarters"]})
    cat = pd.DataFrame({"item_id": ["a", "b", "c", "d"], "khan_slug": ["dec", "dec", "add", None],
                        "problem_type": ["tenths", "quarters", "0", None]})
    pop = junyi_popularity(cat, junyi, types)
    assert pop == {"a": (3, 4), "b": (1, 4), "c": (1, 1), "d": (0, 0)}


class Spy:
    """Records what a policy is given, and picks the first candidate."""

    def __init__(self):
        self.seen = []

    def choose(self, observed_history, candidate_items):
        self.seen.append((observed_history, candidate_items))
        return candidate_items["item_id"].iat[0]


def test_policy_sees_only_observations_and_public_candidates(world):
    env = SimEnv(world, "zpd", n_learners=5)
    cand = {k: g.reset_index(drop=True) for k, g in world.catalog().groupby("skill_tag")}
    spy = Spy()
    practised = assign_skills(env.skills, 5, 0)
    run_policy(env, spy, 0, practised, cand, turns=4, band=(0.6, 0.8), threshold=0.95)
    assert len(spy.seen) == 20
    for hist, c in spy.seen:
        assert isinstance(hist, tuple) and all(type(o) is Observation for o in hist)
        assert set(c.columns) <= set(PUBLIC) and c["skill_tag"].nunique() == 1
    assert [len(h) for h, _ in spy.seen[:4]] == [0, 1, 2, 3]


def test_out_of_skill_choice_is_refused(world):
    env = SimEnv(world, "zpd", n_learners=2)
    cand = {k: g.reset_index(drop=True) for k, g in world.catalog().groupby("skill_tag")}

    class Cheat:
        def choose(self, h, c):
            return "not-an-item"

    with pytest.raises(ValueError):
        run_policy(env, Cheat(), 0, assign_skills(env.skills, 2, 0), cand, 3, (0.6, 0.8), 0.95)


def test_flat_rule_gain_does_not_depend_on_the_policy(world):
    env = SimEnv(world, "flat", n_learners=20)
    cand = {k: g.reset_index(drop=True) for k, g in world.catalog().groupby("skill_tag")}
    practised = assign_skills(env.skills, 20, 3)
    a = run_policy(env, RandomPolicy(), 3, practised, cand, 10, (0.6, 0.8), 0.95)
    b = run_policy(env, FixedLevelPolicy(3), 3, practised, cand, 10, (0.6, 0.8), 0.95)
    env.reset(3)
    eta = env._eta_u  # test only: G = 1, so the gain is eta_u per turn
    assert np.allclose(a["mastery_gain"], 10 * eta) and np.allclose(b["mastery_gain"], 10 * eta)
    assert a["curve"][0] == 0 and np.all(np.diff(a["curve"]) > 0)
    for r in (a, b):
        assert np.all((r["zpd_hit_rate"] >= 0) & (r["zpd_hit_rate"] <= 1))
        assert np.all(r["turns_to_mastery"][r["not_mastered"] == 1] == 10)
        assert np.all(r["turns_to_mastery"][r["mastered_at_start"] == 1] == 0)


def test_summarise_and_paired_differences():
    rng = np.random.default_rng(0)
    v = rng.normal(1.0, 0.5, size=(3, 400))
    idx = bootstrap_index(3, 400, 300, 1)
    s = summarise(v, idx)
    assert s["ci_low"] < s["mean"] < s["ci_high"]
    assert s["mean"] == pytest.approx(v.mean())
    z = summarise(v - v, idx)
    assert z["mean"] == 0 and z["ci_low"] == 0 and z["ci_high"] == 0


def test_missing_proposed_is_skipped_on_tune_and_fatal_on_final(world, capsys):
    pols = make_policies(["random", "proposed"], CFG, world.catalog(), None, required=False)
    assert list(pols) == ["random"] and "not available" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        make_policies(["proposed"], CFG, world.catalog(), None, required=True)


def test_run_writes_paired_tables(world):
    pols = make_policies(["random", "fixed"], CFG, world.catalog(), None, required=False)
    table, diffs, curves = run(world, CFG, pols)
    assert set(table["policy"]) == {"random", "fixed"} and set(table["rule"]) == {"flat", "zpd"}
    assert (table["label"] == "Kết quả mô phỏng").all()
    assert set(diffs["vs"]) == {"fixed"} and set(diffs["policy"]) == {"random"}
    flat_gain = diffs[(diffs["rule"] == "flat") & (diffs["metric"] == "mastery_gain")]
    assert flat_gain["diff"].abs().max() < 1e-12  # same learners, G = 1: no difference at all
    assert len(curves) == 2 * 2 * (12 + 1)
    again, _, _ = run(world, CFG, make_policies(["random", "fixed"], CFG, world.catalog(), None, required=False))
    pd.testing.assert_frame_equal(table, again)


def test_true_p_many_matches_true_p(world):
    env = SimEnv(world, "zpd", n_learners=3)
    env.reset(2)
    ids = world.items["item_id"].tolist()[:7]
    assert np.allclose(env.true_p_many(1, ids), [env.true_p(1, i) for i in ids])


@pytest.mark.parametrize("mode", ["uniform_unmastered", "junyi_unmastered"])
def test_unmastered_assignment_never_starts_mastered(world, mode):
    from lmsrag.sim.run import practised_skills
    env = SimEnv(world, "flat", n_learners=200)
    skills, fallback = practised_skills(env, world, 5, mode, 0.95)
    env.reset(5)
    started = [env.true_mastery(u, env.median_item(k)) >= 0.95 for u, k in enumerate(skills)]
    assert fallback == 0 and not any(started)
    other = SimEnv(world, "zpd", n_learners=200)  # the same learners whatever the rule
    assert practised_skills(other, world, 5, mode, 0.95)[0] == skills


def test_junyi_assignment_follows_junyi_counts(world):
    from lmsrag.sim.run import practised_skills
    w = build_world(_bank(), pd.DataFrame({"junyi_rate": [0.8, 0.7, 0.75], "junyi_n": [9000, 500, 500]},
                                          index=["S1", "S2", "S3"]), PARAMS, seed=1)
    env = SimEnv(w, "flat", n_learners=2000)
    skills, _ = practised_skills(env, w, 0, "junyi", 0.95)
    assert pd.Series(skills).value_counts(normalize=True)["S1"] == pytest.approx(0.9, abs=0.03)
    with pytest.raises(ValueError):
        practised_skills(env, w, 0, "by_magic", 0.95)


def test_oracle_reference_reaches_more_zpd_hits_than_random(world):
    from lmsrag.sim.run import OracleReference
    env = SimEnv(world, "zpd", n_learners=60)
    cand = {k: g.reset_index(drop=True) for k, g in world.catalog().groupby("skill_tag")}
    practised = assign_skills(env.skills, 60, 1)
    o = run_policy(env, OracleReference(0.7), 1, practised, cand, 15, (0.6, 0.8), 0.95)
    r = run_policy(env, RandomPolicy(), 1, practised, cand, 15, (0.6, 0.8), 0.95)
    assert o["zpd_hit_rate"].mean() > r["zpd_hit_rate"].mean()


def test_design_decision_rule():
    from lmsrag.sim.design_check import decide
    res = pd.DataFrame({"skill_assignment": ["uniform"] * 2 + ["junyi"] * 2 + ["uniform_unmastered"] * 2
                        + ["junyi_unmastered"] * 2,
                        "mastered_at_start": [0.2, 0.2, 0.5, 0.5, 0, 0, 0, 0],
                        "sensitivity": [90, 90, 80, 80, 30, 32, 33, 34]})
    pick, s = decide(res)
    assert pick == "uniform_unmastered"  # invalid designs are out; 31 is within 10% of 33.5, so the simpler wins
    assert s.loc["uniform", "valid"] == False and s.loc["junyi_unmastered", "valid"] == True  # noqa: E712
    res.loc[res["skill_assignment"] == "junyi_unmastered", "sensitivity"] = 60
    assert decide(res)[0] == "junyi_unmastered"
