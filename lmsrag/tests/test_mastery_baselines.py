import numpy as np
import pandas as pd

from lmsrag.mastery import baselines as b
from lmsrag.mastery.metrics import ece, metrics


def _df(seed=0, users=30, n=40):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(users):
        ability = rng.normal(0, 1.5)
        for t in range(n):
            sk = f"s{t % 3}"
            p = 1 / (1 + np.exp(-(ability + 2 * t / n - 1)))
            split = "history" if t < 0.7 * n else ("validation" if t < 0.85 * n else "test")
            rows.append(("assist2009", str(u), f"i{t}", sk, sk, None, t * 1000, int(rng.random() < p), 1.0, 1, 0,
                         "x", split))
    return pd.DataFrame(rows, columns=["dataset", "user_id", "item_id", "source_skill", "skill_tag", "khan_slug", "ts",
                                       "correct", "rt_ms", "attempts", "hints", "section_type", "split"])


def test_prequential_counts_ignore_current_row_and_ties():
    d = pd.DataFrame({"user_id": ["a"] * 3, "ts": [1, 1, 2], "y": [1.0, 0.0, 1.0], "skill": ["x", "y", "x"]})
    s, n = b._cum_before(d, ["user_id"], [])
    assert n.tolist() == [0, 0, 2] and s.tolist() == [0, 0, 1]  # rows at ts=1 share the state before them


def test_ema_matches_layer1_update_rule():
    d = b.prepare(_df(users=1, n=6))
    prior = np.full(len(d), 0.5)
    p = b.ema(d, prior, 0.25)
    first = d.groupby("uk").head(1).index
    assert np.allclose(p[first], 0.5)
    i = d.index[d["uk"] == d.at[0, "uk"]]
    assert np.isclose(p[i[1]], 0.25 * d.at[i[0], "y"] + 0.75 * 0.5)


def test_bkt_forward_known_values():
    y, first = np.array([1.0, 0.0]), np.array([True, False])
    p = b.bkt_forward(y, first, 0.5, 0.1, 0.2, 0.1)
    assert np.isclose(p[0], 0.5 * 0.9 + 0.5 * 0.2)
    post = 0.5 * 0.9 / p[0]
    L = post + (1 - post) * 0.1
    assert np.isclose(p[1], L * 0.9 + (1 - L) * 0.2)


def test_last_answer_and_skill_prior():
    d = b.prepare(_df(users=2, n=6))
    prior = b.skill_priors(d)
    la = b.last_answer(d, prior)
    nxt = d.groupby("uk")["y"].shift(1)
    assert np.allclose(la[nxt.notna().to_numpy()], nxt.dropna().to_numpy())


def test_run_beats_majority_on_learning_data_and_is_deterministic():
    r1, r2 = b.run(_df()), b.run(_df())
    t = r1["table"]
    auc = t[t.split == "test"].set_index("model")["AUC"]
    assert auc["ema"] > 0.6 and auc["user_mean"] > 0.6 and set(auc.index) >= {"majority", "bkt", "pfa", "afm"}
    pd.testing.assert_frame_equal(r1["table"], r2["table"])


def test_metrics_and_ece():
    y = np.array([0, 1, 1, 0])
    m = metrics(y, np.array([0.1, 0.9, 0.8, 0.2]))
    assert m["AUC"] == 1.0 and m["ACC"] == 1.0
    assert np.isclose(ece(np.array([1, 1]), np.array([0.75, 0.75])), 0.25)
