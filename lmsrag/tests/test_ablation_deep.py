import zipfile

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from lmsrag.ranking import ablation as A
from lmsrag.retrieval import gold_ablation as GA


def test_ablation_variants_use_known_features():
    for name, feats in A.VARIANTS.items():
        assert set(feats) <= set(A.EXT), name
        assert len(feats) == len(set(feats))
    assert "rt_dev" not in A.VARIANTS["full"] and "rt_dev" in A.VARIANTS["plus_rt"]
    assert not {"p_hat", "zpd_gap", "rank_zpd"} & set(A.VARIANTS["no_correctness"])


def test_retrieval_variants():
    n = 6
    rng = np.random.default_rng(0)
    sc = GA.variant_scores(rng.random(n), rng.random(n), rng.uniform(-1, 1, n), rng.random(n),
                           np.array([1, 2, 3, 4, 5, 3.0]), np.full(n, 0.6), 0.5, 4)
    assert set(sc) == set(GA.VARIANTS)
    assert all(len(v) == n for v in sc.values())
    # semantic_only ignores difficulty; no_semantic ignores the semantic term
    sc2 = GA.variant_scores(np.arange(n, dtype=float), rng.random(n), np.zeros(n), np.zeros(n), np.ones(n),
                            np.full(n, 0.6), 0.5, 4)
    assert np.all(np.diff(sc2["semantic_only"]) > 0)


def test_rt_matrix():
    tr = pd.DataFrame({"user_id": [1, 1, 2, 2], "source_skill": ["a", "a", "a", "b"],
                       "rt_ms": [100.0, 100.0, 10000.0, np.nan]})
    m = A.rt_matrix(tr, np.array([1, 2, 3]), pd.Index(["a", "b"]))
    assert m.shape == (3, 2) and m[0, 0] < 0 < m[1, 0] and m[2].sum() == 0 and m[1, 1] == 0


def test_kaggle_unpack_refuses_other_files(tmp_path):
    from lmsrag import kaggle_bundle as K
    z = tmp_path / "r.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("lmsrag/evil.py", "x")
    with pytest.raises(SystemExit):
        K.unpack(z)


def _toy_log(n_users=12, n=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        for t in range(n):
            sp = "history" if t < 20 else ("validation" if t < 25 else "test")
            k = int(rng.integers(0, 3))
            rows.append({"user_id": u, "ts": t, "item_id": t % 7, "source_skill": f"s{k}",
                         "correct": int(rng.random() < 0.3 + 0.2 * k), "split": sp})
    return pd.DataFrame(rows)


def test_deep_kt_predicts_every_later_row_once():
    pytest.importorskip("torch")
    from lmsrag.mastery import deep_kt as D
    df = _toy_log()
    for kind in ("dkt", "sakt"):
        pred, choice = D.run(df, kind, 0, emb=16, seq_len=8, heads=2, max_epochs=2, batch=8)
        later = np.flatnonzero(df["split"].ne("history").to_numpy())
        assert sorted(pred["row"]) == list(later)
        assert (pred["y"].to_numpy() == df["correct"].to_numpy()[pred["row"]]).all()
        assert pred[kind].between(0, 1).all() and choice["epochs"] >= 1


def test_lightgcn_ranks_unseen_items():
    pytest.importorskip("torch")
    from lmsrag.ranking.lightgcn import LightGCN
    X = sparse.csr_matrix(np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float))
    m = LightGCN(X, seed=0, dim=8, batch=4, device="cpu")
    assert np.isfinite(m.epoch())
    top = m.rank(np.array([0, 1, 2]), X, k=5)
    for r in range(3):
        assert not set(top[r]) & set(X[r].indices)


def test_deep_kt_attempt_rows_do_not_see_each_other():
    torch = pytest.importorskip("torch")
    from lmsrag.mastery import deep_kt as D
    df = _toy_log(n_users=4)
    # one attempt tagged with two skills: a copy of every test row with another skill, same item and time
    dup = df[df["split"] == "test"].assign(source_skill="s9")
    df = pd.concat([df, dup]).sort_values(["user_id", "ts"], kind="mergesort").reset_index(drop=True)
    seqs, num_c = D.sequences(df)
    assert (D.context_index(np.array([0, 1, 1, 2])) == np.array([0, 0, 2])).all()
    for kind in ("dkt", "sakt"):
        torch.manual_seed(0)
        model = D.build_model(kind, num_c, 8, 8, 2, 0.0)
        rows, p, _ = D.predict(model, seqs, 8, (2,), "cpu", 0.5)
        # flip the answer of the first row of each learner's last attempt: its second row must not move
        flipped, second = [], []
        for u, c, r, s, rw, g in seqs:
            r2 = r.copy()
            r2[len(r) - 2] = 1 - r2[len(r) - 2]
            assert g[-1] == g[-2]
            flipped.append((u, c, r2, s, rw, g))
            second.append(rw[-1])
        rows2, p2, _ = D.predict(model, flipped, 8, (2,), "cpu", 0.5)
        a = pd.Series(p, index=rows)
        b = pd.Series(p2, index=rows2)
        assert np.allclose(a[second], b[second], atol=1e-6), kind
