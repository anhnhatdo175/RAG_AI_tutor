import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score

from lmsrag.ranking.tasks import diversify, itemknn_sim, topn
from lmsrag.retrieval.ltr import ndcg_at
from lmsrag.stats.bootstrap import WeightedAUC, holm, p_two_sided


def test_weighted_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y, p, w = rng.integers(0, 2, 400), np.round(rng.random(400), 1), rng.integers(0, 3, 400)
    assert np.isclose(WeightedAUC(y, p)(w), roc_auc_score(y, p, sample_weight=w))


def test_holm_and_p():
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])
    assert p_two_sided(np.array([1.0, 2.0, 3.0])) == 0.0


def test_itemknn_topn_and_diversify():
    X = sparse.csr_matrix(np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=float))
    S = itemknn_sim(X, nn=2)
    assert S[0, 1] > 0 and S[0, 2] == 0
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    assert topn(scores, np.array([True, False, False, False]), 2).tolist() == [1, 2]
    skills = np.array([0, 0, 1, 1])
    assert diversify(np.array([0, 1, 2, 3]), scores, skills, lam=1.0, k=2).tolist() == [0, 2]


def test_ndcg_at():
    assert ndcg_at(np.array([3.0, 2.0, 1.0]), np.array([2, 0, 1]), k=3) > ndcg_at(np.array([1.0, 2.0, 3.0]),
                                                                                   np.array([2, 0, 1]), k=3)
