import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from lmsrag.labels.agreement import agreement_rate, audit, bootstrap_ci, cohen_kappa, pabak, spearman, wilson

A = [1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
B = [1, 0, 0, 0, 1, 1, 1, 1, 0, 1]


def test_kappa_hand_worked():
    # agreement 8/10; both say 1 six times, so chance agreement 0.6·0.6 + 0.4·0.4 = 0.52; κ = (0.8 − 0.52) / 0.48
    assert agreement_rate(A, B) == pytest.approx(0.8)
    assert cohen_kappa(A, B) == pytest.approx(0.28 / 0.48)
    assert cohen_kappa(A, A) == pytest.approx(1.0)
    assert pabak(A, B) == pytest.approx(0.6)
    # prevalence paradox: 18/20 agreement on mostly-1 labels gives κ < 0 but PABAK 0.8
    many = [1] * 18 + [0, 1]
    other = [1] * 18 + [1, 0]
    assert cohen_kappa(many, other) < 0 and pabak(many, other) == pytest.approx(0.8)
    assert math.isnan(cohen_kappa([1, 1, 1], [1, 1, 1]))  # one shared label only: undefined


def test_kappa_matches_sklearn():
    rng = np.random.default_rng(0)
    for k in (2, 3, 5):
        a = rng.integers(0, k, 200)
        b = np.where(rng.random(200) < 0.6, a, rng.integers(0, k, 200))
        assert cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))
        assert cohen_kappa(a, b, "quadratic") == pytest.approx(cohen_kappa_score(a, b, weights="quadratic"))


def test_spearman_matches_scipy_with_ties():
    a = [1, 2, 2, 3, 5, 5, 4, 1]
    b = [2, 1, 3, 3, 5, 4, 4, 1]
    assert spearman(a, b) == pytest.approx(spearmanr(a, b).statistic)


def test_wilson_known_values():
    lo, hi = wilson(8, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.4902, 0.9433)
    lo, hi = wilson(10, 10)  # stays inside [0, 1] at the boundary
    assert hi == pytest.approx(1.0) and hi <= 1.0 and lo == pytest.approx(0.7225, abs=1e-4)


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    a, b = np.array(A * 5), np.array(B * 5)
    ci = bootstrap_ci(cohen_kappa, a, b, seed=2026, n_boot=300)
    assert ci == bootstrap_ci(cohen_kappa, a, b, seed=2026, n_boot=300)
    assert ci[0] < cohen_kappa(a, b) < ci[1]


def test_audit_before_and_after_adjudication():
    items = [f"m{i}" for i in range(10)]
    key = pd.DataFrame({"item": items, "pair_type": ["khan_template"] * 6 + ["rule_chunk"] * 4,
                        "method": ["manual_review"] * 6 + ["rule_text"] * 4})
    l2 = pd.Series([str(x) for x in A], index=items)
    l3 = pd.Series([str(x) for x in B], index=items)
    before = audit(key, l2, l3, None, seed=1).set_index(["group", "metric"])["value"]
    assert before[("all", "kappa_before_adjudication")] == pytest.approx(0.5833, abs=1e-4)
    assert before[("all", "pairs_to_adjudicate")] == 2
    assert ("all", "accuracy_after_adjudication") not in before.index  # not until every disagreement is decided

    final = pd.Series({"m1": "1", "m5": "0"})  # the two disagreements (A, B differ at positions 1 and 5)
    after = audit(key, l2, l3, final, seed=1).set_index(["group", "metric"])
    row = after.loc[("all", "accuracy_after_adjudication")]
    assert row["value"] == pytest.approx(0.6)  # finals: six 1s
    assert (row["ci_low"], row["ci_high"]) == pytest.approx(wilson(6, 10), abs=1e-4)
    # the four rule pairs (positions 6-9) are all agreed: 1, 1, 0, 1
    assert after.loc[("rule_text", "accuracy_after_adjudication"), "value"] == pytest.approx(0.75)
