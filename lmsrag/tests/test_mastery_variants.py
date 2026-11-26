import numpy as np

from lmsrag.mastery import variants as V
from lmsrag.tests.test_mastery_baselines import _df


def test_fixed_mode_equals_p2_01_ema():
    from lmsrag.mastery import baselines as B
    ctx = V.Ctx(_df())
    np.testing.assert_allclose(ctx.ema(mode=0, alpha=0.2), B.ema(ctx.d, ctx.prior, 0.2))


def test_diff_weight_moves_more_on_surprise():
    y = np.array([1.0, 1.0]); uk = np.array([0, 1]); prior = np.array([0.5, 0.5]); n = np.zeros(2)
    hard, easy = np.array([0.2, 0.2]), np.array([0.9, 0.9])
    # second group's state is not observed, so compare one-step updates through a 2-row sequence of the same group
    uk = np.array([0, 0])
    a = V._ema_general(y, uk, prior, n, hard, np.full(2, 0.2), 0.1, 2.0, 1.0, 2, 1)
    b = V._ema_general(y, uk, prior, n, easy, np.full(2, 0.2), 0.1, 2.0, 1.0, 2, 1)
    assert a[1] > b[1] > 0.5


def test_item_easiness_excludes_own_history_row():
    ctx = V.Ctx(_df(users=1, n=10))  # each item answered once -> history rows fall back to the skill prior
    assert np.allclose(ctx.e_item[ctx.h], np.clip(ctx.prior[ctx.h], 0.02, 0.98))


def test_run_outputs_calibrated_models():
    r = V.run(_df())
    models = set(r["table"]["model"])
    best = r["choices"]["best_raw"]
    assert {f"{best}+platt", f"{best}+isotonic", f"{best}+item"} <= models
