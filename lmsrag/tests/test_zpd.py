import numpy as np

from lmsrag.mastery.zpd import run, zone_stats
from lmsrag.tests.test_mastery_baselines import _df


def test_zone_stats():
    y = np.array([1, 0, 1, 1]); p = np.array([0.7, 0.7, 0.2, 0.9]); inz = np.array([True, True, False, False])
    s = zone_stats(y, p, inz)
    assert s["coverage"] == 0.5 and s["acc_in"] == 0.5 and np.isclose(s["gap"], 0.2)


def test_run_picks_band_on_validation_only():
    r = run(_df(users=60))
    t = r["table"]
    assert t["chosen"].sum() == 2  # the same band, flagged on validation and test
    assert {"old", "prob"} == set(t["rule"])
