"""Run LAYER 1:  sources registry -> exercise bank T -> interaction matrix R -> learner states P(u) -> SQLite.   python -m lmsrag.layer1_data.build"""
from __future__ import annotations

import json
import time

import numpy as np

from .. import config, data, profile
from ..skill_mapping import SkillMapper
from . import interaction_matrix as im
from . import learner_state as ls
from . import sources
from .state_store import StateStore


def build() -> dict:
    t0 = time.time()
    mapper = SkillMapper()
    report = {"sources": sources.source_registry()}

    bank = sources.build_exercise_bank(mapper)
    report["exercise_bank_T"] = sources.save_bank(bank)

    df = data.load_logs(mapper)
    R, users, items = im.build_R(df, ("history",))
    im.save(R, users, items)
    report["interaction_matrix_R"] = {"splits": ["history"], **im.stats(R)}
    rh, _ = im.build_R(df, ("history", "val", "test"))[0:2]
    report["interaction_matrix_R"]["density_all_splits"] = float(rh.nnz / (rh.shape[0] * rh.shape[1]))

    alpha, _ = profile.tune_alpha(df)
    priors, tau = profile.skill_priors(df)
    states = ls.build_states(df, alpha, priors, as_of=("history",), tau_base=tau)
    path = config.ART / "learner_state.sqlite"
    path.unlink(missing_ok=True)
    store = StateStore(path)
    store.bulk_put(states)
    S = [s.summary() for s in states.values()]
    col = lambda k: np.array([s[k] for s in S])
    q = lambda a: {f"p{p}": round(float(np.percentile(a, p)), 4) for p in (5, 25, 50, 75, 95)}
    report["learner_state_P_u"] = {"as_of": "history", "ema_alpha": alpha, "n_states": store.count(),
                                   "A": q(col("A")), "sigma": q(col("sigma")), "pace_per_100": q(col("pace_per_100")),
                                   "mean_rt_s": q(col("mean_rt_s")), "n_weak_skills": q(col("n_weak")), "n_strong_skills": q(col("n_strong")),
                                   "share_users_with_weak_skill": float((col("n_weak") > 0).mean()),
                                   "share_users_with_strong_skill": float((col("n_strong") > 0).mean())}
    # leakage guard: the stored state must not have seen a single val/test attempt
    hist_n = df[df.split == "history"].groupby("user_id").size()
    assert all(states[u].n == hist_n[u] for u in list(states)[:500]), "state contains non-history attempts"
    report["leakage_check"] = "state attempt counts == history row counts (500 users sampled)"
    store.close()
    report["elapsed_s"] = round(time.time() - t0, 1)
    (config.ART / "layer1_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    r = build()
    print(json.dumps({k: v for k, v in r.items() if k != "sources"}, indent=1))
