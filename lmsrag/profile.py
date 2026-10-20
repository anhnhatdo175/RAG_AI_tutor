"""
STEP 3 - Offline learner modelling from the interaction log.

For learner u on skill k we keep, updated chronologically after every attempt:

  Mastery (EMA, docx 2.7.2):      M_k(t) = a * S(t) + (1 - a) * M_k(t-1)          S(t) = first-try correctness
  Profile vector (docx 2.3.3):    p_{u,k} = normalise( sum_t w(t) * v_interaction(t) )
        w(t) = exp(-lambda (T - t)) * [ gamma * c(t) + (1 - gamma) * min(1, tau_base / tau(t)) ]
        c(t)       correctness, tau(t) response time, tau_base = skill median response time (history)
        v_interaction(t) = centroid of the corpus chunks having the same (skill_tag, difficulty bucket)
                           as the attempted item  (ASSISTments ships no item text, so the interaction is
                           anchored in the content space through its skill + difficulty).
Because v_interaction only takes 3 values per skill, the decayed sum is stored as 3 weights W_b and the
vector is rebuilt on demand:  p = sum_b W_b * centroid_b / ||.||   (exactly the docx formula, O(1) update).
Every row stores the state *before* the attempt = what the recommender would have seen when serving it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

from . import config


def skill_priors(df: pd.DataFrame):
    h = df[df.split == "history"]
    return h.groupby("skill_id").correct.mean(), h.groupby("skill_id").ms.median()


def run_states(df: pd.DataFrame, alpha: float, with_profile: bool = True,
               half_life: float = config.HALF_LIFE_OPPS, gamma: float = config.GAMMA) -> pd.DataFrame:
    """Returns df-aligned columns: M_before, n_before, W0, W1, W2 (state before each attempt)."""
    p0, tau_base = skill_priors(df)
    lam = math.log(2) / half_life
    decay = math.exp(-lam)
    order = np.lexsort((df.opp.to_numpy(), df.skill_id.to_numpy(), df.user_id.to_numpy()))
    u, k = df.user_id.to_numpy()[order], df.skill_id.to_numpy()[order]
    y, ms, b = df.correct.to_numpy()[order], df.ms.to_numpy()[order], df.bucket.to_numpy()[order]
    pk = p0.reindex(k).fillna(p0.mean()).to_numpy()
    tb = tau_base.reindex(k).fillna(tau_base.median()).to_numpy()
    n = len(df)
    M_b, n_b = np.empty(n), np.empty(n, dtype=np.int32)
    W = np.zeros((n, 3))
    M, cnt, w = 0.0, 0, [0.0, 0.0, 0.0]
    for i in range(n):
        if i == 0 or u[i] != u[i - 1] or k[i] != k[i - 1]:
            M, cnt, w = pk[i], 0, [0.0, 0.0, 0.0]
        M_b[i], n_b[i] = M, cnt
        if with_profile:
            W[i] = w
            speed = min(1.0, tb[i] / ms[i])
            q = gamma * y[i] + (1 - gamma) * speed
            w = [x * decay for x in w]
            w[b[i]] += q
        M = alpha * y[i] + (1 - alpha) * M
        cnt += 1
    out = pd.DataFrame(index=df.index[order])
    out["M_before"], out["n_before"] = M_b, n_b
    out[["W0", "W1", "W2"]] = W
    return out.sort_index()


def tune_alpha(df: pd.DataFrame, grid=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)):
    """Pick the EMA rate that best predicts next-attempt correctness on VALIDATION (log-loss)."""
    res = {}
    mask = (df.split == "val").to_numpy()
    y = df.correct.to_numpy()[mask]
    for a in grid:
        m = run_states(df, a, with_profile=False).M_before.to_numpy()[mask].clip(1e-3, 1 - 1e-3)
        ll = -np.mean(y * np.log(m) + (1 - y) * np.log(1 - m))
        res[a] = float(ll)
    return min(res, key=res.get), res


def mastery_report(df: pd.DataFrame, M_before: np.ndarray) -> dict:
    """Docx 2.9.4 - how well does the EMA mastery predict the next response? (test split)"""
    p0, _ = skill_priors(df)
    t = (df.split == "test").to_numpy()
    y = df.correct.to_numpy()
    last = df.groupby(["user_id", "skill_id"]).correct.shift(1).fillna(df.skill_id.map(p0)).to_numpy()
    h = df[df.split == "history"]
    user_mean = h.groupby("user_id").correct.mean()
    preds = {
        "Global mean": np.full(len(df), h.correct.mean()),
        "Skill mean (history)": df.skill_id.map(p0).fillna(h.correct.mean()).to_numpy(),
        "User mean (history)": df.user_id.map(user_mean).fillna(h.correct.mean()).to_numpy(),
        "Last answer on skill": last,
        "EMA mastery (ours)": M_before,
    }
    rep = {}
    for name, p in preds.items():
        rep[name] = {"AUC": float(roc_auc_score(y[t], p[t])), "RMSE": float(np.sqrt(np.mean((y[t] - p[t]) ** 2))),
                     "MAE": float(mean_absolute_error(y[t], p[t]))}
    return rep
