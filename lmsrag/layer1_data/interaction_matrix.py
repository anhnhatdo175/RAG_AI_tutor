"""
LAYER 1 / interaction matrix (docx 2.2.2):   R in [0,1]^{n x m},   R[u,i] = score(u,i) / max_score.

n learners, m items (ASSISTments problem_id).  Stored sparse (CSR, float32).  Built from a chosen set of splits only, so the
matrix used to train a recommender never contains validation/test attempts.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .. import config


def score(df: pd.DataFrame, mode: str = "binary") -> tuple[np.ndarray, float]:
    """Returns (score, max_score).  binary: first-try correctness.  hint_discounted: correct * (1 - 0.25*hints)."""
    if mode == "binary":
        return df.correct.to_numpy(dtype=float), 1.0
    if mode == "hint_discounted":
        return df.correct.to_numpy(dtype=float) * np.clip(1 - 0.25 * df.hint_count.fillna(0).to_numpy(), 0, 1), 1.0
    raise ValueError(mode)


def build_R(df: pd.DataFrame, splits=("history",), mode: str = "binary"):
    d = df[df.split.isin(splits)].copy()
    s, mx = score(d, mode)
    d["R"] = s / mx
    g = d.groupby(["user_id", "problem_id"]).R.mean().reset_index()      # repeated (u,i) -> mean
    users, items = np.sort(g.user_id.unique()), np.sort(g.problem_id.unique())
    ui = np.searchsorted(users, g.user_id.to_numpy()); ii = np.searchsorted(items, g.problem_id.to_numpy())
    R = sp.csr_matrix((g.R.to_numpy(dtype="float32"), (ui, ii)), shape=(len(users), len(items)))
    assert R.data.min() >= 0 and R.data.max() <= 1
    return R, users, items


def stats(R: sp.csr_matrix) -> dict:
    n, m = R.shape
    per_user = np.diff(R.indptr)
    per_item = np.diff(R.tocsc().indptr)
    q = lambda a: {f"p{p}": float(np.percentile(a, p)) for p in (5, 25, 50, 75, 95)}
    return {"n_users": int(n), "n_items": int(m), "nnz": int(R.nnz), "density": float(R.nnz / (n * m)),
            "mean_score": float(R.data.mean()), "per_user_interactions": q(per_user), "per_item_interactions": q(per_item),
            "users_with_ge20": int((per_user >= 20).sum()), "items_with_lt3": int((per_item < 3).sum())}


def save(R, users, items, name="interaction_matrix_history"):
    sp.save_npz(config.ART / f"{name}.npz", R)
    (config.ART / f"{name}_index.json").write_text(json.dumps({"users": users.tolist(), "items": items.tolist()}))
