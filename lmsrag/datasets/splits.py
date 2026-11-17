"""One split protocol for every log (plan tab "Quy ước chung", section 4).

Per learner, in ``ts`` order: the first 70% of events are ``history``, the next 15% ``validation``, the last 15% ``test``.
The unit is the *event* ``(user_id, ts)``, not the row: ASSISTments 2009 logs a multi-skill question once per skill
with the same ``ts``, and those rows must land in the same split, otherwise a row of the event would be visible as
history while its twin is predicted. For logs with one row per event the two definitions coincide.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def _fractions(cfg: Mapping) -> tuple[float, float]:
    split = cfg.get("split", cfg)
    h, v, t = float(split["history"]), float(split["validation"]), float(split["test"])
    if not np.isclose(h + v + t, 1.0):
        raise ValueError(f"split fractions must sum to 1, got {h + v + t}")
    return h, h + v


def assign_splits(df: pd.DataFrame, cfg: Mapping) -> pd.DataFrame:
    """Return a copy sorted by (user_id, ts) with a ``split`` column.

    An event with rank ``r`` (0-based) among ``n`` events of its learner goes to history if ``r/n < h``, to validation
    if ``r/n < h+v``, otherwise to test. So each part differs from its target share by less than one event.
    """
    h_cut, v_cut = _fractions(cfg)
    out = df.sort_values(["user_id", "ts"], kind="mergesort").reset_index(drop=True)
    rank = out.groupby("user_id", sort=False)["ts"].rank(method="dense").to_numpy() - 1.0
    n_events = out.groupby("user_id", sort=False)["ts"].transform("nunique").to_numpy()
    r = rank / n_events
    out["split"] = pd.array(np.where(r < h_cut, "history", np.where(r < v_cut, "validation", "test")), dtype="string")
    return out


def sample_users(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Keep ``n`` learners drawn uniformly without replacement (all learners when ``n`` is None or too large)."""
    users = np.sort(df["user_id"].unique())
    if n is None or n >= len(users):
        return df
    keep = np.random.default_rng(seed).choice(users, size=n, replace=False)
    return df[df["user_id"].isin(keep)]


def truncate_history(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Cold-start view: every learner keeps only the first ``k`` history events, loses validation, keeps test.

    Evaluation then runs on the same test rows as the full-history setting, so results for k in {0..4} are comparable.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    hist = df[df["split"] == "history"]
    rank = hist.groupby("user_id", sort=False)["ts"].rank(method="dense")
    kept_hist = hist[rank <= k]
    return pd.concat([kept_hist, df[df["split"] == "test"]]).sort_values(["user_id", "ts"], kind="mergesort")
