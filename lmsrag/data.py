"""ASSISTments log loading, per-learner temporal split, and empirical item difficulty."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .skill_mapping import SkillMapper

# Keep all native ASSISTments identifiers.  `problem_id` identifies a question,
# `assistment_id` groups an assistment/scaffold, and `template_id` groups similar
# question templates; they are distinct namespaces and must not be conflated.
COLS = ["order_id", "user_id", "assistment_id", "problem_id", "template_id", "original", "correct", "skill_id", "skill_name", "ms_first_response", "attempt_count", "hint_count"]
BUCKETS = np.asarray(config.DIFFICULTY_LEVELS)


def read_assistments(min_user_rows: int = 20) -> pd.DataFrame:
    """Raw ASSISTments skill-builder rows -> the common log schema (see ``finalize_logs``)."""
    df = pd.read_csv(config.ASSISTMENTS_CSV, encoding="latin-1", usecols=COLS, low_memory=False)
    df = df[(df.original == 1) & df.skill_id.notna() & df.skill_name.notna() & df.correct.isin([0, 1])]
    # a multi-skill problem is logged once per skill: keep one row per (event, skill)
    df = df.drop_duplicates(["order_id", "skill_id"]).copy()
    df["skill_id"] = df.skill_id.astype(int)
    df["correct"] = df.correct.astype(int)
    df["ms"] = df.ms_first_response.clip(500, 120_000)
    return df


def finalize_logs(df: pd.DataFrame, mapper: SkillMapper, min_user_rows: int = 20, shrink_m: float = 5.0, min_item_n: int = 3) -> pd.DataFrame:
    """ASSISTments log schema -> evaluation frame.

    Required input columns: order_id (monotone per user = time order), user_id, problem_id (item), correct (0/1),
    skill_id (int), skill_name (a name ``mapper.to_tag`` understands, or any string -> tag None), ms (response time, ms).
    """
    df = df.sort_values(["user_id", "order_id", "skill_id"]).reset_index(drop=True)
    n = df.groupby("user_id")["order_id"].transform("size")
    df = df[n >= min_user_rows].reset_index(drop=True)

    # ---- per-learner chronological 70/15/15 split (anti-leakage) -------------------------
    pos = df.groupby("user_id").cumcount().to_numpy()
    n = df.groupby("user_id")["order_id"].transform("size").to_numpy()
    r = pos / n
    df["split"] = np.where(r < 0.70, "history", np.where(r < 0.85, "val", "test"))
    df["opp"] = df.groupby(["user_id", "skill_id"]).cumcount()
    df["tag"] = df.skill_name.map(mapper.to_tag)

    # ---- empirical item difficulty, estimated on HISTORY only ----------------------------------
    h = df[df.split == "history"]
    skill_mean = h.groupby("skill_id").correct.mean()
    it = h.groupby(["problem_id", "skill_id"]).correct.agg(["sum", "size"]).reset_index()
    it["p"] = (it["sum"] + shrink_m * it.skill_id.map(skill_mean)) / (it["size"] + shrink_m)
    it["d_raw"] = 1.0 - it.p
    it = it[it["size"] >= min_item_n].copy()
    q = it.groupby("skill_id").d_raw.quantile([1 / 3, 2 / 3]).unstack()
    it = it.join(q, on="skill_id")
    it["item_d"] = np.where(it.d_raw <= it[1 / 3], BUCKETS[0], np.where(it.d_raw <= it[2 / 3], BUCKETS[1], BUCKETS[2]))
    df = df.merge(it[["problem_id", "skill_id", "item_d", "d_raw"]], on=["problem_id", "skill_id"], how="left")
    df["item_known"] = df.item_d.notna()
    df["item_d"] = df.item_d.fillna(BUCKETS[1])
    df["bucket"] = np.searchsorted(BUCKETS, df.item_d.to_numpy() - 1e-9)    # 0,1,2
    return df


def load_logs(mapper: SkillMapper, min_user_rows: int = 20, shrink_m: float = 5.0, min_item_n: int = 3) -> pd.DataFrame:
    """ASSISTments (unchanged behaviour): read -> finalize."""
    return finalize_logs(read_assistments(min_user_rows), mapper, min_user_rows, shrink_m, min_item_n)


def filter_mapped_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Return only interactions with a resolved canonical skill tag.

    Unmapped names are represented as null by ``SkillMapper.to_tag``; checking
    only for the literal string ``"unmapped"`` would incorrectly retain nulls.
    """
    if "tag" not in df:
        raise ValueError("log frame must contain the canonical 'tag' column")
    return df.loc[df["tag"].notna() & df["tag"].ne("unmapped")].copy()
