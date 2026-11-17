"""How the ASSISTments Skill Builder mechanism shapes a log (teacher's item A1).

A Skill Builder assignment keeps giving questions on one skill until the student answers 3 in a row correctly, so a
(learner, skill) sequence tends to end on correct answers and "repeat the last answer on this skill" looks better
than it would on ordinary practice. ``skill_builder_table`` measures this for any log in the common schema; run it
on ASSISTments 2009 (all MasterySection) and on the 2012 rows outside MasterySection to compare.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

MAX_POSITION = 10


def _sequences(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["user_id", "source_skill", "ts"], kind="mergesort").reset_index(drop=True)
    g = d.groupby(["user_id", "source_skill"], sort=False)
    d["position"] = g.cumcount() + 1
    d["prev_correct"] = g["correct"].shift(1)
    return d


def skill_builder_table(df: pd.DataFrame, dataset: str, subset: str) -> pd.DataFrame:
    """Long table (dataset, subset, metric, value) of Skill Builder diagnostics for one log or sub-log."""
    d = _sequences(df)
    rows: list[dict] = []

    def add(metric, value):
        rows.append({"dataset": dataset, "subset": subset, "metric": metric, "value": value})

    keys = ["user_id", "source_skill"]
    g = d.groupby(keys, sort=False)["correct"]
    length = g.size()
    from_end = d.groupby(keys, sort=False)["position"].transform("max") - d["position"]
    tail = d[from_end < 3].groupby(keys, sort=False)["correct"].sum().reindex(length.index)
    last3 = (length >= 3) & (tail == 3)
    add("rows", len(d))
    add("sequences", int(len(length)))
    add("sequence_length_mean", round(float(length.mean()), 3))
    add("sequence_length_median", float(length.median()))
    add("share_sequences_ending_3_correct", round(float(last3.mean()), 4))
    add("share_sequences_len_ge3_ending_3_correct", round(float(last3[length >= 3].mean()), 4))
    add("share_last_answer_correct", round(float(g.last().mean()), 4))
    add("accuracy_overall", round(float(d["correct"].mean()), 4))
    pos = d["position"].clip(upper=MAX_POSITION + 1)
    for p, acc in d.groupby(pos)["correct"].mean().items():
        add(f"accuracy_position_{p if p <= MAX_POSITION else f'{MAX_POSITION + 1}+'}", round(float(acc), 4))

    for split in ("all", "test"):
        part = d[d["prev_correct"].notna()]
        if split == "test":
            part = part[part["split"] == "test"]
        if part["correct"].nunique() == 2:
            auc = roc_auc_score(part["correct"], part["prev_correct"])
            add(f"last_answer_auc_{split}", round(float(auc), 4))
            add(f"last_answer_rows_{split}", int(len(part)))
    return pd.DataFrame(rows)
