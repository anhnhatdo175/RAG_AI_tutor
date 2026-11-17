"""ASSISTments 2009-2010 Skill Builder -> common interaction table (plan card P1-01, teacher's item A1).

Cleaning follows the order of the original pipeline (``lmsrag.data.read_assistments`` then ``finalize_logs`` and
``filter_mapped_logs``) so the numbers stay comparable with earlier reports; every rule is logged in the cleaning
table. ``ts`` is ``order_id``: it orders a learner's events but is not a clock time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..skill_mapping import SkillMapper
from . import schema
from .cleaning import CleaningLog, kept_vs_dropped, skill_counts
from .skill_builder import skill_builder_table
from .splits import assign_splits

DATASET = "assist2009"
COLS = ["order_id", "user_id", "problem_id", "original", "correct", "skill_id", "skill_name", "ms_first_response",
        "attempt_count", "hint_count", "type"]


@dataclass
class AdapterResult:
    interactions: pd.DataFrame
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    facts: dict[str, object] = field(default_factory=dict)


def read_raw(path: Path, encoding: str = "latin-1") -> pd.DataFrame:
    return pd.read_csv(path, encoding=encoding, usecols=COLS, low_memory=False)


def clean(raw: pd.DataFrame, cfg: dict, mapper: SkillMapper | None = None) -> AdapterResult:
    """Apply the cleaning rules in order and return the interactions plus the report tables."""
    mapper = mapper or SkillMapper()
    min_rows = int(cfg["split"]["min_user_interactions"])
    seed = int(cfg["data_seed"])
    n_boot = int(cfg.get("bootstrap", {}).get("n_resamples", 1000))
    log = CleaningLog(event_cols=["order_id"])

    df = log.start(raw)
    log.note("duplicate rows on (order_id, skill_id), counted on the raw file",
             int(raw.duplicated(["order_id", "skill_id"]).sum()), len(raw))
    df = log.apply(df, df[df["original"] == 1], "drop original == 0 (scaffolding sub-questions)")
    df = log.apply(df, df[df["skill_id"].notna() & df["skill_name"].notna()], "drop missing skill_id or skill_name")
    df = log.apply(df, df[df["correct"].isin([0, 1])], "drop correct outside {0, 1}")
    df = log.apply(df, df.drop_duplicates(["order_id", "skill_id"]), "drop duplicates on (order_id, skill_id)")
    n = df.groupby("user_id")["order_id"].transform("size")
    df = log.apply(df, df[n >= min_rows], f"drop learners with < {min_rows} rows")

    df = df.copy()
    df["source_skill"] = df["skill_name"].astype(str)
    tag = {s: mapper.to_tag(s) for s in df["source_skill"].unique()}
    df["skill_tag"] = df["source_skill"].map(tag)
    mapped = df["skill_tag"].notna()
    counts = skill_counts(df, DATASET)
    comparison = kept_vs_dropped(df, mapped, n_boot, seed, reason=mapper.unmapped_reason)
    unmapped_rows = int((~mapped).sum())
    unmapped_pct = 100.0 * unmapped_rows / max(len(df), 1)
    df = log.apply(df, df[mapped], "drop rows whose skill has no canonical tag (SkillMapper)")
    n = df.groupby("user_id")["order_id"].transform("size")
    df = log.apply(df, df[n >= min_rows], f"re-apply: drop learners with < {min_rows} rows after the skill filter")

    out = pd.DataFrame({
        "dataset": DATASET,
        "user_id": df["user_id"].astype("int64").astype(str),
        "item_id": df["problem_id"].astype("int64").astype(str),
        "source_skill": df["source_skill"],
        "skill_tag": df["skill_tag"],
        "khan_slug": None,
        "ts": df["order_id"].astype("int64"),
        "correct": df["correct"].astype("int8"),
        "rt_ms": schema.clean_rt(df["ms_first_response"], tuple(cfg["rt_clip_ms"])),
        "attempts": df["attempt_count"].fillna(0).clip(lower=0).astype("int16"),
        "hints": df["hint_count"].fillna(0).clip(lower=0).astype("int16"),
        "section_type": df["type"].astype(str),
    })
    out = schema.coerce(assign_splits(out, cfg))
    schema.validate(out)

    sb = skill_builder_table(out, DATASET, "all")
    facts = {"unmapped_rows": unmapped_rows, "unmapped_pct": round(unmapped_pct, 2),
             "events_final": int(out[["user_id", "ts"]].drop_duplicates().shape[0])}
    return AdapterResult(out, {"a1_cleaning_assist2009": log.table(DATASET),
                               "a1_kept_vs_dropped": comparison.assign(dataset=DATASET),
                               "a1_skill_builder": sb, "a1_skill_counts_assist2009": counts}, facts)


def build(cfg: dict, repo: Path) -> AdapterResult:
    ds = cfg["datasets"][DATASET]
    return clean(read_raw(repo / ds["path"], ds.get("encoding", "latin-1")), cfg)
