"""ASSISTments 2012-2013 school data -> common interaction table (plan card P1-02, teacher's items A1 and A2).

The file is 3 GB; it is read in chunks with compact dtypes (reading it at once runs out of memory). Cleaning uses the
same rule order as ASSISTments 2009 (P1-01). Unlike 2009, the log mixes Skill Builder (``MasterySection``) with other
assignment types, so the rows outside ``MasterySection`` give the sub-log free of the Skill Builder stopping rule.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..skill_mapping import SkillMapper
from .skill_map import MapTagger, tagger
from . import schema
from .assist2009 import AdapterResult
from .cleaning import CleaningLog, kept_vs_dropped, skill_counts
from .skill_builder import skill_builder_table
from .splits import assign_splits

DATASET = "assist2012"
MASTERY = "MasterySection"
COLS = ["problem_log_id", "skill", "problem_id", "user_id", "assistment_id", "start_time", "problem_type", "original",
        "correct", "hint_count", "attempt_count", "ms_first_response", "type", "template_id"]
DTYPES = {"problem_log_id": "int64", "problem_id": "int32", "user_id": "int32", "assistment_id": "int32",
          "original": "int8", "correct": "float32", "hint_count": "float32", "attempt_count": "float32",
          "ms_first_response": "float64", "template_id": "float64", "skill": "string", "problem_type": "string",
          "type": "string", "start_time": "string"}
CATEGORICAL = ["skill", "problem_type", "type"]


def _compact(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    # start_time looks like "2012-09-28 15:11:27" (sometimes with fractional seconds) -> epoch milliseconds
    # (pandas 3 infers the resolution from the text, so convert the unit explicitly before taking integers)
    chunk["ts"] = pd.to_datetime(chunk.pop("start_time"), format="mixed").dt.as_unit("ms").astype("int64")
    chunk["skill"] = chunk["skill"].str.strip().replace("", pd.NA)
    for col in CATEGORICAL:
        chunk[col] = chunk[col].astype("category")
    return chunk


def read_raw(path: Path, encoding: str = "ISO-8859-15", chunksize: int = 400_000) -> pd.DataFrame:
    parts = [_compact(c) for c in pd.read_csv(path, encoding=encoding, usecols=COLS, dtype=DTYPES,
                                              chunksize=chunksize)]
    df = pd.concat(parts, ignore_index=True)
    for col in CATEGORICAL:  # categories differ between chunks; re-unify to keep memory low
        df[col] = df[col].astype("string").astype("category")
    return df


def clean(raw: pd.DataFrame, cfg: dict, mapper: SkillMapper | MapTagger | None = None,
          tag_source: str = "SkillMapper") -> AdapterResult:
    mapper = mapper or SkillMapper()
    min_rows = int(cfg["split"]["min_user_interactions"])
    seed = int(cfg["data_seed"])
    n_boot = int(cfg.get("bootstrap", {}).get("n_resamples", 1000))
    log = CleaningLog(event_cols=["problem_log_id"])

    df = log.start(raw)
    log.note("rows with a skill, counted on the raw file", int(raw["skill"].notna().sum()), len(raw))
    log.note("duplicate problem_log_id on the raw file", int(raw["problem_log_id"].duplicated().sum()), len(raw))
    df = log.apply(df, df[df["original"] == 1], "drop original == 0 (scaffolding sub-questions)")
    df = log.apply(df, df[df["skill"].notna()], "drop missing skill")
    df = log.apply(df, df[df["correct"].isin([0, 1])], "drop correct outside {0, 1} (partial credit 0.25/0.5/0.75)")
    df = log.apply(df, df.drop_duplicates(["problem_log_id"]), "drop duplicates on problem_log_id")
    n = df.groupby("user_id")["problem_log_id"].transform("size")
    df = log.apply(df, df[n >= min_rows], f"drop learners with < {min_rows} rows")
    users_before_mapping = int(df["user_id"].nunique())

    df = df.copy()
    df["source_skill"] = df["skill"].astype(str)
    tag = {s: mapper.to_tag(s) for s in df["source_skill"].unique()}
    df["skill_tag"] = df["source_skill"].map(tag)
    mapped = df["skill_tag"].notna()
    counts = skill_counts(df, DATASET)
    comparison = kept_vs_dropped(df, mapped, n_boot, seed, reason=mapper.unmapped_reason)
    unmapped_rows, unmapped_skills = int((~mapped).sum()), int(df.loc[~mapped, "source_skill"].nunique())
    df = log.apply(df, df[mapped], f"drop rows whose skill has no canonical tag ({tag_source})")
    n = df.groupby("user_id")["problem_log_id"].transform("size")
    df = log.apply(df, df[n >= min_rows], f"re-apply: drop learners with < {min_rows} rows after the skill filter")

    df = df.sort_values("problem_log_id", kind="mergesort")  # deterministic order for equal start_time
    out = pd.DataFrame({
        "dataset": DATASET,
        "user_id": df["user_id"].astype(str),
        "item_id": df["problem_id"].astype(str),
        "source_skill": df["source_skill"],
        "skill_tag": df["skill_tag"],
        "khan_slug": None,
        "ts": df["ts"].astype("int64"),
        "correct": df["correct"].astype("int8"),
        "rt_ms": schema.clean_rt(df["ms_first_response"], tuple(cfg["rt_clip_ms"])),
        "attempts": df["attempt_count"].fillna(0).clip(lower=0).astype("int16"),
        "hints": df["hint_count"].fillna(0).clip(lower=0).astype("int16"),
        "section_type": df["type"].astype(str),
    })
    out = schema.coerce(assign_splits(out, cfg))
    schema.validate(out)

    mastery = out["section_type"] == MASTERY
    sections = (out.groupby("section_type", observed=True)
                .agg(rows=("correct", "size"), users=("user_id", "nunique"), accuracy=("correct", "mean"))
                .reset_index().assign(dataset=DATASET))
    sb = pd.concat([skill_builder_table(out[mastery], DATASET, "MasterySection"),
                    skill_builder_table(out[~mastery], DATASET, "outside_MasterySection")], ignore_index=True)
    facts = {
        "users_raw": int(raw["user_id"].nunique()),
        "skills_raw": int(raw["skill"].nunique()),
        "users_ge_min_rows_before_mapping": users_before_mapping,
        "unmapped_rows": unmapped_rows, "unmapped_skills": unmapped_skills,
        "rows_outside_mastery": int((~mastery).sum()),
        "users_outside_mastery": int(out.loc[~mastery, "user_id"].nunique()),
        "date_range": f"{pd.to_datetime(raw['ts'].min(), unit='ms')} .. {pd.to_datetime(raw['ts'].max(), unit='ms')}",
    }
    return AdapterResult(out, {"a1_cleaning_assist2012": log.table(DATASET),
                               "a1_kept_vs_dropped_assist2012": comparison.assign(dataset=DATASET),
                               "a1_sections_assist2012": sections,
                               "a1_skill_builder_assist2012": sb, "a1_skill_counts_assist2012": counts}, facts)


def build(cfg: dict, repo: Path) -> AdapterResult:
    ds = cfg["datasets"][DATASET]
    raw = read_raw(repo / ds["path"], ds.get("encoding", "ISO-8859-15"), int(ds.get("chunksize", 400_000)))
    tag, source = tagger(ds, repo)
    return clean(raw, cfg, mapper=tag, tag_source=source)
