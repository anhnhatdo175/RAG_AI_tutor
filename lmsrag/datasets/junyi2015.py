"""Junyi Academy 2015 problem log -> common interaction table (plan card P1-03, teacher's item A2).

Junyi ran the Khan Academy exercise templates (in Chinese), so a Junyi ``exercise`` name equals a khan-exercises
template slug. That is the only log in the project where a response can be joined to the content of the question.

Read in chunks of ``chunksize`` rows (the CSV is 2.7 GB, 25.9 M rows). ``time_done`` is in microseconds, ``time_taken``
in seconds; negative times exist and become missing. ``correct`` is Junyi's own flag (right at the first attempt).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..skill_mapping import SkillMapper
from . import schema
from .assist2009 import AdapterResult
from .cleaning import CleaningLog
from .splits import assign_splits, sample_users

DATASET = "junyi2015"
COLS = ["user_id", "exercise", "problem_type", "time_done", "time_taken", "correct", "count_attempts", "count_hints",
        "review_mode"]
DTYPES = {"user_id": "int32", "exercise": "category", "problem_type": "category", "time_done": "int64",
          "time_taken": "float32", "correct": "string", "count_attempts": "float32", "count_hints": "float32",
          "review_mode": "string"}
DEFAULT_MAPPING = "lmsrag/data/mappings/skill_map_v1.csv"


def _bool(s: pd.Series) -> pd.Series:
    return s.str.strip().str.lower().map({"true": True, "false": False}).astype("boolean")


def read_raw(path: Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    parts = []
    for c in pd.read_csv(path, usecols=COLS, dtype=DTYPES, chunksize=chunksize):
        c["correct"] = _bool(c["correct"])
        c["review_mode"] = _bool(c["review_mode"])
        parts.append(c)
    df = pd.concat(parts, ignore_index=True)
    for col in ("exercise", "problem_type"):
        df[col] = df[col].astype("string").astype("category")
    return df


def khan_tag_lists(mapping_csv: Path, mapper: SkillMapper) -> dict[str, list[str]]:
    """khan_slug -> sorted canonical tags, through the reverse of the ASSISTments-to-Khan dictionary.

    Uses the dictionary's ``canonical_tag`` column when present (v1), otherwise ``mapper.to_tag`` of the log skill.
    """
    if not mapping_csv.is_file():
        return {}
    m = pd.read_csv(mapping_csv)
    skill_col = "source_skill" if "source_skill" in m.columns else "assistments_skill"
    m = m[m["khan_slug"].notna()]
    out: dict[str, set] = {}
    for _, r in m.iterrows():
        tag = r["canonical_tag"] if "canonical_tag" in m.columns else mapper.to_tag(r[skill_col])
        if isinstance(tag, str) and tag:
            out.setdefault(r["khan_slug"], set()).add(tag)
    return {slug: sorted(tags) for slug, tags in out.items()}


def khan_tags(mapping_csv: Path, mapper: SkillMapper) -> dict[str, str]:
    """khan_slug -> its first canonical tag (the value stored as ``skill_tag``)."""
    return {slug: tags[0] for slug, tags in khan_tag_lists(mapping_csv, mapper).items()}


def khan_type_tag_lists(mapping_csv: Path, problem_types_csv: Path) -> dict[tuple[str, str], list[str]]:
    """(khan_slug, problem-type key) -> sorted canonical tags, for templates whose dictionary rows name types.

    From map v1.1 a pair may cover only some problem types of a template (``khan_types``: type ids joined by ';'),
    e.g. Median -> mean_median_and_mode only for type ``median``. A row with no types covers every type. The key is
    the type id or its position, as Junyi logs either; a type no pair covers maps to an empty list.
    """
    if not mapping_csv.is_file() or not problem_types_csv.is_file():
        return {}
    m = pd.read_csv(mapping_csv, dtype=str, keep_default_na=False)
    if "khan_types" not in m.columns:
        return {}
    m = m[(m["khan_slug"] != "") & (m["canonical_tag"] != "")]
    split = set(m.loc[m["khan_types"] != "", "khan_slug"])
    t = pd.read_csv(problem_types_csv, dtype=str)
    out: dict[tuple[str, str], list[str]] = {}
    by_index: dict[tuple[str, str], list[str]] = {}
    for slug, tid, idx in zip(t["slug"], t["problem_id"], t["problem_index"]):
        if slug not in split:
            continue
        rows = m[m["khan_slug"] == slug]
        tags = sorted({tag for tag, types in zip(rows["canonical_tag"], rows["khan_types"])
                       if not types or tid in {x.strip() for x in types.split(";")}})
        out[(slug, tid)], by_index[(slug, idx)] = tags, tags
    # a position never overrides a type id that reads the same (order_of_operations_2 has ids "1", "2", ...)
    return {**by_index, **out}


def template_types(problem_types_csv: Path) -> set[tuple[str, str]]:
    """(template slug, problem-type key) pairs; a key is the type's id or its position, as Junyi logs either.

    Built by ``lmsrag.data_tools.khan_problem_types`` from the 2016 templates; Junyi ran older versions, so a few
    renamed types (e.g. ``subtraction-2``) do not match.
    """
    if not problem_types_csv.is_file():
        return set()
    t = pd.read_csv(problem_types_csv, dtype=str)
    return set(zip(t["slug"], t["problem_id"])) | set(zip(t["slug"], t["problem_index"]))


def coverage(frame: pd.DataFrame, template_slugs: set, usable_slugs: set,
             khan_types: set[tuple[str, str]]) -> dict[str, float | None]:
    """Share of rows on an exercise with a Khan template / a usable template, and, among template rows, the share
    whose problem_type is a type of that template (id or position). Computed on (exercise, problem_type) counts for speed."""
    g = frame.groupby(["exercise", "problem_type"], observed=True).size().rename("n").reset_index()
    g["exercise"], g["problem_type"] = g["exercise"].astype(str), g["problem_type"].astype(str)
    total = g["n"].sum()
    on_t = g["exercise"].isin(template_slugs)
    matched = [(e, p) in khan_types for e, p in zip(g["exercise"], g["problem_type"])]
    t_rows = g.loc[on_t, "n"].sum()
    return {"share_rows_on_khan_template": round(float(t_rows / total), 4),
            "share_rows_on_usable_template": round(float(g.loc[g["exercise"].isin(usable_slugs), "n"].sum() / total), 4),
            "share_template_rows_with_known_problem_type":
                round(float(g.loc[on_t & pd.Series(matched, index=g.index), "n"].sum() / t_rows), 4) if t_rows else None}


def clean(raw: pd.DataFrame, exercises: pd.DataFrame, templates: pd.DataFrame, cfg: dict,
          tags: dict[str, str] | None = None, khan_types: set[tuple[str, str]] | None = None,
          type_tags: dict[str, str] | None = None) -> AdapterResult:
    """``tags``: exercise -> tag; ``type_tags``: "exercise:problem_type" -> tag, for templates split by type
    (wins over ``tags``; a type no dictionary pair covers falls back to the Junyi topic)."""
    ds = cfg["datasets"][DATASET]
    min_rows = int(cfg["split"]["min_user_interactions"])
    tags, khan_types, type_tags = tags or {}, khan_types or set(), type_tags or {}
    log = CleaningLog(event_cols=["user_id", "time_done"])

    df = log.start(raw)
    log.note("duplicate (user_id, time_done) on the raw file",
             int(raw.duplicated(["user_id", "time_done"]).sum()), len(raw))
    log.note("time_taken <= 0 (set to missing, rows kept)", int((raw["time_taken"] <= 0).sum()), len(raw))
    log.note("correct is null", int(raw["correct"].isna().sum()), len(raw))
    known = set(exercises["name"])
    df = log.apply(df, df[df["exercise"].astype(str).isin(known)], "drop exercises missing from the exercise table")
    df = log.apply(df, df[df["correct"].notna()], "drop rows with missing correct")
    n = df.groupby("user_id", observed=True)["time_done"].transform("size")
    df = log.apply(df, df[n >= min_rows], f"drop learners with < {min_rows} rows")
    eligible = df
    df = log.apply(df, sample_users(df, ds.get("sample_users"), int(cfg["data_seed"])),
                   f"sample {ds.get('sample_users')} learners (data_seed {cfg['data_seed']})")

    topic = exercises.drop_duplicates("name").set_index("name")["topic"].astype(str)
    template_slugs = set(templates["slug"])
    usable_slugs = set(templates.loc[templates["status"] == "usable", "slug"])
    ex = df["exercise"].astype(str)
    ptype = df["problem_type"].astype(str)
    has_template = ex.isin(template_slugs)
    item = ex + ":" + ptype
    split_slugs = {k.split(":", 1)[0] for k in type_tags}
    by_type = item.map(type_tags)
    tag = by_type.where(ex.isin(split_slugs), ex.map(tags))  # a split template never falls back to its union of tags
    out = pd.DataFrame({
        "dataset": DATASET,
        "user_id": df["user_id"].astype(str),
        "item_id": item,
        "source_skill": ex,
        "skill_tag": tag.fillna("junyi:" + ex.map(topic).fillna("unknown")),
        "khan_slug": ex.where(has_template),
        "ts": (df["time_done"] // 1000).astype("int64"),
        "correct": df["correct"].astype("int8"),
        "rt_ms": schema.clean_rt(df["time_taken"] * 1000.0, tuple(cfg["rt_clip_ms"])),
        "attempts": df["count_attempts"].fillna(0).clip(lower=0).astype("int16"),
        "hints": df["count_hints"].fillna(0).clip(lower=0).astype("int16"),
        "section_type": df["review_mode"].map({True: "review", False: "practice"}).fillna("practice"),
    })
    out = schema.coerce(assign_splits(out, cfg))
    schema.validate(out)

    acc = float(out["correct"].mean())
    cov_raw = coverage(raw, template_slugs, usable_slugs, khan_types)
    cov_sample = coverage(df, template_slugs, usable_slugs, khan_types)
    facts = {
        "rows_raw": len(raw), "users_raw": int(raw["user_id"].nunique()),
        "users_ge_min_rows": int(eligible["user_id"].nunique()),
        "accuracy_raw": round(float(raw["correct"].astype("float").mean()), 4),
        "accuracy_eligible": round(float(eligible["correct"].astype("float").mean()), 4),
        "accuracy_sample": round(acc, 4),
        "majority_baseline_sample": round(max(acc, 1 - acc), 4),
        **{f"raw_{k}": v for k, v in cov_raw.items()},
        **{f"sample_{k}": v for k, v in cov_sample.items()},
        "share_rows_with_mapped_tag": round(float((~out["skill_tag"].str.startswith("junyi:")).mean()), 4),
        "share_review_rows": round(float((out["section_type"] == "review").mean()), 4),
    }
    outside = outside_sample_counts(eligible, df)
    facts["outside_sample_learners"] = int(eligible["user_id"].nunique() - df["user_id"].nunique())
    facts["outside_sample_rows"] = int(outside["n"].sum())
    cov_table = pd.DataFrame([{"dataset": DATASET, "metric": k, "value": v} for k, v in facts.items()])
    return AdapterResult(out, {"a2_cleaning_junyi2015": log.table(DATASET), "a2_coverage_junyi2015": cov_table}, facts,
                         {OUTSIDE_COUNTS: outside})


OUTSIDE_COUNTS = "junyi2015_outside_sample_counts"


def outside_sample_counts(eligible: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Responses and correct answers per (exercise, problem_type) of the learners who pass every cleaning step but were
    not drawn into the 20,000-learner sample. They take no part in any split, so their answers can inform item
    difficulty without leaking test rows (P1-06 follow-up: the sample alone leaves 25 templates under 200 responses)."""
    out = eligible[~eligible["user_id"].isin(sample["user_id"].unique())]
    g = out.groupby([out["exercise"].astype(str), out["problem_type"].astype(str)], observed=True)["correct"]
    counts = g.agg(n="size", k="sum").reset_index()
    counts.columns = ["khan_slug", "type_key", "n", "k"]
    counts["k"] = counts["k"].astype("int64")
    return counts


def build(cfg: dict, repo: Path) -> AdapterResult:
    ds = cfg["datasets"][DATASET]
    log_path = repo / ds["path"]
    raw = read_raw(log_path, int(ds.get("chunksize", 2_000_000)))
    exercises = pd.read_csv(log_path.parent / "junyi_Exercise_table.csv", usecols=["name", "topic"])
    templates = pd.read_csv(repo / "lmsrag/data/khan_v2/khan_templates.csv")
    mapping = repo / ds.get("skill_map", DEFAULT_MAPPING)
    types_csv = repo / "lmsrag/data/khan_v2/khan_problem_types.csv"
    tags = khan_tags(mapping, SkillMapper())
    type_tags = {f"{s}:{k}": t[0] for (s, k), t in khan_type_tag_lists(mapping, types_csv).items() if t}
    return clean(raw, exercises, templates, cfg, tags, template_types(types_csv), type_tags)
