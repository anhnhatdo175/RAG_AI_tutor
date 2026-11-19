"""Gold set queries (card P1-07, step 1): 36 skills x 5 mastery levels = 180 queries.

Skills: the canonical skill tags with the most ASSISTments 2009 responses among those whose worked examples cover at
least 3 ``difficulty_level`` values (ASSISTments 2012 fills the list if fewer than 36 qualify). ``junyi:`` and ``khan:``
tags are not ASSISTments skills and are left out. A chunk counts for every tag in its ``skill_tags``.

Each query asks for a worked example at a target level fixed by a rule written in ``rubric_v1.md`` on the day the
queries were made and never changed afterwards::

    target_level = clip(round_half_up(1 + 5 * mastery), 1, 5)   ->  0.2: 2, 0.35: 3, 0.5: 4, 0.65: 4, 0.8: 5

Queries are in English because the corpus and the encoders are.

    python -m lmsrag.labels.gold_queries
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from ..datasets.build import REPO, load_config
from ..skill_mapping import CANONICAL, slugify

GOLD = REPO / "lmsrag" / "data" / "gold"
N_SKILLS = 36
MASTERY = (0.2, 0.35, 0.5, 0.65, 0.8)
MIN_LEVELS = 3
TEMPLATE = ("A student with mastery {mastery} on {skill} needs a worked example at difficulty level {level} of 5.")
COLUMNS = ["query_id", "skill_tag", "skill_name", "mastery", "target_level", "query", "skill_rank", "skill_source",
           "rows_assist2009", "rows_assist2012", "worked_docs", "levels_with_docs"]


def target_level(mastery: float) -> int:
    """The fixed rule of rubric v1 (half-up rounding, so 0.5 -> 3.5 -> 4)."""
    return int(min(5, max(1, math.floor(1 + 5 * mastery + 0.5))))


def worked_docs(chunks: pd.DataFrame) -> pd.DataFrame:
    """One row per (worked-example document, canonical tag)."""
    d = chunks[(chunks["content_type"] == "worked_example") & (chunks["chunk_idx"] == 0)]
    d = d[["doc_id", "skill_tags", "difficulty_level"]].explode("skill_tags").rename(columns={"skill_tags": "skill_tag"})
    return d[d["skill_tag"].notna() & ~d["skill_tag"].str.match(r"^(?:junyi|khan):")]


def skill_names(tables: Path) -> dict[str, str]:
    """tag -> the skill name a query shows: the rule's canonical name, else the 2009 (then 2012) skill with most rows."""
    from_logs: dict[str, str] = {}
    for ds in ("assist2012", "assist2009"):  # 2009 last, so it wins
        c = pd.read_csv(tables / f"a1_skill_counts_{ds}.csv").dropna(subset=["skill_tag"])
        top = c.sort_values("rows", ascending=False).drop_duplicates("skill_tag")
        from_logs.update({tag: " ".join(str(s).split()) for tag, s in zip(top["skill_tag"], top["source_skill"])})
    return {**from_logs, **{slugify(name): name for name in CANONICAL}}


def choose_skills(docs: pd.DataFrame, rows: dict[str, pd.Series], n: int = N_SKILLS) -> pd.DataFrame:
    stats = docs.groupby("skill_tag").agg(worked_docs=("doc_id", "nunique"),
                                         levels=("difficulty_level", lambda s: sorted(set(int(x) for x in s))))
    stats = stats[stats["levels"].map(len) >= MIN_LEVELS].copy()
    stats["rows_assist2009"] = stats.index.map(rows["assist2009"]).fillna(0).astype(int)
    stats["rows_assist2012"] = stats.index.map(rows["assist2012"]).fillna(0).astype(int)
    first = stats[stats["rows_assist2009"] > 0].sort_values(["rows_assist2009", "worked_docs"], ascending=False)
    first = first.assign(skill_source="assist2009").head(n)
    fill = stats.drop(first.index).sort_values(["rows_assist2012", "worked_docs"], ascending=False)
    fill = fill[fill["rows_assist2012"] > 0].assign(skill_source="assist2012").head(n - len(first))
    out = pd.concat([first, fill])
    if len(out) < n:
        raise SystemExit(f"only {len(out)} skills have worked examples at >= {MIN_LEVELS} levels")
    out["skill_rank"] = range(1, len(out) + 1)
    return out


def make(cfg: dict, out_dir: Path = GOLD) -> pd.DataFrame:
    path = out_dir / "queries_v1.csv"
    if path.is_file():
        raise SystemExit(f"{path.name} exists; queries are fixed once made (delete it only before the pools are built)")
    out_root = REPO / cfg["output_dir"]
    chunks = pd.read_json(out_root / "corpus_chunks.jsonl", lines=True)
    rows = {ds: pd.read_parquet(out_root / f"interactions_{ds}.parquet", columns=["skill_tag"])["skill_tag"]
            .value_counts() for ds in ("assist2009", "assist2012")}
    skills = choose_skills(worked_docs(chunks), rows)
    names = skill_names(REPO / cfg.get("tables_dir", "reports/tables"))
    records = []
    for tag, s in skills.iterrows():
        for m in MASTERY:
            level = target_level(m)
            records.append({"skill_tag": tag, "skill_name": names.get(tag, tag.replace("-", " ")), "mastery": m,
                            "target_level": level, "skill_rank": s["skill_rank"], "skill_source": s["skill_source"],
                            "rows_assist2009": s["rows_assist2009"], "rows_assist2012": s["rows_assist2012"],
                            "worked_docs": s["worked_docs"], "levels_with_docs": ";".join(map(str, s["levels"]))})
    q = pd.DataFrame(records)
    q["query"] = [TEMPLATE.format(mastery=m, skill=n, level=lv)
                  for m, n, lv in zip(q["mastery"], q["skill_name"], q["target_level"])]
    q.insert(0, "query_id", [f"q{i:03d}" for i in range(1, len(q) + 1)])
    out_dir.mkdir(parents=True, exist_ok=True)
    q[COLUMNS].to_csv(path, index=False, encoding="utf-8")
    return q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    q = make(load_config(REPO / args.config))
    skills = q.drop_duplicates("skill_tag")
    print(f"{len(q)} queries, {len(skills)} skills ({skills['skill_source'].value_counts().to_dict()})")
    print("  target levels:", q.groupby("mastery")["target_level"].first().to_dict())
    print("  fewest worked examples:", skills.nsmallest(3, "worked_docs")[["skill_name", "worked_docs"]].values.tolist())
    print("  ->", (GOLD / "queries_v1.csv").relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
