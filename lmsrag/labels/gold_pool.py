"""Gold set pools and labelling sheets (card P1-07, steps 2-3).

Input: ``queries_v1.csv`` and, from Người 2 (card P2-02), one run per system in ``<output_dir>/pool_runs/<system>.csv``
with columns ``query_id, chunk_id, rank, score`` (rank 1 = best). Every system filters on the query's skill and returns
``worked_example`` chunks only.

Pool of a query (parameters in ``configs/data.yaml``, key ``gold``; randomness from ``data_seed``):

1. top ``top_k`` of every system in ``systems``;
2. ``random_same_skill`` random worked examples of the same skill and ``random_other_skill`` of another skill, as
   checks (they are always kept);
3. merged and deduplicated; system chunks are taken round-robin by rank (rank 1 of each system, then rank 2, ...)
   until the pool holds ``pool_max`` chunks;
4. under ``pool_min``: filled round-robin from ranks ``top_k + 1 .. fill_depth`` of the runs, then with more random
   chunks of the skill;
5. shuffled; nothing on the sheet says which system found a chunk or how hard the corpus says it is.

Outputs: ``lmsrag/data/gold/pool_v1.csv`` (query_id, position, chunk_id), ``pool_provenance.csv`` (which system found
each chunk, at which rank; annotators do not open it), ``reports/tables/a5_gold_pool_levels.csv`` (difficulty_level and
difficulty_tier of every pool, for the Random level that Người 2 reports), and the sheets
``lmsrag/data/labels/a5_gold_pilot_<annotator>.csv`` (``pilot_queries`` random queries) and
``a5_gold_<annotator>.csv`` (the other queries).

    python -m lmsrag.labels.gold_pool
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.build import REPO, load_config
from .difficulty_sheets import display_answer
from .make_sheets import plain_math

GOLD = REPO / "lmsrag" / "data" / "gold"
LABELS = REPO / "lmsrag" / "data" / "labels"
GLOSSARY = LABELS / "a4_skill_glossary.csv"
SHEET_COLUMNS = ["item", "query_id", "skill", "skill_meaning", "mastery", "target_level", "content", "label", "note",
                 "content_tex"]


def load_runs(run_dir: Path, systems: list[str], query_ids: set[str], chunk_ids: set[str]) -> pd.DataFrame:
    frames, errors = [], []
    for s in systems:
        path = run_dir / f"{s}.csv"
        if not path.is_file():
            errors.append(f"missing run {path}")
            continue
        r = pd.read_csv(path, dtype={"query_id": str, "chunk_id": str})
        missing_cols = {"query_id", "chunk_id", "rank"} - set(r.columns)
        if missing_cols:
            errors.append(f"{path.name}: missing columns {sorted(missing_cols)}")
            continue
        unknown_q = set(r["query_id"]) - query_ids
        unknown_c = set(r["chunk_id"]) - chunk_ids
        if unknown_q:
            errors.append(f"{path.name}: {len(unknown_q)} query ids not in queries_v1.csv")
        if unknown_c:
            errors.append(f"{path.name}: {len(unknown_c)} chunk ids are not worked_example chunks of the corpus")
        if r.duplicated(["query_id", "chunk_id"]).any():
            errors.append(f"{path.name}: a chunk appears twice for one query")
        frames.append(r.assign(system=s))
    if errors:
        raise SystemExit("pool runs not usable:\n  " + "\n  ".join(errors))
    return pd.concat(frames, ignore_index=True)


def round_robin(runs_q: pd.DataFrame, systems: list[str], lo: int, hi: int) -> list[tuple[str, str, int]]:
    """(chunk_id, system, rank) for ranks lo..hi, rank by rank across systems in their configured order."""
    out = []
    for rank in range(lo, hi + 1):
        for s in systems:
            hit = runs_q[(runs_q["system"] == s) & (runs_q["rank"] == rank)]
            out += [(c, s, rank) for c in hit["chunk_id"]]
    return out


def build_pool(query: pd.Series, runs_q: pd.DataFrame, same_skill: list[str], other_skill: list[str],
               g: dict, rng: np.random.Generator) -> list[dict]:
    """The chunks of one query's pool, with where each came from, in a random order."""
    systems, k = list(g["systems"]), int(g["top_k"])
    pool: dict[str, dict] = {}

    def add(chunk: str, origin: str, system: str | None = None, rank: int | None = None) -> None:
        entry = pool.setdefault(chunk, {"chunk_id": chunk, "origins": set(), "found_by": {}})
        entry["origins"].add(origin)
        if system is not None:
            entry["found_by"].setdefault(system, rank)

    def draw(candidates: list[str], n: int, origin: str) -> None:
        free = [c for c in candidates if c not in pool]
        for i in rng.permutation(len(free))[:n]:
            add(free[i], origin)

    draw(same_skill, int(g["random_same_skill"]), "random_same_skill")
    draw(other_skill, int(g["random_other_skill"]), "random_other_skill")
    for chunk, s, rank in round_robin(runs_q, systems, 1, k):
        if chunk in pool or len(pool) < int(g["pool_max"]):
            add(chunk, "system", s, rank)
    for chunk, s, rank in round_robin(runs_q, systems, k + 1, int(g["fill_depth"])):
        if len(pool) >= int(g["pool_min"]):
            break
        add(chunk, "fill", s, rank)
    if len(pool) < int(g["pool_min"]):
        draw(same_skill, int(g["pool_min"]) - len(pool), "random_fill")
    entries = list(pool.values())
    return [entries[i] for i in rng.permutation(len(entries))]


def make_pools(queries: pd.DataFrame, runs: pd.DataFrame, chunks: pd.DataFrame, g: dict,
               seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = chunks["chunk_id"].tolist()
    has = {tag: chunks.loc[chunks["skill_tags"].map(lambda t, tag=tag: tag in t), "chunk_id"].tolist()
           for tag in queries["skill_tag"].unique()}
    pool_rows, prov_rows = [], []
    for _, q in queries.iterrows():
        same = has[q["skill_tag"]]
        same_set = set(same)
        other = [c for c in ids if c not in same_set]
        entries = build_pool(q, runs[runs["query_id"] == q["query_id"]], same, other, g, rng)
        for pos, e in enumerate(entries, 1):
            pool_rows.append({"query_id": q["query_id"], "position": pos, "chunk_id": e["chunk_id"]})
            prov_rows.append({"query_id": q["query_id"], "chunk_id": e["chunk_id"],
                              "origins": ";".join(sorted(e["origins"])),
                              **{f"rank_{s}": e["found_by"].get(s) for s in g["systems"]}})
    prov = pd.DataFrame(prov_rows)
    for s in g["systems"]:
        prov[f"rank_{s}"] = prov[f"rank_{s}"].astype("Int64")
    return pd.DataFrame(pool_rows), prov


def pool_levels(pool: pd.DataFrame, chunks: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """Per query: pool size and counts of each difficulty_level and difficulty_tier, next to the target level."""
    p = pool.merge(chunks[["chunk_id", "difficulty_level", "difficulty_tier"]], on="chunk_id")
    lv = p.pivot_table(index="query_id", columns="difficulty_level", values="chunk_id", aggfunc="count", fill_value=0)
    lv.columns = [f"level_{c}" for c in lv.columns]
    tr = p.pivot_table(index="query_id", columns="difficulty_tier", values="chunk_id", aggfunc="count", fill_value=0)
    tr.columns = [f"tier_{c}" for c in tr.columns]
    out = queries.set_index("query_id")[["skill_tag", "target_level"]].join(p.groupby("query_id").size().rename("pool_size"))
    return out.join(lv).join(tr).fillna(0).reset_index()


def chunk_content(text: str, answer: str, answer_type: str) -> str:
    """Readable chunk text; a radical answer gets its root sign back (see difficulty_sheets.display_answer)."""
    shown = display_answer(str(answer), answer_type)
    if shown != str(answer) and f"Answer: {answer}" in text:
        text = text.replace(f"Answer: {answer}", f"Answer: {shown}")
    return plain_math(text)


def make_sheets(pool: pd.DataFrame, chunks: pd.DataFrame, queries: pd.DataFrame, g: dict, seed: int,
                out_dir: Path = LABELS, gold_dir: Path = GOLD) -> dict[str, pd.DataFrame]:
    glossary = pd.read_csv(GLOSSARY, encoding="utf-8").set_index("skill")["meaning_vi"]
    missing = sorted(set(queries["skill_name"]) - set(glossary.index))
    if missing:
        raise SystemExit(f"{GLOSSARY.name} has no meaning for: {missing}")
    rng = np.random.default_rng(seed)
    order = queries["query_id"].to_numpy()[rng.permutation(len(queries))]  # queries in random order
    pilot = set(order[: int(g["pilot_queries"])])
    c = chunks.set_index("chunk_id")
    q = queries.set_index("query_id")
    rows = []
    for qid in order:
        for _, r in pool[pool["query_id"] == qid].sort_values("position").iterrows():
            ch = c.loc[r["chunk_id"]]
            rows.append({"query_id": qid, "chunk_id": r["chunk_id"], "skill": q.at[qid, "skill_name"],
                         "skill_meaning": glossary[q.at[qid, "skill_name"]], "mastery": q.at[qid, "mastery"],
                         "target_level": q.at[qid, "target_level"],
                         "content": chunk_content(ch["text"], ch["answer"], ch["answer_type"]),
                         "content_tex": ch["text"], "label": "", "note": "", "pilot": qid in pilot})
    items = pd.DataFrame(rows)
    items.insert(0, "item", [f"g{i:04d}" for i in range(1, len(items) + 1)])
    sheets = {"pilot": items[items["pilot"]], "full": items[~items["pilot"]]}
    for a in g["annotators"]:
        for part, frame in sheets.items():
            name = f"a5_gold_pilot_{a}.csv" if part == "pilot" else f"a5_gold_{a}.csv"
            path = out_dir / name
            if path.is_file() and pd.read_csv(path, encoding="utf-8-sig", dtype=str)["label"].notna().any():
                raise SystemExit(f"{name} already has labels; not overwritten")
            frame[SHEET_COLUMNS].to_csv(path, index=False, encoding="utf-8-sig")
    items[["item", "query_id", "chunk_id", "pilot"]].to_csv(gold_dir / "items_v1.csv", index=False)
    return sheets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_config(REPO / args.config)
    g, seed = cfg["gold"], int(cfg["data_seed"])
    out_root = REPO / cfg["output_dir"]
    queries = pd.read_csv(GOLD / "queries_v1.csv")
    chunks = pd.read_json(out_root / "corpus_chunks.jsonl", lines=True)
    chunks = chunks[chunks["content_type"] == "worked_example"].reset_index(drop=True)
    runs = load_runs(out_root / "pool_runs", list(g["systems"]), set(queries["query_id"]), set(chunks["chunk_id"]))
    pool, prov = make_pools(queries, runs, chunks, g, seed)
    pool.to_csv(GOLD / "pool_v1.csv", index=False)
    prov.to_csv(GOLD / "pool_provenance.csv", index=False)
    levels = pool_levels(pool, chunks, queries)
    levels.to_csv(REPO / cfg.get("tables_dir", "reports/tables") / "a5_gold_pool_levels.csv", index=False)
    sheets = make_sheets(pool, chunks, queries, g, seed)
    sizes = pool.groupby("query_id").size()
    print(f"{len(pool)} pairs over {len(sizes)} queries; pool size {sizes.min()}-{sizes.max()} (mean {sizes.mean():.1f})")
    print(f"  pilot: {len(sheets['pilot'])} pairs, {sheets['pilot']['query_id'].nunique()} queries; "
          f"full: {len(sheets['full'])} pairs")
    print("  origins:", prov["origins"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
