"""Final retrieval evaluation on the frozen gold set (card P2-05 / P1-07, review A5 and B3).

Runs once, after the ``gold-v1`` tag. Nothing here is tuned: the systems and the LTR model were fixed before the gold
labels existed (``pool_runs/*.csv`` and ``models/b3_ltr.txt``, card P2-05 trained on Junyi validation labels only).
The gold file is checked against ``gold_v1.sha256`` byte for byte before it is read.

Two views, because ``adaptive_ltr`` was produced after the pool was built and so most of its top 10 is unjudged
(pool bias, Buckley & Voorhees 2004; Sakai 2007):

1. **Runs as submitted** (top 10 of each system in ``pool_runs/``):
   ``ndcg@5`` / ``ndcg@10`` with unjudged = 0 (the usual TREC reading, a lower bound for unpooled systems),
   ``condensed_ndcg@5`` (unjudged chunks removed from the run before cutting at 5, Sakai's condensed list),
   ``p@5`` (label 2 = relevant) and ``judged@5``.
2. **Pool re-ranking** (every chunk fully judged): each system that can be recomputed here ranks the judged pool of
   the query; ``bm25``, ``lsa``, the LTR model and its single signals, and ``random``. ``bge_ce`` / ``adaptive_old``
   need the HuggingFace hub and are left out of this view.

Gains are ``2^label - 1``; the ideal ranking uses every judged chunk of the query. Differences are tested with a paired
bootstrap over queries (1,000 resamples, ``data_seed``), two-sided p, Holm within each view.

    python -m lmsrag.retrieval.gold_eval --config configs/data.yaml

Outputs ``reports/tables/b3_gold_eval.csv`` (mean and 95% CI per system and metric), ``b3_gold_bootstrap.csv`` (pairs)
and the per-query scores ``<output_dir>/predictions/b3_gold_per_query.csv``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.build import REPO, load_config
from ..stats.bootstrap import holm, p_two_sided

GOLD = REPO / "lmsrag" / "data" / "gold" / "gold_v1.csv"
RUN_SYSTEMS = ["bm25", "lsa", "bge_ce", "adaptive_old", "adaptive_ltr"]
RUN_PAIRS = [("adaptive_ltr", "adaptive_old"), ("adaptive_ltr", "bge_ce"), ("adaptive_ltr", "bm25"),
             ("adaptive_old", "bge_ce"), ("bge_ce", "bm25")]
RERANK_PAIRS = [("ltr", "bm25"), ("ltr", "lsa"), ("ltr", "random"), ("ltr", "cf_popularity_only"),
                ("ltr", "zpd_level_only")]


def load_gold(path: Path = GOLD) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} missing: merge the gold-v1 tag first (git merge gold-v1)")
    want = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0].strip().lower()
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"{path.name} SHA-256 {got} != {want}: the gold set is not the frozen gold-v1")
    return pd.read_csv(path, dtype={"query_id": str, "chunk_id": str})


def dcg(labels: np.ndarray, k: int) -> float:
    lab = np.asarray(labels, dtype=float)[:k]
    return float(((2.0 ** lab - 1) / np.log2(np.arange(2, len(lab) + 2))).sum())


def query_metrics(ranked: list[str], judged: dict[str, int]) -> dict[str, float]:
    ideal = sorted(judged.values(), reverse=True)
    i5, i10 = dcg(ideal, 5), dcg(ideal, 10)
    lab = [judged.get(c, 0) for c in ranked]
    cond = [judged[c] for c in ranked if c in judged]
    return {"ndcg@5": dcg(lab, 5) / i5 if i5 else 0.0, "ndcg@10": dcg(lab, 10) / i10 if i10 else 0.0,
            "condensed_ndcg@5": dcg(cond, 5) / i5 if i5 else 0.0,
            "p@5": float(np.mean([v == 2 for v in lab[:5]])) if lab else 0.0,
            "judged@5": float(np.mean([c in judged for c in ranked[:5]])) if ranked else 0.0}


def score_runs(runs: dict[str, pd.DataFrame], gold: pd.DataFrame) -> pd.DataFrame:
    judged = {q: dict(zip(g["chunk_id"], g["label_final"].astype(int))) for q, g in gold.groupby("query_id")}
    rows = []
    for name, run in runs.items():
        by_q = {q: g.sort_values("rank")["chunk_id"].tolist() for q, g in run.groupby("query_id")}
        for q, j in judged.items():
            rows.append({"view": "runs", "system": name, "query_id": q, **query_metrics(by_q.get(q, []), j)})
    return pd.DataFrame(rows)


def rerank_scores(gold: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """Every recomputable system ranks the judged pool of each gold query."""
    import lightgbm as lgb

    from .ltr import Featurizer

    out_root = REPO / cfg["output_dir"]
    chunks = pd.read_json(out_root / "corpus_chunks.jsonl", lines=True, dtype={"chunk_id": str})
    worked = chunks[chunks["content_type"] == "worked_example"].sort_values("chunk_id").reset_index(drop=True)
    junyi = pd.read_parquet(out_root / "interactions_junyi2015.parquet")
    fz = Featurizer(chunks, worked, junyi[junyi["split"] == "history"], seed)
    model = lgb.Booster(model_file=str(out_root / "models" / "b3_ltr.txt"))
    pos = pd.Series(np.arange(len(worked)), index=worked["chunk_id"])
    queries = pd.read_csv(REPO / "lmsrag" / "data" / "gold" / "queries_v1.csv", dtype={"query_id": str})
    queries = queries.set_index("query_id")
    rng = np.random.default_rng(seed)
    scorers = {"ltr": lambda F: model.predict(F), "bm25": lambda F: F[:, 0], "lsa": lambda F: F[:, 1],
               "zpd_level_only": lambda F: -F[:, 2], "zpd_p_only": lambda F: -F[:, 4],
               "cf_popularity_only": lambda F: F[:, 6], "random": lambda F: rng.random(len(F))}
    rows = []
    for q, g in gold.groupby("query_id"):
        g = g.sort_values("chunk_id")
        qq = queries.loc[q]
        idx = pos.loc[g["chunk_id"]].to_numpy()
        F = fz(qq["query"], float(qq["mastery"]), int(qq["target_level"]), idx)
        judged = dict(zip(g["chunk_id"], g["label_final"].astype(int)))
        ids = g["chunk_id"].to_numpy()
        for name, f in scorers.items():
            s = np.asarray(f(F), dtype=float)
            ranked = ids[np.lexsort((ids, -s))].tolist()
            m = query_metrics(ranked, judged)
            rows.append({"view": "pool_rerank", "system": name, "query_id": q, "ndcg@5": m["ndcg@5"],
                         "ndcg@10": m["ndcg@10"], "p@5": m["p@5"]})
    return pd.DataFrame(rows)


def summarize(per_q: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rows = []
    metrics = [c for c in ("ndcg@5", "ndcg@10", "condensed_ndcg@5", "p@5", "judged@5") if c in per_q]
    for (view, system), g in per_q.groupby(["view", "system"], sort=False):
        g = g.sort_values("query_id")
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(g), size=(n, len(g)))
        for m in metrics:
            v = g[m].to_numpy(dtype=float)
            if np.isnan(v).all():
                continue
            bs = v[idx].mean(axis=1)
            rows.append({"view": view, "system": system, "metric": m, "mean": v.mean(),
                         "ci_low": np.quantile(bs, 0.025), "ci_high": np.quantile(bs, 0.975), "n_queries": len(v)})
    return pd.DataFrame(rows)


def paired(per_q: pd.DataFrame, view: str, pairs, metrics, n: int, seed: int) -> pd.DataFrame:
    d = per_q[per_q["view"] == view]
    rows = []
    for m in metrics:
        wide = d.pivot(index="query_id", columns="system", values=m).sort_index()
        idx = np.random.default_rng(seed).integers(0, len(wide), size=(n, len(wide)))
        for a, b in pairs:
            diff = (wide[a] - wide[b]).to_numpy(dtype=float)
            bs = diff[idx].mean(axis=1)
            sd = diff.std(ddof=1)
            rows.append({"view": view, "metric": m, "system": a, "baseline": b, "mean_a": wide[a].mean(),
                         "mean_b": wide[b].mean(), "diff": diff.mean(), "ci_low": np.quantile(bs, 0.025),
                         "ci_high": np.quantile(bs, 0.975), "p": p_two_sided(bs),
                         "cohen_dz": diff.mean() / sd if sd > 0 else 0.0,
                         "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()), "losses": int((diff < 0).sum())})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-rerank", action="store_true", help="skip the pool re-ranking view (needs the Junyi parquet)")
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    seed = int(cfg["data_seed"])
    out_root = REPO / cfg["output_dir"]
    tables = REPO / cfg.get("tables_dir", "reports/tables")
    gold = load_gold()
    runs = {s: pd.read_csv(out_root / "pool_runs" / f"{s}.csv", dtype={"query_id": str, "chunk_id": str})
            for s in RUN_SYSTEMS if (out_root / "pool_runs" / f"{s}.csv").exists()}
    missing = sorted(set(RUN_SYSTEMS) - set(runs))
    if missing:
        print("runs missing (left out):", missing)
    per_q = score_runs(runs, gold)
    pairs_run = [(a, b) for a, b in RUN_PAIRS if a in runs and b in runs]
    tests = [paired(per_q, "runs", pairs_run, ["ndcg@5", "condensed_ndcg@5"], args.n_boot, seed)]
    if not args.no_rerank:
        rr = rerank_scores(gold, cfg, seed)
        per_q = pd.concat([per_q, rr], ignore_index=True)
        tests.append(paired(per_q, "pool_rerank", RERANK_PAIRS, ["ndcg@5"], args.n_boot, seed))
    summ = summarize(per_q, args.n_boot, seed)
    boot = pd.concat(tests, ignore_index=True)
    meta = {"gold": "gold_v1.csv", "sha256": hashlib.sha256(GOLD.read_bytes()).hexdigest(), "queries": int(
        gold["query_id"].nunique()), "pairs": len(gold), "n_boot": args.n_boot, "seed": seed}
    summ.round(4).to_csv(tables / "b3_gold_eval.csv", index=False)
    boot.round(4).to_csv(tables / "b3_gold_bootstrap.csv", index=False)
    (tables / "b3_gold_eval_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    per_q.round(4).to_csv(out_root / "predictions" / "b3_gold_per_query.csv", index=False)
    with pd.option_context("display.width", 200):
        print(summ.pivot_table(index=["view", "system"], columns="metric", values="mean").round(4).to_string())
        print(boot.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
