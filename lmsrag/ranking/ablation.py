"""Ablation of the hybrid task ranker (card P2-07, review C3), test split scored once.

Same protocol as ``lmsrag.ranking.tasks`` (card P2-06): LTR fitted on the ``val`` phase (history -> validation),
then models refitted on history + validation and scored on ``test``, ALS/BPR over 5 seeds. Every variant is the same
LightGBM LambdaRank, same candidates (union of the base models' top 50), same seed; only its feature set changes:

``full``              the P2-06 features (knn, als, log_pop, p_hat, zpd_gap, log_n_skill, skill_recency, ranks)
``no_zpd``            without zpd_gap and rank_zpd (p_hat kept)
``no_correctness``    without everything computed from answers being right: p_hat, zpd_gap, rank_zpd
``no_cf``             without the collaborative signals: knn, als, rank_knn
``no_popularity``     without log_pop and rank_pop
``no_recency``        without skill_recency
``plus_rt``           full + response time: the learner's mean log RT on the item's skill minus the skill's mean
                      (the P2-06 model has no RT feature, so RT is tested by adding it)
``swap_zpd_skill``    ZPD from the skill-level EMA mastery M (no item difficulty): p_hat, zpd_gap, rank_zpd replaced by
                      M, |M - 0.7| and its rank

Metric: NDCG@10 and Recall@10 per learner (mean over the 5 seeds), paired bootstrap over learners (1,000 resamples,
``data_seed``), each variant against ``full``, Holm over the variants.

    python -m lmsrag.ranking.ablation --dataset assist2009
    python -m lmsrag.ranking.ablation --dataset junyi2015 --sample-users 4000

Outputs ``reports/tables/c3_ablation_ranking_<dataset>.csv`` and the per-learner scores
``<output_dir>/predictions/c3_ablation_ranking_<dataset>.csv``.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from ..datasets.build import REPO, load_config
from ..stats.bootstrap import holm, p_two_sided
from . import tasks as T

EXT = T.FEATURES + ["rt_dev", "m_skill", "zpd_gap_skill", "rank_zpd_skill"]


def _drop(*names):
    return [f for f in T.FEATURES if f not in names]


VARIANTS = {
    "full": list(T.FEATURES),
    "no_zpd": _drop("zpd_gap", "rank_zpd"),
    "no_correctness": _drop("p_hat", "zpd_gap", "rank_zpd"),
    "no_cf": _drop("knn", "als", "rank_knn"),
    "no_popularity": _drop("log_pop", "rank_pop"),
    "no_recency": _drop("skill_recency"),
    "plus_rt": list(T.FEATURES) + ["rt_dev"],
    "swap_zpd_skill": [f for f in T.FEATURES if f not in ("p_hat", "zpd_gap", "rank_zpd")]
    + ["m_skill", "zpd_gap_skill", "rank_zpd_skill"],
}


def rt_matrix(train: pd.DataFrame, users: np.ndarray, skills: pd.Index) -> np.ndarray:
    """Learner x skill: mean log1p(rt) of the learner minus the skill's mean (0 where unknown)."""
    t = train[train["rt_ms"].notna() & (train["rt_ms"] > 0)]
    lr = np.log1p(t["rt_ms"].to_numpy(dtype=float))
    t = t.assign(lr=lr, k=skills.get_indexer(t["source_skill"].astype(str)))
    sk = t.groupby("k")["lr"].mean()
    us = t.groupby(["user_id", "k"])["lr"].mean().reset_index()
    out = np.zeros((len(users), len(skills)))
    ui = pd.Index(users).get_indexer(us["user_id"])
    ok = ui >= 0
    out[ui[ok], us["k"].to_numpy()[ok]] = us["lr"].to_numpy()[ok] - sk.reindex(us["k"]).to_numpy()[ok]
    return out


def ext_features(sc: T.Scorer, rt: np.ndarray, ui: int, base: dict, cand: np.ndarray) -> np.ndarray:
    F = T.features(sc, ui, base, cand)
    k = sc.item_k[cand]
    m = sc.M[ui, k]
    gap = np.abs(m - T.TARGET)
    rank = np.argsort(np.argsort(gap)) / len(cand)
    return np.column_stack([F, rt[ui, k], m, gap, rank])


def collect(sc: T.Scorer, ph: T.Phase, rt: np.ndarray):
    """Per evaluated learner: candidates, extended features, labels (target = in the learner's next split)."""
    out = []
    for u in ph.targets:
        ui = ph.uidx[u]
        seen = np.zeros(ph.n_items, bool)
        seen[ph.X[ui].indices] = True
        base = sc.base_scores(ui)
        cand = np.unique(np.concatenate([T.topn(s, seen, T.TOPN) for s in base.values()]))
        out.append((u, cand, ext_features(sc, rt, ui, base, cand), np.isin(cand, list(ph.targets[u])).astype(int)))
    return out


def per_learner(rows, model, cols, ph: T.Phase) -> pd.DataFrame:
    res = []
    for u, cand, F, _ in rows:
        s = model.predict(F[:, cols])
        top = cand[np.lexsort((cand, -s))][:T.K]
        t = ph.targets[u]
        hits = np.array([i in t for i in top], dtype=float)
        idcg = sum(1 / np.log2(r + 2) for r in range(min(len(t), T.K)))
        res.append({"user_id": u, "ndcg@10": float((hits / np.log2(np.arange(2, len(hits) + 2))).sum() / idcg),
                    "recall@10": hits.sum() / len(t)})
    return pd.DataFrame(res)


def run(df: pd.DataFrame, cfg_mastery: dict, seed: int, seeds=T.SEEDS):
    df = df.copy()
    df["source_skill"] = df["source_skill"].astype(str)
    item_index = pd.Index(np.sort(df["item_id"].unique()))
    item_skill_name = df.drop_duplicates("item_id").set_index("item_id")["source_skill"].reindex(item_index).to_numpy()
    skills = pd.Index(np.sort(df["source_skill"].unique()))
    item_skill = skills.get_indexer(item_skill_name)
    t0 = time.time()

    def make(train_splits, target_split):
        ph = T.Phase(df, train_splits, target_split, item_index, item_skill, None, None)
        ph.item_skill_name = item_skill_name
        return ph

    ph_v = make(("history",), "validation")
    F = T.fit_mf("als", ph_v.Xall, seed)
    sc_v = T.Scorer(df, ph_v, item_index, skills, cfg_mastery, seed, mf={"als": (T.user_factors(F, ph_v.X), F)})
    rows_v = collect(sc_v, ph_v, rt_matrix(ph_v.train, ph_v.users, skills))
    X = np.vstack([r[2] for r in rows_v])
    y = np.concatenate([r[3] for r in rows_v])
    g = np.array([len(r[1]) for r in rows_v])
    models = {}
    for name, feats in VARIANTS.items():
        cols = [EXT.index(f) for f in feats]
        models[name] = (T.fit_ltr((X[:, cols], y, g), seed), cols)
    print(f"val phase: {len(rows_v)} learners, {len(VARIANTS)} models ({time.time() - t0:.0f} s)")

    ph_t = make(("history", "validation"), "test")
    rt = rt_matrix(ph_t.train, ph_t.users, skills)
    per = []
    for s in seeds:
        mf = {}
        for kind in ("als", "bpr"):
            Fm = T.fit_mf(kind, ph_t.Xall, s)
            mf[kind] = (T.user_factors(Fm, ph_t.X), Fm)
        sc = T.Scorer(df, ph_t, item_index, skills, cfg_mastery, s, mf=mf)
        rows_t = collect(sc, ph_t, rt)
        for name, (m, cols) in models.items():
            per.append(per_learner(rows_t, m, cols, ph_t).assign(variant=name, seed=s))
        print(f"test seed {s} done ({time.time() - t0:.0f} s)")
    return pd.concat(per, ignore_index=True)


def compare(per: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rows = []
    for metric in ("ndcg@10", "recall@10"):
        wide = per.groupby(["user_id", "variant"])[metric].mean().unstack().sort_index()
        idx = np.random.default_rng(seed).integers(0, len(wide), size=(n, len(wide)))
        for v in VARIANTS:
            if v == "full":
                continue
            d = (wide[v] - wide["full"]).to_numpy()
            bs = d[idx].mean(axis=1)
            sd = d.std(ddof=1)
            rows.append({"metric": metric, "variant": v, "full": wide["full"].mean(), "value": wide[v].mean(),
                         "diff": d.mean(), "rel_diff": d.mean() / wide["full"].mean(), "ci_low": np.quantile(bs, 0.025),
                         "ci_high": np.quantile(bs, 0.975), "p": p_two_sided(bs),
                         "cohen_dz": d.mean() / sd if sd > 0 else 0.0, "n_learners": len(d)})
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for metric, g in out.groupby("metric"):
        out.loc[g.index, "p_holm"] = holm(g["p"].to_numpy())
    sd = per.groupby(["variant", "seed"])["ndcg@10"].mean().groupby("variant").std()
    out["ndcg_seed_sd"] = out["variant"].map(sd)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--dataset", required=True, choices=["assist2009", "assist2012", "junyi2015"])
    ap.add_argument("--sample-users", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    seed = int(cfg["data_seed"])
    out_root = REPO / cfg["output_dir"]
    tables = REPO / cfg.get("tables_dir", "reports/tables")
    df = pd.read_parquet(out_root / f"interactions_{args.dataset}.parquet")
    if args.sample_users:
        from ..datasets.splits import sample_users
        df = sample_users(df, args.sample_users, seed)
    per = run(df, T.mastery_cfg(tables, args.dataset), seed)
    res = compare(per, args.n_boot, seed)
    res.round(5).to_csv(tables / f"c3_ablation_ranking_{args.dataset}.csv", index=False)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    per.round(5).to_csv(out_root / "predictions" / f"c3_ablation_ranking_{args.dataset}.csv", index=False)
    print(res.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
