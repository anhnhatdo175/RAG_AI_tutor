"""Learning-to-rank retrieval with separate signals (card P2-05, review B3), trained on logs only.

Training data never touches the gold set (gold is test only, review A5). It comes from Junyi, the only log whose
answers join corpus content (Khan template = ``khan_slug``):

* one training query per (learner, canonical skill) practised in ``history``; the learner's mastery M is the EMA of
  its history answers on the skill (alpha of P2-03), and the target level follows the gold rule
  ``target_level(M)`` of ``gold_queries.py``, so a training query looks like a gold query;
* candidates = Khan worked examples of the skill (only those can be labelled from the log);
* label of a chunk = what the learner did on its template in ``validation``: 2 = answered it right, 1 = tried it
  but never right, 0 = did not meet it.

**No leakage** (review B3, "tính vòng tròn"): every feature uses ``history`` rows only (M, popularity) or the corpus;
labels use ``validation`` rows only; the difficulty of a chunk enters as a feature, never into the label.

Signals, kept apart so their contribution can be reported:

* semantic: ``bm25``, ``lsa`` (P2-02 implementations; BGE / cross-encoder need the HuggingFace hub and can be added on
  a machine that has it);
* ZPD: ``level_gap`` = |difficulty_level - target_level|, signed gap, ``zpd_gap`` = |p_hat - 0.7| with
  p_hat = sigmoid(logit M - logit(1 - e)), e = the chunk's ``p_correct_hist`` (else 1 - difficulty);
* CF: log1p of history answers on the chunk's template by all learners (population signal; a gold query has no
  learner, so per-learner CF is not used).

Model: LightGBM LambdaRank, picked on held-out learners (20%, ``data_seed``) by NDCG@5; ablations drop one signal at a
time and single-signal rankers are reported on the same held-out learners. Outputs ``reports/tables/b3_ltr_validation.csv``
(held-out learners, log labels: **not** the gold evaluation), ``b3_ltr_importance.csv``, the model in
``<output_dir>/models/b3_ltr.txt`` and, for the gold queries, the run ``<output_dir>/pool_runs/adaptive_ltr.csv``
(top 10, same format as P2-02) to be scored once after the ``gold-v1`` tag.

    python -m lmsrag.retrieval.ltr --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from ..datasets.build import REPO, load_config
from ..embedder import LsaEmbedder
from ..labels.gold_queries import TEMPLATE, target_level
from ..layer3_retrieval.query import QueryRewriter
from . import pool_runs as P

GROUPS = {"semantic": ["bm25", "lsa"], "zpd": ["level_gap", "signed_gap", "zpd_gap", "difficulty"],
          "cf": ["log_pop"]}
FEATURES = [f for g in GROUPS.values() for f in g]


def skill_name(tag: str) -> str:
    return tag.replace("-", " ")


def ema_mastery(rows: pd.DataFrame, alpha: float, prior: float) -> float:
    m = prior
    for y in rows["correct"].to_numpy(dtype=float):
        m = alpha * y + (1 - alpha) * m
    return m


class Featurizer:
    def __init__(self, chunks: pd.DataFrame, worked: pd.DataFrame, history: pd.DataFrame, seed: int):
        self.w = worked.reset_index(drop=True)
        stop = P.english_stop_words()
        self.stop = stop
        self.bm = P.BM25([P.tokenize(t, stop) for t in self.w["text"]])
        self.lsa = LsaEmbedder(dim=256, seed=seed).fit(chunks["text"].tolist())
        self.v = self.lsa.encode(self.w["text"].tolist())
        pop = history["khan_slug"].dropna().value_counts()
        self.log_pop = np.log1p(self.w["khan_slug"].map(pop).fillna(0).to_numpy(dtype=float))
        e = pd.to_numeric(self.w.get("p_correct_hist"), errors="coerce").to_numpy(dtype=float)
        self.e = np.clip(np.where(np.isnan(e), 1 - self.w["difficulty"].to_numpy(dtype=float), e), 0.02, 0.98)
        self.level = self.w["difficulty_level"].to_numpy(dtype=float)
        self.diff = self.w["difficulty"].to_numpy(dtype=float)
        self.rw = QueryRewriter()

    def __call__(self, text: str, mastery: float, level: int, idx: np.ndarray) -> np.ndarray:
        q = self.rw.rewrite(text)
        bm = self.bm.scores(P.tokenize(q, self.stop), idx)
        ls = self.v[idx] @ self.lsa.encode_queries([q])[0]
        m = np.clip(mastery, 0.02, 0.98)
        p = 1 / (1 + np.exp(-(np.log(m / (1 - m)) - np.log((1 - self.e[idx]) / self.e[idx]))))
        return np.column_stack([bm, ls, np.abs(self.level[idx] - level), self.level[idx] - level, np.abs(p - 0.7),
                                self.diff[idx], self.log_pop[idx]])


def training_queries(junyi: pd.DataFrame, worked: pd.DataFrame, alpha: float, max_learners: int, seed: int):
    j = junyi[junyi["skill_tag"].notna() & ~junyi["skill_tag"].astype(str).str.startswith("junyi:")]
    tags = worked["skill_tags"].map(set)
    khan = worked["source"].eq("khan").to_numpy()
    cand = {}
    users = np.sort(j["user_id"].unique())
    users = np.random.default_rng(seed).choice(users, size=min(max_learners, len(users)), replace=False)
    hist = j[(j["split"] == "history") & j["user_id"].isin(users)].sort_values(["user_id", "ts"], kind="mergesort")
    val = j[(j["split"] == "validation") & j["user_id"].isin(users)]
    vlab = val.groupby(["user_id", "khan_slug"])["correct"].max()  # 1 if ever right
    prior = j[j["split"] == "history"].groupby("skill_tag")["correct"].mean()
    out = []
    for (u, k), g in hist.groupby(["user_id", "skill_tag"], sort=False):
        k = str(k)
        if k not in cand:
            cand[k] = np.flatnonzero(tags.map(lambda s, k=k: k in s).to_numpy() & khan)
        idx = cand[k]
        if len(idx) < 5:
            continue
        M = ema_mastery(g, alpha, float(prior.get(k, 0.5)))
        slugs = worked["khan_slug"].to_numpy()[idx]
        lab = np.array([2 if vlab.get((u, s), -1) == 1 else (1 if vlab.get((u, s), -1) == 0 else 0) for s in slugs])
        if lab.max() == 0:
            continue
        out.append({"user_id": u, "skill_tag": k, "mastery": round(M, 4), "level": target_level(M), "idx": idx,
                    "label": lab})
    return out


def ndcg_at(scores, labels, k=5):
    o = np.lexsort((np.arange(len(scores)), -scores))[:k]
    g = (2.0 ** labels[o] - 1) / np.log2(np.arange(2, len(o) + 2))
    ideal = np.sort(labels)[::-1][:k]
    ig = ((2.0 ** ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))).sum()
    return g.sum() / ig if ig > 0 else 0.0


def evaluate(qs, X, scorer) -> float:
    return float(np.mean([ndcg_at(scorer(X[i]), q["label"]) for i, q in enumerate(qs)]))


def fit(X_tr, y_tr, g_tr, X_va, y_va, g_va, seed, cols, leaves):
    import lightgbm as lgb
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.05, num_leaves=leaves,
                       min_child_samples=50, random_state=seed, verbose=-1)
    m.fit(X_tr[:, cols], y_tr, group=g_tr, eval_set=[(X_va[:, cols], y_va)], eval_group=[g_va], eval_at=[5],
          callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--max-learners", type=int, default=5000)
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    seed = int(cfg["data_seed"])
    out_root = REPO / cfg["output_dir"]
    tables = REPO / cfg.get("tables_dir", "reports/tables")
    t0 = time.time()
    chunks = pd.read_json(out_root / "corpus_chunks.jsonl", lines=True, dtype={"chunk_id": str})
    worked = chunks[chunks["content_type"] == "worked_example"].sort_values("chunk_id").reset_index(drop=True)
    junyi = pd.read_parquet(out_root / "interactions_junyi2015.parquet")
    alpha = json.loads((tables / "b1_variants_choices_junyi2015.json").read_text(encoding="utf-8"))["ema"]["alpha"]
    fz = Featurizer(chunks, worked, junyi[junyi["split"] == "history"], seed)
    qs = training_queries(junyi, worked, alpha, args.max_learners, seed)
    X = [fz(TEMPLATE.format(mastery=q["mastery"], skill=skill_name(q["skill_tag"]), level=q["level"]), q["mastery"],
            q["level"], q["idx"]) for q in qs]
    users = np.array([q["user_id"] for q in qs])
    uu = np.unique(users)
    held = set(np.random.default_rng(seed).choice(uu, size=len(uu) // 5, replace=False))
    va = np.array([u in held for u in users])
    stack = lambda sel: (np.vstack([X[i] for i in np.flatnonzero(sel)]),  # noqa: E731
                         np.concatenate([qs[i]["label"] for i in np.flatnonzero(sel)]),
                         np.array([len(qs[i]["label"]) for i in np.flatnonzero(sel)]))
    X_tr, y_tr, g_tr = stack(~va)
    X_va, y_va, g_va = stack(va)
    qs_va = [qs[i] for i in np.flatnonzero(va)]
    Xs_va = [X[i] for i in np.flatnonzero(va)]
    print(f"{len(qs)} training queries ({va.sum()} held out), {len(X_tr) + len(X_va)} pairs - {time.time() - t0:.0f} s")
    rows = []
    rng = np.random.default_rng(seed)
    rows.append({"model": "random", "ndcg@5": evaluate(qs_va, Xs_va, lambda F: rng.random(len(F)))})
    for name, col, sign in (("bm25", 0, 1), ("lsa", 1, 1), ("zpd_level_only", 2, -1), ("zpd_p_only", 4, -1),
                            ("cf_popularity_only", 6, 1)):
        rows.append({"model": name, "ndcg@5": evaluate(qs_va, Xs_va, lambda F, c=col, s=sign: s * F[:, c])})
    all_cols = list(range(len(FEATURES)))
    best = None
    for leaves in (15, 31, 63):
        m = fit(X_tr, y_tr, g_tr, X_va, y_va, g_va, seed, all_cols, leaves)
        sc = evaluate(qs_va, Xs_va, lambda F, m=m: m.predict(F))
        if best is None or sc > best[0]:
            best = (sc, leaves, m)
    rows.append({"model": "ltr_all", "ndcg@5": best[0], "num_leaves": best[1]})
    for gname, feats in GROUPS.items():
        cols = [i for i, f in enumerate(FEATURES) if f not in feats]
        m = fit(X_tr, y_tr, g_tr, X_va, y_va, g_va, seed, cols, best[1])
        rows.append({"model": f"ltr_without_{gname}",
                     "ndcg@5": evaluate(qs_va, Xs_va, lambda F, m=m, c=cols: m.predict(F[:, c])), "num_leaves": best[1]})
    res = pd.DataFrame(rows)
    res["label"] = "held-out learners, log labels (validation split); not the gold evaluation"
    res.round(4).to_csv(tables / "b3_ltr_validation.csv", index=False)
    model = best[2]
    imp = pd.DataFrame({"feature": FEATURES, "gain": model.booster_.feature_importance("gain")})
    imp["share"] = (imp["gain"] / imp["gain"].sum()).round(4)
    imp.to_csv(tables / "b3_ltr_importance.csv", index=False)
    (out_root / "models").mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out_root / "models" / "b3_ltr.txt"))
    # run for the gold queries (scored only after gold-v1)
    queries = pd.read_csv(REPO / "lmsrag" / "data" / "gold" / "queries_v1.csv", dtype={"query_id": str})
    cand = P.candidates(queries, worked)
    runs = []
    for q in queries.itertuples():
        idx = cand[q.query_id]
        s = model.predict(fz(q.query, float(q.mastery), int(q.target_level), idx))
        runs.append(P.top(q.query_id, worked["chunk_id"].to_numpy()[idx], s, int(cfg["gold"]["fill_depth"])))
    run = pd.concat(runs, ignore_index=True)
    P.check_run(run, queries, worked, cand, int(cfg["gold"]["fill_depth"]))
    run.to_csv(out_root / "pool_runs" / "adaptive_ltr.csv", index=False)
    print(res.round(4).to_string(index=False))
    print(imp.to_string(index=False))
    print(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
