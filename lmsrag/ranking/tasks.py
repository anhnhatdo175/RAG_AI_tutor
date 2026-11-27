"""Task ranking on the interaction logs (card P2-06, review B4 and C1).

Question: which tasks (items) should a learner get next? Two phases, both on the common split:

* ``val``: models are fitted on ``history``; the targets of a learner are the items of its ``validation`` rows it had
  not met in history. LTR weights, the diversification strength and nothing else are chosen here.
* ``test``: models are refitted on ``history + validation``; targets = ``test`` items not met before. Scored once.

Candidates = every catalog item the learner has not met (an item's skill is public metadata, so items never seen in
training are candidates too). Metrics at K = 10 per learner, averaged: Recall, NDCG, MRR (target = "the learner
did this task next"), ``zpd_gap`` = mean |p_hat - 0.7| over the top K (how well the list fits the learner's
ZPD), ``skill_cover`` = distinct skills in the top K, ``new_item_recall`` = recall restricted to targets that were
never answered in training (new tasks).

Models (review C1, B4): ``random``, ``most_popular``, ``itemknn`` (cosine item-item on the binary learner x item
matrix, 100 neighbours), ``als`` and ``bpr`` (implicit, 64 factors, 5 seeds: mean and sd), ``cb_zpd`` (-|p_hat - 0.7|,
p_hat = item-aware calibrated EMA of P2-03 at the cut-off), ``hybrid_ltr`` (LightGBM LambdaRank on the union of the
base models' top 50 with features: knn, als, popularity, p_hat, |p_hat - 0.7|, practice count and recency of the
item's skill, base ranks), ``hybrid_ltr_div`` (same, re-ranked with a penalty per repeated skill in the list).
LightGCN and "pure semantic" are not run here: LightGCN needs a GPU run (Kaggle) and ASSISTments items have no text.

Cold start (``--cold``): 20% of the test learners (drawn with ``data_seed``) keep only their first k history events,
k in {0, 1, 2, 4}; everything else is unchanged and the same LTR model is used.

    python -m lmsrag.ranking.tasks --dataset assist2009 [--cold]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ..datasets.build import REPO, load_config

K = 10
TOPN = 50
TARGET = 0.7
SEEDS = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------- data
class Phase:
    """Training rows, targets and per-learner state for one phase."""

    def __init__(self, df: pd.DataFrame, train_splits: tuple[str, ...], target_split: str, item_index: pd.Index,
                 item_skill: np.ndarray, users: np.ndarray | None = None, cold: dict | None = None):
        tr = df[df["split"].isin(train_splits)]
        if cold:  # cold-start learners keep their first k history rows only
            cu, k = cold["users"], cold["k"]
            keep = ~tr["user_id"].isin(cu) | (tr.groupby("user_id").cumcount() < k)
            tr = tr[keep]
        self.train = tr
        tg = df[df["split"] == target_split]
        self.n_items = len(item_index)
        self.item_skill = item_skill
        self.users = np.sort(tg["user_id"].unique()) if users is None else users
        self.uidx = {u: i for i, u in enumerate(self.users)}
        ti = item_index.get_indexer(tr["item_id"])
        tu = tr["user_id"].map(self.uidx)
        m = tu.notna().to_numpy()
        # learner x item matrix for the evaluated learners (binary)
        self.X = sparse.csr_matrix((np.ones(m.sum()), (tu[m].astype(int), ti[m])), shape=(len(self.users), self.n_items))
        self.X.data[:] = 1.0
        # full training matrix (all learners) for fitting CF
        allu, inv = np.unique(tr["user_id"], return_inverse=True)
        self.Xall = sparse.csr_matrix((np.ones(len(tr)), (inv, ti)), shape=(len(allu), self.n_items))
        self.Xall.data[:] = 1.0
        self.pop = np.asarray(self.Xall.sum(0)).ravel()
        seen = set(zip(tr["user_id"], tr["item_id"]))
        tg = tg[tg["user_id"].isin(self.uidx)]
        tgt = tg[[(u, i) not in seen for u, i in zip(tg["user_id"], tg["item_id"])]]
        self.targets = {u: set(item_index.get_indexer(g["item_id"].unique())) for u, g in tgt.groupby("user_id")}
        self.new_items = set(np.flatnonzero(self.pop == 0))


def mastery_state(train: pd.DataFrame, users: np.ndarray, skills: pd.Index, alpha: float, prior: pd.Series):
    """EMA mastery M, answer count n and last position of each (learner, skill) at the end of ``train``."""
    tr = train.sort_values(["user_id", "ts"], kind="mergesort")
    M = np.tile(prior.reindex(skills).fillna(prior.mean()).to_numpy(), (len(users), 1))
    n = np.zeros_like(M)
    last = np.full_like(M, np.inf)
    uidx = {u: i for i, u in enumerate(users)}
    for u, g in tr[tr["user_id"].isin(uidx)].groupby("user_id", sort=False):
        i = uidx[u]
        ks = skills.get_indexer(g["source_skill"].astype(str))
        L = len(g)
        for pos, (k, y) in enumerate(zip(ks, g["correct"].to_numpy(dtype=float))):
            M[i, k] = alpha * y + (1 - alpha) * M[i, k]
            n[i, k] += 1
            last[i, k] = L - pos  # 1 = the learner's most recent answer
    return M, n, last


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- models
def itemknn_sim(X: sparse.csr_matrix, nn: int = 100) -> sparse.csr_matrix:
    Xn = X.multiply(1 / np.sqrt(np.maximum(np.asarray(X.sum(0)).ravel(), 1))).tocsc()
    S = (Xn.T @ Xn).tocsr()
    S.setdiag(0)
    S.eliminate_zeros()
    rows, cols, vals = [], [], []
    for i in range(S.shape[0]):
        a, b = S.indptr[i], S.indptr[i + 1]
        if b - a > nn:
            keep = np.argpartition(-S.data[a:b], nn)[:nn] + a
        else:
            keep = np.arange(a, b)
        rows += [i] * len(keep); cols += S.indices[keep].tolist(); vals += S.data[keep].tolist()
    return sparse.csr_matrix((vals, (rows, cols)), shape=S.shape)


def fit_mf(kind: str, Xall: sparse.csr_matrix, seed: int):
    import os
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking
    if kind == "als":
        m = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=15, random_state=seed)
    else:
        m = BayesianPersonalizedRanking(factors=64, learning_rate=0.05, regularization=0.01, iterations=60,
                                        random_state=seed)
    m.fit((Xall * 20.0).tocsr() if kind == "als" else Xall, show_progress=False)
    return np.asarray(m.item_factors)


def user_factors(item_f: np.ndarray, X: sparse.csr_matrix, reg: float = 0.05) -> np.ndarray:
    """Least-squares user vectors from the items each evaluated learner met (fold-in; same for every phase)."""
    F = item_f
    A = F.T @ F + reg * np.eye(F.shape[1])
    return np.asarray(X @ F) @ np.linalg.inv(A)


# ---------------------------------------------------------------- evaluation
def metrics_for(ranked: dict, ph: Phase, phat_fn, k: int = K) -> dict[str, float]:
    rec, ndcg, mrr, gap, cover, newrec = [], [], [], [], [], []
    for u, items in ranked.items():
        t = ph.targets.get(u)
        if not t:
            continue
        top = list(items[:k])
        hits = [1.0 if i in t else 0.0 for i in top]
        rec.append(sum(hits) / len(t))
        dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
        idcg = sum(1 / np.log2(r + 2) for r in range(min(len(t), k)))
        ndcg.append(dcg / idcg)
        mrr.append(next((1 / (r + 1) for r, h in enumerate(hits) if h), 0.0))
        gap.append(float(np.mean(np.abs(phat_fn(u, np.array(top)) - TARGET))))
        cover.append(len(set(ph.item_skill[top])))
        tn = t & ph.new_items
        if tn:
            newrec.append(len(tn & set(top)) / len(tn))
    return {"recall@10": np.mean(rec), "ndcg@10": np.mean(ndcg), "mrr@10": np.mean(mrr), "zpd_gap@10": np.mean(gap),
            "skill_cover@10": np.mean(cover), "new_item_recall@10": np.mean(newrec) if newrec else np.nan,
            "n_learners": len(rec)}


def topn(scores: np.ndarray, seen_row: np.ndarray, n: int) -> np.ndarray:
    s = scores.copy()
    s[seen_row] = -np.inf
    idx = np.argpartition(-s, n)[:n]
    return idx[np.lexsort((idx, -s[idx]))]


class Scorer:
    """Scores of every base model for a batch of learners, plus LTR features."""

    def __init__(self, df_all: pd.DataFrame, ph: Phase, item_index: pd.Index, skills: pd.Index, cfg: dict, seed: int,
                 mf: dict | None = None):
        self.ph = ph
        self.S = itemknn_sim(ph.Xall)
        self.logpop = np.log1p(ph.pop)
        tr = ph.train
        mu = tr["correct"].mean()
        st = tr.groupby("source_skill")["correct"].agg(["sum", "size"])
        prior = (st["sum"] + 10 * mu) / (st["size"] + 10)
        prior.index = prior.index.astype(str)
        sk = pd.Series(ph.item_skill_name).map(prior).fillna(mu).to_numpy()
        it = tr.groupby("item_id")["correct"].agg(["sum", "size"]).reindex(item_index).fillna(0)
        self.e_item = np.clip((it["sum"].to_numpy() + 10 * sk) / (it["size"].to_numpy() + 10), 0.02, 0.98)
        self.item_k = skills.get_indexer(ph.item_skill_name)
        self.M, self.n, self.last = mastery_state(tr, ph.users, skills, cfg["alpha"], prior)
        self.coef = cfg["coef"]
        self.mf = mf or {}

    def phat(self, ui: int, items: np.ndarray) -> np.ndarray:
        k = self.item_k[items]
        m, n = self.M[ui, k], self.n[ui, k]
        se = np.sqrt(m * (1 - m) / (n + 1))
        c = self.coef
        z = c[0] + c[1] * logit(m) + c[2] * logit(self.e_item[items]) + c[3] * np.log1p(n) + c[4] * se
        return 1 / (1 + np.exp(-z))

    def base_scores(self, ui: int) -> dict[str, np.ndarray]:
        x = self.ph.X[ui]
        all_items = np.arange(self.ph.n_items)
        out = {"most_popular": self.logpop, "itemknn": np.asarray((x @ self.S).todense()).ravel(),
               "cb_zpd": -np.abs(self.phat(ui, all_items) - TARGET)}
        for name, (U, F) in self.mf.items():
            out[name] = F @ U[ui]
        return out


def features(sc: Scorer, ui: int, base: dict, cand: np.ndarray) -> np.ndarray:
    p = sc.phat(ui, cand)
    k = sc.item_k[cand]
    cols = [base["itemknn"][cand], base.get("als", np.zeros(sc.ph.n_items))[cand], sc.logpop[cand], p,
            np.abs(p - TARGET), np.log1p(sc.n[ui, k]), np.where(np.isfinite(sc.last[ui, k]),
                                                                1 / sc.last[ui, k], 0.0)]
    for name in ("itemknn", "most_popular", "cb_zpd"):
        s = base[name][cand]
        cols.append(np.argsort(np.argsort(-s)) / len(cand))
    return np.column_stack(cols)


FEATURES = ["knn", "als", "log_pop", "p_hat", "zpd_gap", "log_n_skill", "skill_recency", "rank_knn", "rank_pop",
            "rank_zpd"]


def run_phase(sc: Scorer, ph: Phase, rng: np.random.Generator, ltr=None, div_lambda: float = 0.0,
              collect: bool = False):
    ranked: dict[str, dict] = {m: {} for m in ("random", "most_popular", "itemknn", "cb_zpd", *sc.mf)}
    if ltr is not None:
        ranked["hybrid_ltr"], ranked["hybrid_ltr_div"] = {}, {}
    X_rows, y_rows, groups = [], [], []
    for u in ph.targets:
        ui = ph.uidx[u]
        seen = np.zeros(ph.n_items, bool)
        seen[ph.X[ui].indices] = True
        base = sc.base_scores(ui)
        free = np.flatnonzero(~seen)
        ranked["random"][u] = rng.choice(free, size=min(K, len(free)), replace=False)
        tops = {m: topn(s, seen, TOPN) for m, s in base.items()}
        for m in base:
            ranked[m][u] = tops[m][:K]
        if ltr is None and not collect:
            continue
        cand = np.unique(np.concatenate(list(tops.values())))
        F = features(sc, ui, base, cand)
        if collect:
            X_rows.append(F); y_rows.append(np.isin(cand, list(ph.targets[u])).astype(int)); groups.append(len(cand))
        if ltr is not None:
            s = ltr.predict(F)
            order = cand[np.lexsort((cand, -s))]
            ranked["hybrid_ltr"][u] = order[:K]
            ranked["hybrid_ltr_div"][u] = diversify(order, s[np.searchsorted(cand, order)], sc.ph.item_skill, div_lambda)
    if collect:
        return ranked, (np.vstack(X_rows), np.concatenate(y_rows), np.array(groups))
    return ranked, None


def diversify(order: np.ndarray, scores: np.ndarray, item_skill: np.ndarray, lam: float, k: int = K) -> np.ndarray:
    """Greedy re-rank: score minus lam * (items of the same skill already in the list), scores min-max scaled."""
    if lam <= 0:
        return order[:k]
    s = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-12)
    chosen, count = [], {}
    pool = list(range(len(order)))
    while pool and len(chosen) < k:
        best = max(pool, key=lambda j: (s[j] - lam * count.get(item_skill[order[j]], 0), -order[j]))
        chosen.append(order[best]); count[item_skill[order[best]]] = count.get(item_skill[order[best]], 0) + 1
        pool.remove(best)
    return np.array(chosen)


def fit_ltr(data, seed: int):
    import lightgbm as lgb
    X, y, g = data
    starts = np.r_[0, np.cumsum(g)]
    rng = np.random.default_rng(seed)
    val_g = rng.random(len(g)) < 0.2
    idx_tr = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in np.flatnonzero(~val_g)])
    idx_va = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in np.flatnonzero(val_g)])
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=31,
                       min_child_samples=50, random_state=seed, verbose=-1)
    m.fit(X[idx_tr], y[idx_tr], group=g[~val_g], eval_set=[(X[idx_va], y[idx_va])], eval_group=[g[val_g]],
          eval_at=[K], callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


def build(df: pd.DataFrame, cfg_mastery: dict, seed: int, cold: bool = False):
    df = df.copy()
    df["source_skill"] = df["source_skill"].astype(str)
    item_index = pd.Index(np.sort(df["item_id"].unique()))
    item_skill_name = df.drop_duplicates("item_id").set_index("item_id")["source_skill"].reindex(item_index).to_numpy()
    skills = pd.Index(np.sort(df["source_skill"].unique()))
    item_skill = skills.get_indexer(item_skill_name)
    t0 = time.time()
    rows = []

    def make(train_splits, target_split, users=None, cold_spec=None):
        ph = Phase(df, train_splits, target_split, item_index, item_skill, users, cold_spec)
        ph.item_skill_name = item_skill_name
        return ph

    # ---- val phase: fit LTR and lambda
    ph_v = make(("history",), "validation")
    sc_v = Scorer(df, ph_v, item_index, skills, cfg_mastery, seed,
                  mf={"als": (user_factors(F := fit_mf("als", ph_v.Xall, seed), ph_v.X), F)})
    rng = np.random.default_rng(seed)
    ranked_v, data = run_phase(sc_v, ph_v, rng, collect=True)
    ltr = fit_ltr(data, seed)
    lam_scores = {}
    for lam in (0.0, 0.1, 0.2, 0.4):
        r, _ = run_phase(sc_v, ph_v, np.random.default_rng(seed), ltr=ltr, div_lambda=lam)
        m = metrics_for(r["hybrid_ltr_div"], ph_v, sc_v.phat_by_user if hasattr(sc_v, "phat_by_user") else
                        (lambda u, it: sc_v.phat(ph_v.uidx[u], it)))
        lam_scores[lam] = m["ndcg@10"]
    # rule written before the run: the largest lambda whose val NDCG@10 is within 2% of lambda = 0
    lam = max(l for l, v in lam_scores.items() if v >= 0.98 * lam_scores[0.0])
    print(f"val phase done ({time.time() - t0:.0f} s); lambda {lam}; val ndcg {lam_scores}")

    # ---- test phase
    def evaluate(ph, setting):
        mf_all = {}
        out = []
        for s in SEEDS:
            mf = {}
            for kind in ("als", "bpr"):
                F = fit_mf(kind, ph.Xall, s)
                mf[kind] = (user_factors(F, ph.X), F)
            sc = Scorer(df, ph, item_index, skills, cfg_mastery, s, mf=mf)
            r, _ = run_phase(sc, ph, np.random.default_rng(s), ltr=ltr, div_lambda=lam)
            for m, rk in r.items():
                if s != SEEDS[0] and m in ("most_popular", "itemknn", "cb_zpd"):
                    continue  # deterministic: one seed is enough
                out.append({"setting": setting, "model": m, "seed": s,
                            **metrics_for(rk, ph, lambda u, it, sc=sc: sc.phat(ph.uidx[u], it))})
        return out

    ph_t = make(("history", "validation"), "test")
    rows += evaluate(ph_t, "all")
    if cold:
        users = np.sort(df.loc[df["split"] == "test", "user_id"].unique())
        cu = np.random.default_rng(seed).choice(users, size=max(1, len(users) // 5), replace=False)
        for k in (0, 1, 2, 4):
            ph_c = make(("history",), "test", users=np.sort(cu), cold_spec={"users": set(cu), "k": k})
            rows += [dict(r, setting=f"cold_k{k}") for r in evaluate(ph_c, f"cold_k{k}")]
    res = pd.DataFrame(rows)
    agg = res.groupby(["setting", "model"]).agg(**{c: (c, "mean") for c in
                                                   ["recall@10", "ndcg@10", "mrr@10", "zpd_gap@10", "skill_cover@10",
                                                    "new_item_recall@10", "n_learners"]},
                                                ndcg_sd=("ndcg@10", "std"), n_seeds=("seed", "nunique")).reset_index()
    imp = pd.DataFrame({"feature": FEATURES, "gain": ltr.booster_.feature_importance("gain")})
    imp["share"] = imp["gain"] / imp["gain"].sum()
    return agg, res, imp, {"div_lambda": lam, "val_ndcg_by_lambda": lam_scores}


def mastery_cfg(tables: Path, dataset: str) -> dict:
    ch = json.loads((tables / f"b1_variants_choices_{dataset}.json").read_text(encoding="utf-8"))
    coef = ch["item_coef(intercept,logitM,logit_e,log1p_n,se)"]
    return {"alpha": float(ch["ema"]["alpha"]), "coef": coef}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--dataset", required=True, choices=["assist2009", "assist2012", "junyi2015"])
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--sample-users", type=int, default=None, help="evaluate on a random subset (speed)")
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    out_root = REPO / cfg["output_dir"]
    tables = REPO / cfg.get("tables_dir", "reports/tables")
    df = pd.read_parquet(out_root / f"interactions_{args.dataset}.parquet")
    if args.sample_users:
        from ..datasets.splits import sample_users
        df = sample_users(df, args.sample_users, int(cfg["data_seed"]))
    agg, res, imp, choice = build(df, mastery_cfg(tables, args.dataset), int(cfg["data_seed"]), args.cold)
    agg.round(5).to_csv(tables / f"b4_task_ranking_{args.dataset}.csv", index=False)
    res.round(5).to_csv(out_root / "predictions" / f"b4_task_ranking_seeds_{args.dataset}.csv", index=False)
    imp.round(4).to_csv(tables / f"b4_ltr_importance_{args.dataset}.csv", index=False)
    (tables / f"b4_choices_{args.dataset}.json").write_text(json.dumps(choice, indent=2), encoding="utf-8")
    print(agg.round(4).to_string(index=False))
    print(imp.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
