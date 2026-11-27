"""Baselines for next-answer prediction on the common interaction table (card P2-01, review B1 and C1).

Protocol (same for every log and every model):

* rows sorted by ``user_id, ts`` (stable); the knowledge component is ``source_skill`` (the log's own skill: skill
  name in ASSISTments, exercise in Junyi), so the logs are compared with their native KCs;
* **prequential**: the prediction of a row only uses rows strictly before it. Rows of one learner with the same ``ts``
  are one answer tagged with several skills (ASSISTments 2009), so learner- and population-level counts are updated
  only after the whole group; inside one (learner, skill) there are no ties;
* parameters (BKT, PFA, AFM weights, EMA alpha, smoothing) are fitted on ``history`` rows only; the few choices
  (EMA alpha, L2 strength of PFA/AFM) are picked on ``validation`` by log-loss; ``test`` is scored once.
  Learner states run through history and validation before test, so test predictions use all earlier answers;
* every metric table carries the ``majority`` row (constant prediction of the history mean) because Junyi is 83% correct.

Models: ``majority``, ``global_mean``, ``skill_mean``, ``user_mean`` (cumulative, smoothed), ``last_answer`` (last answer
of the learner on the skill, else the skill mean), ``ema`` (Layer 1 EMA: prior = skill mean of history, alpha picked on
validation), ``bkt`` (per-skill L0, T, G, S fitted by maximum likelihood, G and S <= 0.3), ``pfa`` (Pavlik et al. 2009),
``pfa_item`` (PFA plus one easiness weight per item: the item-aware baseline), ``afm`` (Cen et al. 2006; learner intercept + skill easiness + skill learning rate x opportunities).
DKT / SAKT / AKT are run on Kaggle with pykt (``requirements-kaggle.txt``) on the same splits.

Outputs: ``reports/tables/b1_baselines_<dataset>.csv`` (validation and test metrics), ``b1_reliability_<dataset>.csv``,
``<output_dir>/predictions/b1_<dataset>.parquet`` (row id, split, y, one column per model) for the paired bootstrap of
P2-10, and ``b1_choices_<dataset>.json`` (picked alpha / C).

    python -m lmsrag.mastery.baselines --dataset assist2009
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from ..datasets.build import REPO, load_config
from .metrics import metrics, reliability

EMA_ALPHAS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
L2_CS = (0.01, 0.1, 1.0)
SMOOTH_M = 10.0  # pseudo-counts towards the parent mean for skill_mean / user_mean / skill priors


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.reset_index(drop=True).copy()
    d["row"] = np.arange(len(d))
    d = d.sort_values(["user_id", "ts", "row"], kind="mergesort").reset_index(drop=True)
    d["skill"] = d["source_skill"].astype(str)
    d["y"] = d["correct"].astype("float64")
    d["uk"] = d.groupby(["user_id", "skill"], sort=False).ngroup()
    d["u"] = d.groupby("user_id", sort=False).ngroup()
    d["k"] = pd.factorize(d["skill"])[0]
    # prior attempts / successes of the learner on the skill (no ties inside (user, skill))
    d["n_prev"] = d.groupby("uk").cumcount().astype("float64")
    d["s_prev"] = d.groupby("uk")["y"].cumsum() - d["y"]
    d["f_prev"] = d["n_prev"] - d["s_prev"]
    return d


def _cum_before(d: pd.DataFrame, keys: list[str], order: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(successes, attempts) of all rows of the same ``keys`` with a strictly smaller ``ts``; ties share the state."""
    g = d.groupby(keys + ["ts"], sort=False)["y"].agg(["sum", "size"]).reset_index().sort_values(keys + ["ts"])
    g["s_cum"] = g.groupby(keys)["sum"].cumsum() - g["sum"] if keys else g["sum"].cumsum() - g["sum"]
    g["n_cum"] = g.groupby(keys)["size"].cumsum() - g["size"] if keys else g["size"].cumsum() - g["size"]
    m = d[keys + ["ts"]].merge(g[keys + ["ts", "s_cum", "n_cum"]], on=keys + ["ts"], how="left")
    return m["s_cum"].to_numpy(), m["n_cum"].to_numpy()


def mean_models(d: pd.DataFrame, hist_mean: float) -> dict[str, np.ndarray]:
    gs, gn = _cum_before(d, [], [])
    glob = (gs + hist_mean) / (gn + 1.0)
    ks, kn = _cum_before(d, ["skill"], [])
    skill = (ks + SMOOTH_M * glob) / (kn + SMOOTH_M)
    us, un = _cum_before(d, ["user_id"], [])
    user = (us + SMOOTH_M * glob) / (un + SMOOTH_M)
    return {"global_mean": glob, "skill_mean": skill, "user_mean": user}


def skill_priors(d: pd.DataFrame) -> np.ndarray:
    """Smoothed history mean of each row's skill (unseen skill -> history mean)."""
    h = d[d["split"] == "history"]
    mu = h["y"].mean()
    st = h.groupby("skill")["y"].agg(["sum", "size"])
    prior = (st["sum"] + SMOOTH_M * mu) / (st["size"] + SMOOTH_M)
    return d["skill"].map(prior).fillna(mu).to_numpy()


def last_answer(d: pd.DataFrame, prior: np.ndarray) -> np.ndarray:
    prev = d.groupby("uk")["y"].shift(1)
    return np.where(prev.isna(), prior, prev.to_numpy())


def _ema_loop(y, uk, prior, alpha, n_groups):
    state = np.full(n_groups, np.nan)
    out = np.empty(len(y))
    for i in range(len(y)):
        m = state[uk[i]]
        if m != m:  # first answer on the skill: start from the prior
            m = prior[i]
        out[i] = m
        state[uk[i]] = alpha * y[i] + (1 - alpha) * m
    return out


def ema(d: pd.DataFrame, prior: np.ndarray, alpha: float) -> np.ndarray:
    """M before the update; M <- alpha * y + (1 - alpha) * M, M0 = skill prior (layer1 LearnerState.update)."""
    uk = d["uk"].to_numpy()
    return _ema_loop(d["y"].to_numpy(), uk, np.asarray(prior, dtype="float64"), float(alpha), int(uk.max()) + 1)


# ---------------------------------------------------------------- BKT
def _bkt_seq_arrays(d: pd.DataFrame):
    """Rows ordered by (skill, user, ts) with a flag for the first row of each (user, skill) sequence."""
    o = np.lexsort((np.arange(len(d)), d["uk"].to_numpy(), d["k"].to_numpy()))
    k = d["k"].to_numpy()[o]
    first = np.r_[True, d["uk"].to_numpy()[o][1:] != d["uk"].to_numpy()[o][:-1]]
    return o, k, first


def bkt_forward(y: np.ndarray, first: np.ndarray, L0: float, T: float, G: float, S: float) -> np.ndarray:
    """P(correct) before each observation along concatenated sequences."""
    p = np.empty(len(y))
    L = L0
    for i in range(len(y)):
        if first[i]:
            L = L0
        pc = L * (1 - S) + (1 - L) * G
        p[i] = pc
        if y[i] > 0.5:
            post = L * (1 - S) / pc
        else:
            post = L * S / (1 - pc)
        L = post + (1 - post) * T
    return p


try:  # numba makes BKT fitting ~100x faster; the pure-Python path gives the same numbers
    from numba import njit
    bkt_forward = njit(cache=True)(bkt_forward)
    _ema_loop = njit(cache=True)(_ema_loop)
except ImportError:  # pragma: no cover
    pass

BKT_BOUNDS = [(1e-3, 0.999), (1e-3, 0.5), (1e-3, 0.3), (1e-3, 0.3)]


def fit_bkt_skill(y: np.ndarray, first: np.ndarray) -> np.ndarray:
    def nll(x):
        p = np.clip(bkt_forward(y, first, *x), 1e-6, 1 - 1e-6)
        return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    best = None
    for x0 in ((0.4, 0.1, 0.2, 0.1), (0.7, 0.05, 0.15, 0.1), (0.2, 0.2, 0.25, 0.05)):
        r = minimize(nll, x0, method="L-BFGS-B", bounds=BKT_BOUNDS)
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def bkt(d: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    o, k, first = _bkt_seq_arrays(d)
    y = d["y"].to_numpy()[o]
    hist = (d["split"].to_numpy()[o] == "history")
    out = np.empty(len(d))
    params = []
    default = None
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    ends = np.r_[starts[1:], len(k)]
    for a, b in zip(starts, ends):
        h = hist[a:b]
        if h.sum() >= 20:  # history rows come first inside each (user, skill) sequence
            x = fit_bkt_skill(y[a:b][h], first[a:b][h])
        else:
            if default is None:
                default = fit_bkt_skill(y[hist], first[hist])
            x = default
        params.append((int(k[a]), *x))
        out[o[a:b]] = bkt_forward(y[a:b], first[a:b], *x)
    return out, pd.DataFrame(params, columns=["k", "L0", "T", "G", "S"])


# ---------------------------------------------------------------- PFA / AFM
def _design(d: pd.DataFrame, kind: str) -> sparse.csr_matrix:
    n, K = len(d), int(d["k"].max()) + 1
    r = np.arange(n)
    k = d["k"].to_numpy()
    blocks = [sparse.csr_matrix((np.ones(n), (r, k)), shape=(n, K))]
    if kind in ("pfa", "pfa_item"):
        blocks += [sparse.csr_matrix((np.log1p(d["s_prev"].to_numpy()), (r, k)), shape=(n, K)),
                   sparse.csr_matrix((np.log1p(d["f_prev"].to_numpy()), (r, k)), shape=(n, K))]
    if kind == "pfa_item":  # PFA + one easiness weight per item (an item-aware baseline, as fair as ema+item)
        items = pd.factorize(d["item_id"])[0]
        blocks.append(sparse.csr_matrix((np.ones(n), (r, items)), shape=(n, int(items.max()) + 1)))
    elif kind == "afm":
        U = int(d["u"].max()) + 1
        blocks += [sparse.csr_matrix((np.ones(n), (r, d["u"].to_numpy())), shape=(n, U)),
                   sparse.csr_matrix((np.log1p(d["n_prev"].to_numpy()), (r, k)), shape=(n, K))]
    return sparse.hstack(blocks, format="csr")


def logistic_model(d: pd.DataFrame, kind: str, seed: int) -> tuple[np.ndarray, float]:
    """Fit on history for each C, pick by validation log-loss; return predictions of the picked C and C.

    Counts are log1p-transformed (a common variant; raw counts let a 200-attempt sequence dominate)."""
    X, y = _design(d, kind), d["y"].to_numpy()
    h, v = (d["split"] == "history").to_numpy(), (d["split"] == "validation").to_numpy()
    best = None
    for C in L2_CS:
        m = LogisticRegression(C=C, max_iter=300, solver="lbfgs", random_state=seed).fit(X[h], y[h])
        p = m.predict_proba(X)[:, 1]
        ll = metrics(y[v], p[v])["logloss"]
        if best is None or ll < best[0]:
            best = (ll, C, p)
    return best[2], best[1]


# ---------------------------------------------------------------- run
def run(df: pd.DataFrame, seed: int = 2026, models: tuple[str, ...] | None = None) -> dict:
    d = prepare(df)
    h, v = (d["split"] == "history").to_numpy(), (d["split"] == "validation").to_numpy()
    y = d["y"].to_numpy()
    mu = float(y[h].mean())
    preds: dict[str, np.ndarray] = {"majority": np.full(len(d), mu)}
    preds.update(mean_models(d, mu))
    prior = skill_priors(d)
    preds["last_answer"] = last_answer(d, prior)
    choices: dict[str, float] = {}
    best = None
    for a in EMA_ALPHAS:
        p = ema(d, prior, a)
        ll = metrics(y[v], p[v])["logloss"]
        if best is None or ll < best[0]:
            best = (ll, a, p)
    preds["ema"], choices["ema_alpha"] = best[2], best[1]
    preds["bkt"], bkt_params = bkt(d)
    for kind in ("pfa", "pfa_item", "afm"):
        preds[kind], choices[f"{kind}_C"] = logistic_model(d, kind, seed)
    if models:
        preds = {m: p for m, p in preds.items() if m in models}
    rows = []
    for split in ("validation", "test"):
        s = (d["split"] == split).to_numpy()
        for m, p in preds.items():
            rows.append({"split": split, "model": m, **metrics(y[s], p[s])})
    t = (d["split"] == "test").to_numpy()
    rel = pd.concat([reliability(y[t], p[t]).assign(model=m) for m, p in preds.items()], ignore_index=True)
    pred = pd.DataFrame({"row": d["row"], "user_id": d["user_id"], "split": d["split"], "y": d["correct"],
                         **{m: p.astype("float32") for m, p in preds.items()}})
    return {"table": pd.DataFrame(rows), "reliability": rel, "predictions": pred, "choices": choices,
            "bkt_params": bkt_params}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--dataset", required=True, choices=["assist2009", "assist2012", "junyi2015"])
    ap.add_argument("--interactions", default=None, help="default: <output_dir>/interactions_<dataset>.parquet")
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    out_root = REPO / cfg["output_dir"]
    path = Path(args.interactions) if args.interactions else out_root / f"interactions_{args.dataset}.parquet"
    t0 = time.time()
    res = run(pd.read_parquet(path), seed=int(cfg["data_seed"]))
    tables = REPO / cfg.get("tables_dir", "reports/tables")
    res["table"].round(4).to_csv(tables / f"b1_baselines_{args.dataset}.csv", index=False)
    res["reliability"].round(4).to_csv(tables / f"b1_reliability_{args.dataset}.csv", index=False)
    (out_root / "predictions").mkdir(parents=True, exist_ok=True)
    res["predictions"].to_parquet(out_root / "predictions" / f"b1_{args.dataset}.parquet", index=False)
    res["bkt_params"].to_csv(out_root / "predictions" / f"b1_bkt_params_{args.dataset}.csv", index=False)
    (tables / f"b1_choices_{args.dataset}.json").write_text(json.dumps(res["choices"], indent=2), encoding="utf-8")
    print(res["table"][res["table"]["split"] == "test"].round(4).to_string(index=False))
    print("choices:", res["choices"], f"- {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
