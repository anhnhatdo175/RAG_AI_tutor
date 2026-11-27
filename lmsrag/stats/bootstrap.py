"""Paired cluster bootstrap, Holm correction and effect sizes (card P2-10, review C4).

* Prediction (P2-01/P2-03): clusters = learners. One resample draws learners with replacement; every model is scored
  on the same resampled rows, so a difference is paired. AUC uses row weights (number of times the learner was drawn),
  computed in O(n) from a presorted order, so 1,000 resamples on 760,000 Junyi rows take seconds.
* Retrieval / ranking (P2-05/P2-06): clusters = queries (or learners), metric = mean of a per-query score.
* p value: two-sided bootstrap p = min(1, 2 min(P(d <= 0), P(d >= 0))). Holm controls the family-wise error over
  all comparisons of one table. Effect size: the difference itself (metric units), its 95% percentile interval,
  and for AUC the share of the reference model's lift over chance (``(AUC_a - AUC_b) / (AUC_b - 0.5)``).

    python -m lmsrag.stats.bootstrap --dataset assist2009      # reads predictions of P2-01/P2-03

DKT/SAKT predictions (``predictions/b1_deep_<model>_<dataset>.parquet``, Kaggle) are added when present.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.build import REPO, load_config

EPS = 1e-6


class WeightedAUC:
    """AUC with integer row weights; the score order is computed once."""

    def __init__(self, y: np.ndarray, p: np.ndarray):
        o = np.argsort(p, kind="mergesort")
        self.y, self.p, self.o = y[o].astype(bool), p[o], o
        # ties: rows with equal p share a block; a positive scores 0.5 per negative in its block
        self.block = np.r_[0, np.cumsum(self.p[1:] != self.p[:-1])]

    def __call__(self, w: np.ndarray) -> float:
        w = w[self.o].astype("float64")
        wn, wp = np.where(self.y, 0.0, w), np.where(self.y, w, 0.0)
        neg_block = np.bincount(self.block, weights=wn)
        neg_below = np.r_[0.0, np.cumsum(neg_block)[:-1]]
        score = (neg_below[self.block] + 0.5 * neg_block[self.block]) * wp
        P, N = wp.sum(), wn.sum()
        return float(score.sum() / (P * N)) if P > 0 and N > 0 else float("nan")


def logloss_w(y, p, w):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum())


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype="float64")
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def p_two_sided(d: np.ndarray) -> float:
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def cluster_weights(cluster_codes: np.ndarray, n_clusters: int, n: int, seed: int):
    """Yields, per resample, the integer weight of every row: how many times its cluster was drawn."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield np.bincount(rng.integers(0, n_clusters, size=n_clusters), minlength=n_clusters)[cluster_codes]


def compare_predictions(pred: pd.DataFrame, models: list[str], pairs: list[tuple[str, str]], n: int = 1000,
                        seed: int = 0, split: str = "test") -> pd.DataFrame:
    t = pred[pred["split"] == split]
    y = t["y"].to_numpy(dtype="float64")
    codes, uniq = pd.factorize(t["user_id"])
    weights = cluster_weights(codes, len(uniq), n, seed)
    aucs = {m: WeightedAUC(y, t[m].to_numpy(dtype="float64")) for m in models}
    boot = {m: {"AUC": np.empty(n), "logloss": np.empty(n)} for m in models}
    for b, w in enumerate(weights):
        for m in models:
            boot[m]["AUC"][b] = aucs[m](w)
            boot[m]["logloss"][b] = logloss_w(y, t[m].to_numpy(dtype="float64"), w)
    full = {m: {"AUC": aucs[m](np.ones(len(y))), "logloss": logloss_w(y, t[m].to_numpy(dtype="float64"), np.ones(len(y)))}
            for m in models}
    rows = []
    for a, bm in pairs:
        for metric in ("AUC", "logloss"):
            d = boot[a][metric] - boot[bm][metric]
            diff = full[a][metric] - full[bm][metric]
            row = {"model": a, "vs": bm, "metric": metric, "value_model": full[a][metric], "value_vs": full[bm][metric],
                   "diff": diff, "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
                   "p_boot": p_two_sided(d), "n_learners": len(uniq), "n_resamples": n}
            if metric == "AUC" and full[bm][metric] > 0.5:
                row["rel_lift_over_chance"] = diff / (full[bm][metric] - 0.5)
            rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_boot"].to_numpy())
    out["significant_holm_0.05"] = out["p_holm"] < 0.05
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--dataset", required=True, choices=["assist2009", "assist2012", "junyi2015"])
    ap.add_argument("--proposed", default="ema_skill+item")
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    out_root = REPO / cfg["output_dir"]
    pred = pd.read_parquet(out_root / "predictions" / f"b1v_{args.dataset}.parquet")
    for kind in ("dkt", "sakt"):  # Kaggle runs of lmsrag.mastery.deep_kt, when present
        f = out_root / "predictions" / f"b1_deep_{kind}_{args.dataset}.parquet"
        if f.exists():
            deep = pd.read_parquet(f, columns=["row", kind])
            pred = pred.merge(deep, on="row", how="left")
            pred = pred[(pred["split"] != "test") | pred[kind].notna()]
    base = [m for m in ("ema", "bkt", "pfa", "pfa_item", "afm", "dkt", "sakt", "majority") if m in pred.columns]
    models = [args.proposed, *base]
    pairs = [(args.proposed, b) for b in base]
    n = int(cfg.get("bootstrap", {}).get("n_resamples", 1000))
    res = compare_predictions(pred, models, pairs, n=n, seed=int(cfg["data_seed"]))
    tables = REPO / cfg.get("tables_dir", "reports/tables")
    res.round(5).to_csv(tables / f"c4_prediction_bootstrap_{args.dataset}.csv", index=False)
    print(res.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
