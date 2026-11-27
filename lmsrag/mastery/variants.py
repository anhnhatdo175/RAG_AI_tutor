"""EMA mastery variants and calibration (card P2-03, review B1), compared with the P2-01 baselines.

Same protocol as ``baselines.py`` (prequential, KC = ``source_skill``, fit on history, pick on validation, test once).

Raw mastery variants, each an EMA ``M <- M + a_t * (y - M)`` started at the skill prior:

``ema``             fixed alpha (P2-01 baseline, alpha picked on validation);
``ema_count``       (i) alpha shrinks with practice: ``a_t = max(a_min, 1 / (n + c))``, n = earlier answers on the skill;
``ema_skill``       (i) one alpha per skill, picked on the skill's history rows (skills with < 50 rows keep the global);
``ema_diff``        (ii) difficulty-weighted: a right answer on a hard item and a wrong answer on an easy item move M
                    more: ``a_t = alpha * (1 + lam * (1 - e_i))`` if right, ``alpha * (1 + lam * e_i)`` if wrong, capped
                    at 1; ``e_i`` = smoothed history success rate of the item (unseen item -> skill prior);
``ema_count_diff``  (i) and (ii) together.

(iii) uncertainty: every variant also gives ``n`` (earlier answers on the skill) and ``se = sqrt(M (1 - M) / (n + 1))``;
both go into the learner profile (``profile_frame``) and into the item-aware calibrator below.

(iv) calibration on top of the best raw variant (picked on validation log-loss), fitted on history predictions:
``+platt`` (logistic on logit M), ``+isotonic``, and ``+item`` = logistic on [logit M, logit e_i, log1p n, se] -- the
probability of a right answer on item i, ``p_hat(u, i)``, that the probability-based ZPD of P2-04 uses.

    python -m lmsrag.mastery.variants --dataset assist2009
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..datasets.build import REPO, load_config
from . import baselines as B
from .metrics import metrics, reliability

A_MIN = (0.05, 0.1, 0.15, 0.2)
C_COUNT = (1.0, 2.0, 4.0)
LAMBDAS = (0.5, 1.0, 2.0)


def _ema_general(y, uk, prior, n_prev, e_item, alpha, a_min, c, lam, mode, n_groups):
    """mode: 0 fixed, 1 count, 2 diff, 3 count+diff. alpha may be per row (per-skill alphas)."""
    state = np.full(n_groups, np.nan)
    out = np.empty(len(y))
    for i in range(len(y)):
        m = state[uk[i]]
        if m != m:
            m = prior[i]
        out[i] = m
        a = alpha[i]
        if mode == 1 or mode == 3:
            a = max(a_min, 1.0 / (n_prev[i] + c))
        if mode == 2 or mode == 3:
            w = 1.0 + lam * (1.0 - e_item[i]) if y[i] > 0.5 else 1.0 + lam * e_item[i]
            a = min(1.0, a * w)
        state[uk[i]] = m + a * (y[i] - m)
    return out


try:
    from numba import njit
    _ema_general = njit(cache=True)(_ema_general)
except ImportError:  # pragma: no cover
    pass


class Ctx:
    def __init__(self, df: pd.DataFrame):
        d = B.prepare(df)
        self.d = d
        self.y = d["y"].to_numpy()
        self.uk = d["uk"].to_numpy()
        self.ng = int(self.uk.max()) + 1
        self.n_prev = d["n_prev"].to_numpy()
        self.h = (d["split"] == "history").to_numpy()
        self.v = (d["split"] == "validation").to_numpy()
        self.t = (d["split"] == "test").to_numpy()
        self.prior = B.skill_priors(d)
        hist = d[self.h]
        st = hist.groupby("item_id")["y"].agg(["sum", "size"])
        s = d["item_id"].map(st["sum"]).fillna(0).to_numpy()
        n = d["item_id"].map(st["size"]).fillna(0).to_numpy()
        # smoothed towards the skill prior; on a history row the row itself is excluded (no self-information)
        own = np.where(self.h, self.y, 0.0)
        own_n = self.h.astype(float)
        self.e_item = np.clip((s - own + B.SMOOTH_M * self.prior) / (n - own_n + B.SMOOTH_M), 0.02, 0.98)

    def ema(self, mode=0, alpha=0.2, a_min=0.1, c=2.0, lam=0.0, alpha_rows=None):
        a = np.full(len(self.y), alpha) if alpha_rows is None else alpha_rows
        return _ema_general(self.y, self.uk, self.prior, self.n_prev, self.e_item, a, a_min, c, lam, mode, self.ng)

    def ll(self, p, mask):
        return metrics(self.y[mask], p[mask])["logloss"]


def pick(ctx: Ctx, grid: list[dict], mode: int) -> tuple[np.ndarray, dict]:
    best = None
    for g in grid:
        p = ctx.ema(mode=mode, **g)
        ll = ctx.ll(p, ctx.v)
        if best is None or ll < best[0]:
            best = (ll, g, p)
    return best[2], best[1]


def per_skill_alpha(ctx: Ctx, global_alpha: float) -> np.ndarray:
    """Alpha of each skill picked on its history rows (log-loss); skills with < 50 history rows keep the global."""
    k = ctx.d["k"].to_numpy()
    runs = {a: ctx.ema(alpha=a) for a in B.EMA_ALPHAS}
    y = ctx.y
    best = np.full(k.max() + 1, global_alpha)
    hk = pd.Series(k[ctx.h])
    counts = hk.value_counts()
    for kk in counts.index[counts >= 50]:
        m = ctx.h & (k == kk)
        lls = {a: -np.mean(y[m] * np.log(np.clip(p[m], 1e-6, 1)) + (1 - y[m]) * np.log(np.clip(1 - p[m], 1e-6, 1)))
               for a, p in runs.items()}
        best[kk] = min(lls, key=lls.get)
    return best[k]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def features(m: np.ndarray, ctx: Ctx) -> np.ndarray:
    se = np.sqrt(m * (1 - m) / (ctx.n_prev + 1))
    return np.column_stack([_logit(m), _logit(ctx.e_item), np.log1p(ctx.n_prev), se])


def calibrate(ctx: Ctx, m: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    h = ctx.h
    out = {}
    lr = LogisticRegression(C=1.0, random_state=seed).fit(_logit(m)[h, None], ctx.y[h])
    out["platt"] = lr.predict_proba(_logit(m)[:, None])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(m[h], ctx.y[h])
    out["isotonic"] = iso.predict(m)
    X = features(m, ctx)
    lr2 = LogisticRegression(C=1.0, random_state=seed).fit(X[h], ctx.y[h])
    out["item"] = lr2.predict_proba(X)[:, 1]
    out["_item_coef"] = np.r_[lr2.intercept_, lr2.coef_.ravel()]
    return out


def run(df: pd.DataFrame, seed: int = 2026, baseline_preds: pd.DataFrame | None = None) -> dict:
    ctx = Ctx(df)
    preds, choices = {}, {}
    preds["ema"], g = pick(ctx, [{"alpha": a} for a in B.EMA_ALPHAS], 0)
    choices["ema"] = g
    preds["ema_count"], choices["ema_count"] = pick(ctx, [{"a_min": a, "c": c} for a in A_MIN for c in C_COUNT], 1)
    preds["ema_skill"] = ctx.ema(alpha_rows=per_skill_alpha(ctx, g["alpha"]))
    choices["ema_skill"] = {"fallback_alpha": g["alpha"]}
    preds["ema_diff"], choices["ema_diff"] = pick(ctx, [{"alpha": a, "lam": l} for a in B.EMA_ALPHAS for l in LAMBDAS], 2)
    preds["ema_count_diff"], choices["ema_count_diff"] = pick(
        ctx, [{"a_min": a, "c": c, "lam": l} for a in A_MIN for c in C_COUNT for l in LAMBDAS], 3)
    raw = [m for m in preds]
    best_raw = min(raw, key=lambda m: ctx.ll(preds[m], ctx.v))
    choices["best_raw"] = best_raw
    cal = calibrate(ctx, preds[best_raw], seed)
    choices["item_coef(intercept,logitM,logit_e,log1p_n,se)"] = [round(float(x), 4) for x in cal.pop("_item_coef")]
    for name, p in cal.items():
        preds[f"{best_raw}+{name}"] = p
    if baseline_preds is not None:  # P2-01 predictions, aligned on the original row id
        bp = baseline_preds.set_index("row").loc[ctx.d["row"]]
        for m in ("bkt", "pfa", "afm", "majority"):
            if m in bp:
                preds[m] = bp[m].to_numpy(dtype="float64")
    rows = []
    for split, mask in (("validation", ctx.v), ("test", ctx.t)):
        for m, p in preds.items():
            rows.append({"split": split, "model": m, **metrics(ctx.y[mask], p[mask])})
    rel = pd.concat([reliability(ctx.y[ctx.t], np.clip(p[ctx.t], 0, 1)).assign(model=m) for m, p in preds.items()],
                    ignore_index=True)
    pred = pd.DataFrame({"row": ctx.d["row"], "user_id": ctx.d["user_id"], "split": ctx.d["split"], "y": ctx.d["correct"],
                         **{m: np.asarray(p, dtype="float32") for m, p in preds.items()}})
    return {"table": pd.DataFrame(rows), "reliability": rel, "predictions": pred, "choices": choices, "ctx": ctx}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--dataset", required=True, choices=["assist2009", "assist2012", "junyi2015"])
    ap.add_argument("--interactions", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    out_root = REPO / cfg["output_dir"]
    path = Path(args.interactions) if args.interactions else out_root / f"interactions_{args.dataset}.parquet"
    bpath = out_root / "predictions" / f"b1_{args.dataset}.parquet"
    t0 = time.time()
    res = run(pd.read_parquet(path), int(cfg["data_seed"]), pd.read_parquet(bpath) if bpath.is_file() else None)
    tables = REPO / cfg.get("tables_dir", "reports/tables")
    res["table"].round(4).to_csv(tables / f"b1_variants_{args.dataset}.csv", index=False)
    res["reliability"].round(4).to_csv(tables / f"b1_variants_reliability_{args.dataset}.csv", index=False)
    res["predictions"].to_parquet(out_root / "predictions" / f"b1v_{args.dataset}.parquet", index=False)
    (tables / f"b1_variants_choices_{args.dataset}.json").write_text(json.dumps(res["choices"], indent=2), encoding="utf-8")
    print(res["table"][res["table"]["split"] == "test"].round(4).to_string(index=False))
    print(json.dumps(res["choices"]), f"- {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
