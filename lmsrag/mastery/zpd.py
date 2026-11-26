"""Probability-based ZPD (card P2-04, review B2) against the old threshold rule of Bảng 2.4, on real logs.

Each answered row is a (learner state, item) pair. A ZPD rule says whether the item was "in the zone" for the learner
just before the answer. Rules:

* ``old``: delta = difficulty - mastery in [0.05, 0.30] (``layer4_recommender.zpd_score`` == 1), with mastery = the
  learner's EMA on the skill and difficulty = 1 - e_i (item failure rate in history, the 0-1 scale the rule expects);
* ``prob[lo, hi]``: p_hat(u, i) in [lo, hi], p_hat = the item-aware calibrated EMA of P2-03 (``<best>+item``).

What a ZPD rule should deliver is items the learner gets right with probability about 0.6-0.8. So for the in-zone rows
of each rule we report: ``coverage`` (share of rows in zone), ``acc_in`` (observed accuracy), ``gap`` = |acc_in - 0.7|,
``in_band`` (share of in-zone rows whose own p_hat-bin accuracy, 0.05 bins, lies in [0.6, 0.8]). The band of ``prob``
is swept on validation (sensitivity table and figure), the band with the smallest gap among those with coverage >= 10%
is chosen there, and test is scored once for the chosen band and the old rule.

    python -m lmsrag.mastery.zpd --dataset assist2009
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.build import REPO, load_config
from . import variants as V

LOS = (0.45, 0.5, 0.55, 0.6, 0.65)
HIS = (0.75, 0.8, 0.85, 0.9)
TARGET = 0.7


def zone_stats(y: np.ndarray, p: np.ndarray, inz: np.ndarray) -> dict[str, float]:
    if not inz.any():
        return {"coverage": 0.0, "acc_in": float("nan"), "gap": float("nan"), "in_band": float("nan"), "n_in": 0}
    yb, pb = y[inz], p[inz]
    bins = np.minimum((pb * 20).astype(int), 19)
    acc_bin = pd.Series(yb).groupby(bins).transform("mean").to_numpy()
    acc = float(yb.mean())
    return {"coverage": float(inz.mean()), "acc_in": acc, "gap": abs(acc - TARGET),
            "in_band": float(((acc_bin >= 0.6) & (acc_bin <= 0.8)).mean()), "n_in": int(inz.sum())}


def run(df: pd.DataFrame, seed: int = 2026) -> dict:
    res = V.run(df, seed)
    ctx, preds = res["ctx"], res["predictions"]
    best = res["choices"]["best_raw"]
    m = preds[best].to_numpy(dtype="float64")
    p = preds[f"{best}+item"].to_numpy(dtype="float64")
    delta = (1.0 - ctx.e_item) - m
    old = (delta >= 0.05) & (delta <= 0.30)
    rows = []
    for split, mask in (("validation", ctx.v), ("test", ctx.t)):
        y = ctx.y[mask]
        rows.append({"split": split, "rule": "old", "lo": None, "hi": None, **zone_stats(y, p[mask], old[mask])})
        for lo in LOS:
            for hi in HIS:
                inz = (p[mask] >= lo) & (p[mask] <= hi)
                rows.append({"split": split, "rule": "prob", "lo": lo, "hi": hi, **zone_stats(y, p[mask], inz)})
    t = pd.DataFrame(rows)
    val = t[(t.split == "validation") & (t.rule == "prob") & (t.coverage >= 0.10)]
    pick = val.sort_values(["gap", "coverage"], ascending=[True, False]).iloc[0]
    t["chosen"] = (t.rule == "prob") & (t.lo == pick.lo) & (t.hi == pick.hi)
    return {"table": t, "choice": {"lo": float(pick.lo), "hi": float(pick.hi), "p_hat_model": f"{best}+item"}}


def plot(t: pd.DataFrame, path: Path, dataset: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = t[(t.split == "validation") & (t.rule == "prob")]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for hi, g in v.groupby("hi"):
        ax[0].plot(g["lo"], g["acc_in"], marker="o", label=f"hi = {hi}")
        ax[1].plot(g["lo"], g["coverage"], marker="o", label=f"hi = {hi}")
    o = t[(t.split == "validation") & (t.rule == "old")].iloc[0]
    ax[0].axhline(o["acc_in"], ls="--", c="grey", label="old rule")
    ax[1].axhline(o["coverage"], ls="--", c="grey")
    ax[0].axhspan(0.6, 0.8, color="0.9", zorder=0)
    ax[0].set(xlabel="lower bound of p_hat", ylabel="observed accuracy in zone")
    ax[1].set(xlabel="lower bound of p_hat", ylabel="share of rows in zone")
    ax[0].legend(fontsize=7)
    fig.suptitle(f"ZPD band sensitivity, validation, {dataset}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--dataset", required=True, choices=["assist2009", "assist2012", "junyi2015"])
    ap.add_argument("--interactions", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    out_root = REPO / cfg["output_dir"]
    path = Path(args.interactions) if args.interactions else out_root / f"interactions_{args.dataset}.parquet"
    r = run(pd.read_parquet(path), int(cfg["data_seed"]))
    tables = REPO / cfg.get("tables_dir", "reports/tables")
    r["table"].round(4).to_csv(tables / f"b2_zpd_{args.dataset}.csv", index=False)
    (tables / f"b2_zpd_choice_{args.dataset}.json").write_text(json.dumps(r["choice"], indent=2), encoding="utf-8")
    plot(r["table"], REPO / "reports" / "figures" / f"b2_zpd_sensitivity_{args.dataset}.png", args.dataset)
    t = r["table"]
    print(t[(t.split == "test") & ((t.rule == "old") | t.chosen)].round(4).to_string(index=False))
    print("chosen on validation:", r["choice"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
