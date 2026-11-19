"""Difficulty audit (card P1-06, steps 3-4): automatic difficulty against human ratings, and the Khan filter.

1. Người 2 and Người 3 fill ``a3_difficulty_nguoi2.csv`` / ``a3_difficulty_nguoi3.csv`` independently
   (``rating`` 1-5; on Khan rows also ``figure_ok`` and ``answer_ok``, có/không).
2. ``python -m lmsrag.labels.difficulty_agreement`` writes ``reports/tables/a3_difficulty_audit.csv`` and lists the
   Khan checks on which the two disagree in ``a3_difficulty_adjudication.csv``.
3. Người 1 fills ``final`` there (có/không); running the command again adds the Khan filter accuracy.

Reported, overall, per source and per ``difficulty_source``: Spearman between ``difficulty`` and the mean of the two
ratings with a bootstrap 95% interval over problems (``data_seed``), the same against each rater, and agreement
between raters (Spearman, κ with quadratic weights on the 1-5 scale, exact and within-one agreement). Also the mean
rating per (``difficulty_level``, source), which shows whether one level means the same across sources. Ratings are
not adjudicated: the card asks for the mean of the two.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.build import REPO, load_config
from .agreement import agreement_rate, bootstrap_ci, cohen_kappa, pabak, spearman, wilson

LABELS = REPO / "lmsrag" / "data" / "labels"
TABLES = REPO / "reports" / "tables"
ANNOTATORS = ("nguoi2", "nguoi3")
SCALE = [1, 2, 3, 4, 5]
CHECKS = ("figure_ok", "answer_ok")
YES = {"có", "co", "yes", "y", "1", "c"}
NO = {"không", "khong", "no", "n", "0", "k"}
NOT_ASKED = {"–", "-", ""}


def _yes_no(value: str) -> float:
    v = str(value).strip().casefold()
    if v in YES:
        return 1.0
    if v in NO:
        return 0.0
    if v in NOT_ASKED:
        return np.nan
    raise ValueError(value)


def read_sheet(path: Path, khan_items: set[str]) -> tuple[pd.DataFrame, list[str]]:
    """``rating`` as a number (NaN if empty) and the two checks as 1/0 (NaN if empty), plus a list of problems."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False).set_index("item")
    errors, out = [], pd.DataFrame(index=df.index)
    rating = df["rating"].str.strip()
    bad = rating[~rating.isin([str(x) for x in SCALE] + [""])]
    errors += [f"{path.name} {i}: rating {v!r} is not 1-5" for i, v in bad.items()]
    out["rating"] = pd.to_numeric(rating.where(~rating.index.isin(bad.index), ""), errors="coerce")
    for c in CHECKS:
        vals = []
        for i, v in df[c].items():
            try:
                x = _yes_no(v)
            except ValueError:
                errors.append(f"{path.name} {i}: {c} {v!r} is not có/không")
                x = np.nan
            if i not in khan_items and not np.isnan(x):
                errors.append(f"{path.name} {i}: {c} is only asked on Khan rows")
            vals.append(x)
        out[c] = vals
    out["note"] = df["note"]
    return out, errors


def _row(level, group, n, metric, value, lo=np.nan, hi=np.nan):
    return {"level": level, "group": group, "n": n, "metric": metric, "value": value, "ci_low": lo, "ci_high": hi}


def rating_rows(level: str, group: str, g: pd.DataFrame, seed: int) -> list[dict]:
    n, rows = len(g), []
    d, mean, r2, r3 = g["difficulty"].to_numpy(float), g["mean_rating"].to_numpy(), g["r2"].to_numpy(), g["r3"].to_numpy()
    wk = lambda a, b: cohen_kappa(a, b, "quadratic", SCALE)
    for metric, fn, a, b in (("spearman_difficulty_vs_mean_rating", spearman, d, mean),
                             ("spearman_between_raters", spearman, r2, r3),
                             ("weighted_kappa_between_raters", wk, r2, r3)):
        lo, hi = bootstrap_ci(fn, a, b, seed) if n > 2 else (np.nan, np.nan)
        rows.append(_row(level, group, n, metric, fn(a, b), lo, hi))
    rows += [_row(level, group, n, "spearman_difficulty_vs_nguoi2", spearman(d, r2)),
             _row(level, group, n, "spearman_difficulty_vs_nguoi3", spearman(d, r3)),
             _row(level, group, n, "exact_agreement", agreement_rate(r2, r3)),
             _row(level, group, n, "within_one_agreement", float(np.mean(np.abs(r2 - r3) <= 1))),
             _row(level, group, n, "rating_gap_ge2", int(np.sum(np.abs(r2 - r3) >= 2))),
             _row(level, group, n, "mean_rating_nguoi2", float(r2.mean())),
             _row(level, group, n, "mean_rating_nguoi3", float(r3.mean()))]
    return rows


def check_rows(k: pd.DataFrame, final: dict, seed: int) -> list[dict]:
    """Khan filter: one block per check, then both checks together (a problem passes if both are 'có')."""
    rows, passed = [], {}
    for c in CHECKS:
        g = k[[f"{c}_2", f"{c}_3"]].dropna()
        a, b, n = g[f"{c}_2"].to_numpy(int), g[f"{c}_3"].to_numpy(int), len(g)
        lo, hi = bootstrap_ci(cohen_kappa, a, b, seed) if n > 1 else (np.nan, np.nan)
        rows += [_row("khan_filter", c, n, "kappa_between_raters", cohen_kappa(a, b), lo, hi),
                 _row("khan_filter", c, n, "agreement_between_raters", agreement_rate(a, b)),
                 # κ collapses when nearly every answer is 'có' (prevalence paradox); PABAK does not
                 _row("khan_filter", c, n, "pabak_between_raters", pabak(a, b) if n else np.nan),
                 _row("khan_filter", c, n, "share_yes_nguoi2", float(a.mean()) if n else np.nan),
                 _row("khan_filter", c, n, "share_yes_nguoi3", float(b.mean()) if n else np.nan),
                 _row("khan_filter", c, n, "pairs_to_adjudicate", int((a != b).sum()))]
        fin = pd.Series(np.where(a == b, a, np.nan), index=g.index)
        for i in fin.index[fin.isna()]:
            v = final.get((i, c), "")
            fin[i] = _yes_no(v) if str(v).strip() and str(v).strip() not in NOT_ASKED else np.nan
        if n and fin.notna().all():
            passed[c] = fin
            s = int(fin.sum())
            rows.append(_row("khan_filter", c, n, "accuracy_after_adjudication", s / n, *wilson(s, n)))
    if len(passed) == len(CHECKS):
        both = (passed[CHECKS[0]] * passed[CHECKS[1]]).dropna()
        s = int(both.sum())
        rows.append(_row("khan_filter", "both", len(both), "accuracy_after_adjudication", s / len(both),
                         *wilson(s, len(both))))
    return rows


def audit(key: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, final: dict, seed: int) -> pd.DataFrame:
    d = key.set_index("item").copy()
    d["r2"], d["r3"] = s2["rating"], s3["rating"]
    for c in CHECKS:
        d[f"{c}_2"], d[f"{c}_3"] = s2[c], s3[c]
    rated = d[d["r2"].notna() & d["r3"].notna()].copy()
    rated["mean_rating"] = (rated["r2"] + rated["r3"]) / 2
    rows = rating_rows("all", "all", rated, seed)
    for level, col in (("source", "source"), ("difficulty_source", "difficulty_source")):
        for name, g in rated.groupby(col):
            rows += rating_rows(level, name, g, seed)
    for (lvl, src), g in rated.groupby(["difficulty_level", "source"]):
        rows.append(_row("level_source", f"L{lvl} {src}", len(g), "mean_rating", float(g["mean_rating"].mean())))
    for lvl, g in rated.groupby("difficulty_level"):
        rows.append(_row("level_source", f"L{lvl} all", len(g), "mean_rating", float(g["mean_rating"].mean())))
    rows += check_rows(d[d["source"] == "khan"], final, seed)
    out = pd.DataFrame(rows)
    for c in ("value", "ci_low", "ci_high"):
        out[c] = out[c].astype(float).round(4)
    return out


def adjudication_sheet(key: pd.DataFrame, sheet: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
                       old: pd.DataFrame | None) -> pd.DataFrame:
    """The Khan checks the two raters answered differently, with Người 1's earlier decisions kept."""
    word = {1.0: "có", 0.0: "không"}
    rows = []
    for i in key.loc[key["source"] == "khan", "item"]:
        for c in CHECKS:
            a, b = s2.at[i, c], s3.at[i, c]
            if np.isnan(a) or np.isnan(b) or a == b:
                continue
            rows.append({"item": i, "check": c, "problem": sheet.at[i, "problem"], "answer": sheet.at[i, "answer"],
                         "nguoi2": word[a], "nguoi3": word[b], "note_nguoi2": s2.at[i, "note"],
                         "note_nguoi3": s3.at[i, "note"], "final": "", "note_final": ""})
    out = pd.DataFrame(rows, columns=["item", "check", "problem", "answer", "nguoi2", "nguoi3", "note_nguoi2",
                                      "note_nguoi3", "final", "note_final"])
    if old is not None and not out.empty:
        prev = old.set_index(["item", "check"])
        for col in ("final", "note_final"):
            out[col] = [prev[col].get((i, c), "") for i, c in zip(out["item"], out["check"])]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    seed = int(load_config(REPO / args.config)["data_seed"])
    key = pd.read_csv(LABELS / "a3_difficulty_key.csv", encoding="utf-8-sig", keep_default_na=False)
    khan = set(key.loc[key["source"] == "khan", "item"])
    (s2, e2), (s3, e3) = (read_sheet(LABELS / f"a3_difficulty_{a}.csv", khan) for a in ANNOTATORS)
    if e2 + e3:
        raise SystemExit("fix the sheets first:\n  " + "\n  ".join(e2 + e3))
    missing = {a: int(s["rating"].isna().sum()) for a, s in zip(ANNOTATORS, (s2, s3))}
    if any(missing.values()):
        print(f"problems without a rating (skipped): {missing}")

    adj_path = LABELS / "a3_difficulty_adjudication.csv"
    old = pd.read_csv(adj_path, encoding="utf-8-sig", dtype=str, keep_default_na=False) if adj_path.is_file() else None
    final = {} if old is None else {(i, c): f for i, c, f in zip(old["item"], old["check"], old["final"])}
    table = audit(key, s2, s3, final, seed)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "a3_difficulty_audit.csv", index=False)
    sheet = pd.read_csv(LABELS / f"a3_difficulty_{ANNOTATORS[0]}.csv", encoding="utf-8-sig", dtype=str,
                        keep_default_na=False).set_index("item")
    adj = adjudication_sheet(key, sheet, s2, s3, old)
    adj.to_csv(adj_path, index=False, encoding="utf-8-sig")

    show = table[table["level"].isin(["all", "source", "khan_filter"])]
    print(show.to_string(index=False))
    print(f"\n{len(adj)} Khan checks to adjudicate in {adj_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
