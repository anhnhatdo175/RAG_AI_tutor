"""Agreement statistics and the audit of the skill dictionary (card P1-05, steps 4 and 6).

Statistics (pure functions, tested against hand-worked examples and scikit-learn / scipy): Cohen's κ, κ with
quadratic weights, Spearman's ρ, raw agreement, PABAK, and accuracy with a Wilson 95% interval. The same functions serve the
difficulty audit (P1-06) and the gold set (P1-07).

Audit workflow:

1. Người 2 and Người 3 fill ``label`` in ``a4_mapping_nguoi2.csv`` / ``a4_mapping_nguoi3.csv`` independently.
2. ``python -m lmsrag.labels.agreement`` reports κ and agreement *before* adjudication and writes the pairs on which
   they disagree to ``a4_mapping_adjudication.csv``.
3. Người 1 fills ``label_final`` there; running the command again adds the accuracy *after* adjudication.

Every statistic is reported overall and per ``method`` (and per pair type) in ``reports/tables/a4_mapping_audit.csv``;
κ and agreement carry a bootstrap 95% interval over items (``data_seed``), accuracy a Wilson interval.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.build import REPO, load_config

LABELS = REPO / "lmsrag" / "data" / "labels"
TABLES = REPO / "reports" / "tables"
ANNOTATORS = ("nguoi2", "nguoi3")
N_BOOT = 1000


# ---------------------------------------------------------------- statistics
def agreement_rate(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(a == b))


def pabak(a, b) -> float:
    """Prevalence- and bias-adjusted κ for two categories, 2·agreement − 1 (Byrt et al., 1993).

    Reported next to κ because κ drops sharply when almost every pair has the same label (most dictionary pairs are
    expected to be correct), even at high agreement.
    """
    return 2.0 * agreement_rate(a, b) - 1.0


def cohen_kappa(a, b, weights: str | None = None) -> float:
    """Cohen's κ of two label vectors; ``weights='quadratic'`` gives the weighted κ for ordinal labels."""
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    if weights is None:
        w = 1.0 - np.eye(k)
    elif weights == "quadratic":
        i, j = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
        w = ((i - j) / max(k - 1, 1)) ** 2
    else:
        raise ValueError(f"unknown weights {weights!r}")
    den = float((w * exp).sum())
    if den == 0.0:  # both annotators used one and the same label: κ is undefined
        return float("nan")
    return 1.0 - float((w * obs).sum()) / den


def spearman(a, b) -> float:
    """Spearman's ρ: Pearson correlation of mid-ranks (ties averaged)."""
    ra = pd.Series(np.asarray(a, float)).rank().to_numpy()
    rb = pd.Series(np.asarray(b, float)).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(stat, a, b, seed: int, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile 95% interval of ``stat(a, b)`` over items resampled with replacement."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        v = stat(a[i], b[i])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- audit
def read_labels(path: Path) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    lab = df.set_index("item")["label"].str.strip()
    bad = lab[~lab.isin(["0", "1", ""])]
    if not bad.empty:
        raise SystemExit(f"{path.name}: labels must be 0 or 1, got {bad.to_dict()}")
    return lab


def audit(key: pd.DataFrame, l2: pd.Series, l3: pd.Series, final: pd.Series | None, seed: int) -> pd.DataFrame:
    """One row per (group, metric): overall, per pair type and per method."""
    d = key.set_index("item").assign(l2=l2, l3=l3)
    d = d[(d["l2"] != "") & (d["l3"] != "")].copy()
    d["l2"], d["l3"] = d["l2"].astype(int), d["l3"].astype(int)
    agreed = d["l2"] == d["l3"]
    d["final"] = np.where(agreed, d["l2"], np.nan)
    if final is not None:
        f = final.reindex(d.index)
        fill = ~agreed & f.isin(["0", "1"])
        d.loc[fill, "final"] = f[fill].astype(int)

    groups = [("all", "all", d)]
    groups += [("pair_type", k, g) for k, g in d.groupby("pair_type")]
    groups += [("method", k, g) for k, g in d.groupby("method")]
    rows = []
    for level, name, g in groups:
        base = {"level": level, "group": name, "n": len(g)}
        for metric, fn in (("kappa_before_adjudication", cohen_kappa), ("agreement_before_adjudication", agreement_rate),
                           ("pabak_before_adjudication", pabak)):
            lo, hi = bootstrap_ci(fn, g["l2"], g["l3"], seed) if len(g) > 1 else (np.nan, np.nan)
            rows.append({**base, "metric": metric, "value": fn(g["l2"], g["l3"]), "ci_low": lo, "ci_high": hi})
        rows.append({**base, "metric": "share_label1_nguoi2", "value": g["l2"].mean(), "ci_low": np.nan,
                     "ci_high": np.nan})
        rows.append({**base, "metric": "share_label1_nguoi3", "value": g["l3"].mean(), "ci_low": np.nan,
                     "ci_high": np.nan})
        done = g["final"].notna()
        rows.append({**base, "metric": "pairs_to_adjudicate", "value": int((g["l2"] != g["l3"]).sum()),
                     "ci_low": np.nan, "ci_high": np.nan})
        if done.all():
            k = int(g["final"].sum())
            lo, hi = wilson(k, len(g))
            rows.append({**base, "metric": "accuracy_after_adjudication", "value": k / len(g), "ci_low": lo,
                         "ci_high": hi})
    out = pd.DataFrame(rows)
    out["value"] = out["value"].astype(float).round(4)
    out[["ci_low", "ci_high"]] = out[["ci_low", "ci_high"]].astype(float).round(4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    seed = int(load_config(REPO / args.config)["data_seed"])
    key = pd.read_csv(LABELS / "a4_mapping_key.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    l2, l3 = (read_labels(LABELS / f"a4_mapping_{a}.csv") for a in ANNOTATORS)
    missing = {a: int((lab == "").sum()) for a, lab in zip(ANNOTATORS, (l2, l3))}
    if any(missing.values()):
        print(f"unlabelled pairs (skipped): {missing}")

    adj_path = LABELS / "a4_mapping_adjudication.csv"
    final = None
    if adj_path.is_file():
        adj = pd.read_csv(adj_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        final = adj.set_index("item")["label_final"].str.strip()
    table = audit(key, l2, l3, final, seed)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "a4_mapping_audit.csv", index=False)

    # the disagreements, for Người 1 to adjudicate (kept labels already written are preserved)
    sheet = pd.read_csv(LABELS / f"a4_mapping_{ANNOTATORS[0]}.csv", encoding="utf-8-sig", dtype=str,
                        keep_default_na=False).set_index("item")
    both = (l2 != "") & (l3 != "")
    diff = sheet.loc[both & (l2 != l3), ["skill", "source", "title", "problem", "answer"]].copy()
    diff["label_nguoi2"], diff["label_nguoi3"] = l2[diff.index], l3[diff.index]
    diff["note_nguoi2"] = sheet.loc[diff.index, "note"]
    diff["note_nguoi3"] = pd.read_csv(LABELS / f"a4_mapping_{ANNOTATORS[1]}.csv", encoding="utf-8-sig", dtype=str,
                                      keep_default_na=False).set_index("item").loc[diff.index, "note"]
    diff["label_final"] = final.reindex(diff.index).fillna("") if final is not None else ""
    diff["note_final"] = ""
    if final is not None and "note_final" in adj.columns:
        diff["note_final"] = adj.set_index("item")["note_final"].reindex(diff.index).fillna("")
    diff.reset_index().to_csv(adj_path, index=False, encoding="utf-8-sig")

    # once every disagreement is decided: the pairs whose final label is 0, i.e. the dictionary errors to fix
    decided = final.reindex(diff.index).isin(["0", "1"]).all() if final is not None else diff.empty
    if both.all() and decided:
        lab = l2.where(l2 == l3, final.reindex(l2.index) if final is not None else l2)
        err = key.set_index("item").loc[lab[lab == "0"].index,
                                        ["pair_type", "method", "skill", "canonical_tag", "khan_slug", "doc_id"]]
        err["title"], err["problem"] = sheet.loc[err.index, "title"], sheet.loc[err.index, "problem"]
        err["how_decided"] = ["adjudicated" if l2[i] != l3[i] else "both_0" for i in err.index]
        err["note_nguoi2"] = sheet.loc[err.index, "note"]
        err["note_nguoi3"] = pd.read_csv(LABELS / f"a4_mapping_{ANNOTATORS[1]}.csv", encoding="utf-8-sig", dtype=str,
                                         keep_default_na=False).set_index("item").loc[err.index, "note"]
        err.reset_index().to_csv(TABLES / "a4_mapping_errors.csv", index=False, encoding="utf-8-sig")
        print(f"{len(err)} pairs with final label 0 -> reports/tables/a4_mapping_errors.csv")

    overall = table[table["group"] == "all"].set_index("metric")
    print(table[table["level"] != "method"].to_string(index=False))
    kappa = overall.loc["kappa_before_adjudication", "value"]
    print(f"\nκ = {kappa}; {len(diff)} pairs to adjudicate in {adj_path.name}")
    if kappa < 0.6:
        print("κ < 0.6: record why and how the instructions were revised (card P1-05); do not relabel to raise it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
