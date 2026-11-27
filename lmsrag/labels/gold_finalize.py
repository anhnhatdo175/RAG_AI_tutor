"""Gold set agreement, adjudication and freezing (card P1-07, steps 4-6).

Sheets (from ``gold_pool``): ``a5_gold_pilot_<a>.csv`` (pilot queries) and ``a5_gold_<a>.csv`` (the rest), for the two
annotators in ``configs/data.yaml`` (``gold.annotators``: Người 1 then Người 3). Labels 0/1/2 per ``rubric_v1.md``.

1. Pilot: ``python -m lmsrag.labels.gold_finalize`` reports κ with quadratic weights. Under 0.6 the three meet, write
   ``rubric_v2.md`` with examples and relabel the pilot into ``a5_gold_pilot_v2_<a>.csv``; the first round stays in
   the report.
2. Full labelling, then the same command writes every disagreement to ``a5_gold_adjudication.csv`` for Người 2, who
   fills ``label_final`` without seeing which system found the chunk (``pool_provenance.csv`` is not on the sheet).
3. When every pair has a final label: ``gold_v1.csv`` (query_id, chunk_id, label_final, label_1, label_3,
   adjudicated), its SHA-256 in ``gold_v1.sha256`` and ``prereg/hypotheses.md``, and ``reports/tables/
   a5_gold_agreement.csv`` with the label distribution per source system. Then commit and tag ``gold-v1``; after the
   tag nobody edits ``gold_v1.csv`` (the command refuses to overwrite a different frozen file).

    python -m lmsrag.labels.gold_finalize
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.build import REPO, load_config
from .agreement import agreement_rate, bootstrap_ci, cohen_kappa

GOLD = REPO / "lmsrag" / "data" / "gold"
LABELS = REPO / "lmsrag" / "data" / "labels"
TABLES = REPO / "reports" / "tables"
PREREG = REPO / "prereg" / "hypotheses.md"
SCALE = [0, 1, 2]
KAPPA_MIN = 0.6


def read_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False).set_index("item")
    lab = df["label"].str.strip()
    bad = lab[~lab.isin(["0", "1", "2", ""])]
    if not bad.empty:
        raise SystemExit(f"{path.name}: labels must be 0, 1 or 2, got {dict(list(bad.items())[:5])}")
    return pd.DataFrame({"label": pd.to_numeric(lab, errors="coerce"), "note": df["note"]})


def wkappa(a, b) -> float:
    return cohen_kappa(a, b, "quadratic", SCALE)


def agreement_rows(stage: str, a: pd.Series, b: pd.Series, seed: int) -> list[dict]:
    both = a.notna() & b.notna()
    x, y = a[both].astype(int).to_numpy(), b[both].astype(int).to_numpy()
    lo, hi = bootstrap_ci(wkappa, x, y, seed) if len(x) > 2 else (np.nan, np.nan)
    rows = [{"stage": stage, "group": "all", "n": len(x), "metric": "weighted_kappa", "value": wkappa(x, y) if len(x) else np.nan,
             "ci_low": lo, "ci_high": hi},
            {"stage": stage, "group": "all", "n": len(x), "metric": "exact_agreement",
             "value": agreement_rate(x, y) if len(x) else np.nan},
            {"stage": stage, "group": "all", "n": len(x), "metric": "pairs_to_adjudicate", "value": int((x != y).sum())},
            {"stage": stage, "group": "all", "n": int((~both).sum()), "metric": "pairs_unlabelled", "value": int((~both).sum())}]
    for who, v in (("label_1", x), ("label_3", y)):
        for lab in SCALE:
            rows.append({"stage": stage, "group": who, "n": len(v), "metric": f"share_label_{lab}",
                         "value": float(np.mean(v == lab)) if len(v) else np.nan})
    return rows


def load_stage(names: list[str], annotators: list[str], labels_dir: Path) -> pd.DataFrame | None:
    paths = [labels_dir / n.format(a=a) for a in annotators for n in names]
    if not all(p.is_file() for p in paths):
        return None
    parts = []
    for n in names:
        l1, l3 = (read_labels(labels_dir / n.format(a=a)) for a in annotators)
        parts.append(pd.DataFrame({"label_1": l1["label"], "label_3": l3["label"], "note_1": l1["note"],
                                   "note_3": l3["note"]}))
    return pd.concat(parts)


def adjudication(final_stage: pd.DataFrame, items: pd.DataFrame, sheet: pd.DataFrame, old: pd.DataFrame | None):
    diff = final_stage[final_stage["label_1"].notna() & final_stage["label_3"].notna()
                       & (final_stage["label_1"] != final_stage["label_3"])]
    out = sheet.loc[diff.index, ["query_id", "skill", "target_level", "content"]].copy()
    out["label_1"], out["label_3"] = diff["label_1"].astype(int), diff["label_3"].astype(int)
    out["note_1"], out["note_3"] = diff["note_1"], diff["note_3"]
    out["label_final"], out["note_final"] = "", ""
    if old is not None and len(old):
        prev = old.set_index("item")
        out["label_final"] = prev["label_final"].reindex(out.index).fillna("").astype(str)
        out["note_final"] = prev["note_final"].reindex(out.index).fillna("").astype(str)
    return out.reset_index().rename(columns={"index": "item"})


def finalize(final_stage: pd.DataFrame, items: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame | None:
    """gold_v1 rows, or None while a pair is unlabelled or an adjudication is missing."""
    if final_stage[["label_1", "label_3"]].isna().any().any():
        return None
    decided = adj.set_index("item")["label_final"].astype(str).str.strip()
    if not decided.isin(["0", "1", "2"]).all():
        return None
    d = final_stage.join(items.set_index("item")[["query_id", "chunk_id"]])
    d["adjudicated"] = d["label_1"] != d["label_3"]
    d["label_final"] = d["label_1"].where(~d["adjudicated"], decided.reindex(d.index).astype(float))
    out = d[["query_id", "chunk_id", "label_final", "label_1", "label_3", "adjudicated"]].copy()
    for c in ("label_final", "label_1", "label_3"):
        out[c] = out[c].astype(int)
    return out.sort_values(["query_id", "chunk_id"]).reset_index(drop=True)


def by_system(gold: pd.DataFrame, prov: pd.DataFrame, systems: list[str], top_k: int) -> list[dict]:
    """Final label distribution of the chunks each system put in its top-k, and of the random checks."""
    m = gold.merge(prov, on=["query_id", "chunk_id"])
    groups = {s: m[m[f"rank_{s}"].notna() & (m[f"rank_{s}"] <= top_k)] for s in systems}
    for origin in ("random_same_skill", "random_other_skill"):
        groups[origin] = m[m["origins"].str.contains(origin)]
    rows = []
    for name, g in groups.items():
        for lab in SCALE:
            rows.append({"stage": "final", "group": name, "n": len(g), "metric": f"share_label_{lab}",
                         "value": float((g["label_final"] == lab).mean()) if len(g) else np.nan})
    return rows


def freeze(gold: pd.DataFrame, gold_dir: Path, prereg: Path) -> str:
    """No default paths on purpose: a default would be bound at import time and a dry run could freeze into the repo."""
    path = gold_dir / "gold_v1.csv"
    data = gold.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if path.is_file() and path.read_bytes() != data:
        raise SystemExit("gold_v1.csv is frozen and differs from the labels now; do not edit it after the gold-v1 tag")
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    # LF on every OS: sha256sum -c reads a CRLF line as a file name ending in \r
    (gold_dir / "gold_v1.sha256").write_text(f"{digest}  gold_v1.csv\n", encoding="utf-8", newline="\n")
    prereg.parent.mkdir(parents=True, exist_ok=True)
    text = prereg.read_text(encoding="utf-8") if prereg.is_file() else "# Giả thuyết và tiêu chí kết luận\n"
    line = f"- Gold set đóng băng: `lmsrag/data/gold/gold_v1.csv`, SHA-256 `{digest}` (tag `gold-v1`)."
    if digest not in text:
        text = text.rstrip("\n") + ("\n\n## Gold set\n\n" if "## Gold set" not in text else "\n") + line + "\n"
        prereg.write_text(text, encoding="utf-8")
    return digest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_config(REPO / args.config)
    g, seed = cfg["gold"], int(cfg["data_seed"])
    annotators = list(g["annotators"])
    items = pd.read_csv(GOLD / "items_v1.csv", dtype={"item": str})
    rows = []

    pilot = load_stage(["a5_gold_pilot_{a}.csv"], annotators, LABELS)
    if pilot is None:
        raise SystemExit("pilot sheets not found; run python -m lmsrag.labels.gold_pool first")
    rows += agreement_rows("pilot_v1", pilot["label_1"], pilot["label_3"], seed)
    pilot_v2 = load_stage(["a5_gold_pilot_v2_{a}.csv"], annotators, LABELS)
    if pilot_v2 is not None:
        rows += agreement_rows("pilot_v2", pilot_v2["label_1"], pilot_v2["label_3"], seed)
    full = load_stage(["a5_gold_{a}.csv"], annotators, LABELS)
    if full is not None:
        rows += agreement_rows("full", full["label_1"], full["label_3"], seed)

    k_pilot = [r["value"] for r in rows if r["metric"] == "weighted_kappa" and r["stage"].startswith("pilot")][-1]
    print(f"pilot κ (quadratic): {k_pilot:.3f}" + (" < 0.6: meet, write rubric_v2.md, relabel the pilot into "
                                                  "a5_gold_pilot_v2_<annotator>.csv" if k_pilot < KAPPA_MIN else ""))
    out = pd.DataFrame(rows)
    if full is not None:
        final_stage = pd.concat([pilot_v2 if pilot_v2 is not None else pilot, full])
        sheet = pd.concat([pd.read_csv(LABELS / f"a5_gold_pilot_{annotators[0]}.csv", encoding="utf-8-sig", dtype=str,
                                       keep_default_na=False),
                           pd.read_csv(LABELS / f"a5_gold_{annotators[0]}.csv", encoding="utf-8-sig", dtype=str,
                                       keep_default_na=False)]).set_index("item")
        adj_path = LABELS / "a5_gold_adjudication.csv"
        old = pd.read_csv(adj_path, encoding="utf-8-sig", dtype=str, keep_default_na=False) if adj_path.is_file() else None
        adj = adjudication(final_stage, items, sheet, old)
        adj.to_csv(adj_path, index=False, encoding="utf-8-sig")
        print(f"{len(adj)} pairs to adjudicate in {adj_path.name} (Người 2)")
        gold = finalize(final_stage, items, adj)
        if gold is not None:
            out = pd.concat([out, pd.DataFrame(
                [{"stage": "final", "group": "all", "n": len(gold), "metric": "share_adjudicated",
                  "value": float(gold["adjudicated"].mean())}]
                + [{"stage": "final", "group": "all", "n": len(gold), "metric": f"share_label_{lab}",
                    "value": float((gold["label_final"] == lab).mean())} for lab in SCALE]
                + by_system(gold, pd.read_csv(GOLD / "pool_provenance.csv"), list(g["systems"]), int(g["top_k"])))])
            digest = freeze(gold, GOLD, PREREG)
            print(f"gold_v1.csv: {len(gold)} pairs, {gold['query_id'].nunique()} queries, SHA-256 {digest}")
            print("  next: commit, then  git tag gold-v1  (and push the tag)")
    for c in ("value", "ci_low", "ci_high"):
        if c in out:
            out[c] = out[c].astype(float).round(4)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "a5_gold_agreement.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
