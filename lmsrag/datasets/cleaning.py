"""Bookkeeping shared by the adapters: a step-by-step cleaning table and the kept-vs-dropped comparison."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

CLEANING_COLUMNS = ["step", "rule", "applied", "rows_before", "rows_after", "dropped", "pct_dropped",
                    "users_after", "events_after"]


@dataclass
class CleaningLog:
    """Records one line per cleaning rule: rows before/after, rows dropped, learners and distinct events left.

    ``event_cols`` names the columns that identify one logged event (for ASSISTments 2009 a question answered on
    several skills is one event on several rows).
    """
    event_cols: list[str]
    user_col: str = "user_id"
    rows: list[dict] = field(default_factory=list)

    def _count(self, df: pd.DataFrame) -> tuple[int, int]:
        users = int(df[self.user_col].nunique())
        events = int(df[self.event_cols].drop_duplicates().shape[0]) if len(df) else 0
        return users, events

    def start(self, df: pd.DataFrame, rule: str = "raw file") -> pd.DataFrame:
        users, events = self._count(df)
        self.rows.append({"step": 0, "rule": rule, "applied": True, "rows_before": len(df), "rows_after": len(df),
                          "dropped": 0, "pct_dropped": 0.0, "users_after": users, "events_after": events})
        return df

    def apply(self, before: pd.DataFrame, after: pd.DataFrame, rule: str) -> pd.DataFrame:
        users, events = self._count(after)
        dropped = len(before) - len(after)
        self.rows.append({"step": len([r for r in self.rows if r["applied"]]), "rule": rule, "applied": True,
                          "rows_before": len(before), "rows_after": len(after), "dropped": dropped,
                          "pct_dropped": round(100.0 * dropped / max(len(before), 1), 4),
                          "users_after": users, "events_after": events})
        return after

    def note(self, rule: str, count: int, base: int) -> None:
        """A count reported for context only (not a filter), e.g. duplicates measured on the raw file."""
        self.rows.append({"step": "info", "rule": rule, "applied": False, "rows_before": base, "rows_after": None,
                          "dropped": count, "pct_dropped": round(100.0 * count / max(base, 1), 4),
                          "users_after": None, "events_after": None})

    def table(self, dataset: str) -> pd.DataFrame:
        out = pd.DataFrame(self.rows, columns=CLEANING_COLUMNS)
        for col in ("rows_before", "rows_after", "dropped", "users_after", "events_after"):
            out[col] = out[col].astype("Int64")
        out.insert(0, "dataset", dataset)
        return out


def skill_counts(df: pd.DataFrame, dataset: str, skill_col: str = "source_skill", tag_col: str = "skill_tag") -> pd.DataFrame:
    """Rows, learners and accuracy per source skill, taken before the skill filter (used by P1-04 and P1-05)."""
    g = (df.groupby(skill_col, observed=True)
         .agg(rows=("correct", "size"), users=("user_id", "nunique"), accuracy=("correct", "mean"),
              skill_tag=(tag_col, "first"))
         .reset_index().rename(columns={skill_col: "source_skill"}).sort_values("rows", ascending=False))
    g["accuracy"] = g["accuracy"].round(4)
    g.insert(0, "dataset", dataset)
    return g


def cluster_bootstrap_diff(kept: pd.DataFrame, dropped: pd.DataFrame, n_resamples: int, seed: int,
                           user_col: str = "user_id") -> tuple[float, float, float]:
    """Accuracy(kept) - accuracy(dropped) with a 95% percentile interval, resampling learners (clusters)."""
    users = np.union1d(kept[user_col].unique(), dropped[user_col].unique())
    index = pd.Index(users)

    def sums(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        g = part.groupby(user_col)["correct"].agg(["sum", "size"]).reindex(index, fill_value=0)
        return g["sum"].to_numpy(float), g["size"].to_numpy(float)

    ks, kn = sums(kept)
    ds, dn = sums(dropped)
    point = ks.sum() / kn.sum() - ds.sum() / dn.sum()
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for b in range(n_resamples):
        w = np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users)).astype(float)
        diffs[b] = (w @ ks) / max(w @ kn, 1.0) - (w @ ds) / max(w @ dn, 1.0)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def kept_vs_dropped(before: pd.DataFrame, keep_mask: pd.Series, n_resamples: int, seed: int,
                    skill_col: str = "source_skill", reason=None, min_rows_learner: int = 5) -> pd.DataFrame:
    """Compare rows kept by a filter with rows it removes: accuracy, bootstrap CI, chi-square, per-learner
    accuracy quantiles and, for the removed part, a per-skill breakdown that shows where any gap comes from."""
    kept, dropped = before[keep_mask], before[~keep_mask]
    rows: list[dict] = []

    def add(section, group, metric, value, lo=None, hi=None, skill=None, note=None):
        rows.append({"section": section, "group": group, "skill": skill, "metric": metric, "value": value,
                     "ci_low": lo, "ci_high": hi, "note": note})

    for name, part in (("kept", kept), ("dropped", dropped)):
        add("overall", name, "rows", len(part))
        add("overall", name, "users", int(part["user_id"].nunique()))
        add("overall", name, "accuracy", round(float(part["correct"].mean()), 4) if len(part) else None)
    if len(kept) and len(dropped):
        point, lo, hi = cluster_bootstrap_diff(kept, dropped, n_resamples, seed)
        add("overall", "kept-dropped", "accuracy_diff", round(point, 4), round(lo, 4), round(hi, 4),
            note=f"cluster bootstrap over learners, {n_resamples} resamples, seed {seed}")
        table = [[int(kept.correct.sum()), int((1 - kept.correct).sum())],
                 [int(dropped.correct.sum()), int((1 - dropped.correct).sum())]]
        chi2, p, _, _ = chi2_contingency(table)
        add("overall", "kept-dropped", "chi2", round(float(chi2), 3), note="row-level test; rows are not independent")
        add("overall", "kept-dropped", "chi2_p", float(p))

    for name, part in (("kept", kept), ("dropped", dropped)):
        per = part.groupby("user_id")["correct"].agg(["mean", "size"])
        per = per[per["size"] >= min_rows_learner]["mean"]
        add("per_learner", name, "learners", len(per), note=f"learners with >= {min_rows_learner} rows in this part")
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            add("per_learner", name, f"accuracy_q{int(q * 100):02d}", round(float(per.quantile(q)), 4) if len(per) else None)

    kept_acc = float(kept["correct"].mean()) if len(kept) else np.nan
    by_skill = dropped.groupby(skill_col)["correct"].agg(["size", "mean"]).sort_values("size", ascending=False)
    for skill, r in by_skill.iterrows():
        note = reason(skill) if reason else None
        add("dropped_by_skill", "dropped", "rows", int(r["size"]), skill=skill, note=note)
        add("dropped_by_skill", "dropped", "accuracy", round(float(r["mean"]), 4), skill=skill)
        add("dropped_by_skill", "dropped", "accuracy_minus_kept", round(float(r["mean"]) - kept_acc, 4), skill=skill)
    return pd.DataFrame(rows)
