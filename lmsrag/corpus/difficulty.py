"""Continuous difficulty for corpus v2 (plan card P1-04, teacher's item A3).

Raw difficulty per source, then a percentile rank inside each source so all sources share the scale [0, 1]:

* Khan with Junyi data: 1 - accuracy of the (template, problem type) unit on the Junyi **history** split, shrunk
  toward the template's accuracy with weight ``m``; needs ``min_n`` responses, else the template's accuracy, else none.
* Khan without data: isotonic regression from the number of hint steps to that raw difficulty, fitted on the Khan
  problems that have data.
* GSM8K: number of calculator annotations ``<<...>>`` in the reference solution (0-9 operations).
* MATH: the dataset's Level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def junyi_history_counts(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Responses and correct answers per (khan_slug, problem_type key) and per khan_slug, history split only."""
    h = interactions[(interactions["split"] == "history") & interactions["khan_slug"].notna()]
    h = h.assign(type_key=h["item_id"].str.split(":", n=1).str[1])
    unit = h.groupby(["khan_slug", "type_key"], observed=True)["correct"].agg(n="size", k="sum").reset_index()
    template = h.groupby("khan_slug", observed=True)["correct"].agg(n="size", k="sum").reset_index()
    return unit, template


def khan_history_difficulty(problems: pd.DataFrame, unit: pd.DataFrame, template: pd.DataFrame,
                            m: float, min_n: int) -> pd.DataFrame:
    """Columns p_correct_hist, difficulty_n, difficulty_unit for each problem (keys: slug, type_key)."""
    t = template.set_index("khan_slug")
    u = unit.set_index(["khan_slug", "type_key"])
    rows = []
    for slug, key in zip(problems["slug"], problems["type_key"]):
        n_t, k_t = (int(t.at[slug, "n"]), int(t.at[slug, "k"])) if slug in t.index else (0, 0)
        n_u, k_u = (int(u.at[(slug, key), "n"]), int(u.at[(slug, key), "k"])) if (slug, key) in u.index else (0, 0)
        if n_u >= min_n and n_t > 0:
            p_t = k_t / n_t
            rows.append(((k_u + m * p_t) / (n_u + m), n_u, "problem_type"))
        elif n_t >= min_n:
            rows.append((k_t / n_t, n_t, "template"))
        else:
            rows.append((np.nan, 0, None))
    return pd.DataFrame(rows, columns=["p_correct_hist", "difficulty_n", "difficulty_unit"], index=problems.index)


def fit_steps_model(steps: np.ndarray, raw_difficulty: np.ndarray) -> IsotonicRegression:
    """Monotone map from number of hint steps to raw difficulty (1 - p_correct_hist)."""
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(np.asarray(steps, float), np.asarray(raw_difficulty, float))
    return model


def gsm8k_operations(raw_answer: str) -> int:
    """Calculator annotations ``<<48/2=24>>`` in the original GSM8K solution = number of arithmetic steps."""
    return raw_answer.count("<<")


def percentile_within(values: pd.Series, groups: pd.Series) -> pd.Series:
    """Mid-rank percentile inside each group: (average rank - 0.5) / n, so ties share a value and the scale is (0, 1)."""
    rank = values.groupby(groups).rank(method="average")
    n = values.groupby(groups).transform("size")
    return (rank - 0.5) / n


def quintile_levels(values: pd.Series) -> pd.Series:
    """Level 1-5 by the quintiles of ``values`` over the whole corpus (ties at a cut go to the lower level)."""
    cuts = np.quantile(values.to_numpy(float), [0.2, 0.4, 0.6, 0.8])
    return pd.Series(1 + np.searchsorted(cuts, values.to_numpy(float), side="left"), index=values.index).astype(int)
