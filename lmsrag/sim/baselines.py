"""Baseline selection policies for the closed-loop experiment (card P1-09).

Every policy has the interface of P2-08 (``lmsrag/sim/policies.py``, Người 2)::

    choose(observed_history, candidate_items) -> item_id

``observed_history`` is ``SimEnv.observe(u)`` (tuples ``(t, item_id, correct)``); ``candidate_items`` are the public
catalog rows of the skill the learner practises. ``reset(seed)``, when present, is called before each seed. A policy
never sees the environment.

All three baselines prefer the candidates the learner has seen the fewest times, so an item comes back only when every
candidate of the same rank has been given as often.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def _least_seen(order: np.ndarray, history) -> np.ndarray:
    seen = Counter(o.item_id for o in history)
    counts = np.array([seen.get(i, 0) for i in order])
    return order[counts == counts.min()]


class _Cached:
    """Per-skill arrays, built on the first call for that skill (all candidates of a call share one skill)."""

    def __init__(self):
        self._cache: dict[str, np.ndarray] = {}

    def _order(self, candidates: pd.DataFrame) -> np.ndarray:
        key = candidates["skill_tag"].iat[0]
        if key not in self._cache:
            self._cache[key] = self._build(candidates)
        return self._cache[key]

    def _build(self, candidates: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class RandomPolicy(_Cached):
    """A random candidate."""
    name = "random"

    def __init__(self, seed: int = 0):
        super().__init__()
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng([int(seed), 101])

    def _build(self, candidates):
        return candidates["item_id"].to_numpy()

    def choose(self, observed_history, candidate_items) -> str:
        pool = _least_seen(self._order(candidate_items), observed_history)
        return str(pool[self._rng.integers(len(pool))])


class FixedLevelPolicy(RandomPolicy):
    """Always difficulty level ``level`` (3): a random candidate at that level; a skill without one uses the nearest
    level present (the lower one on a tie)."""
    name = "fixed"

    def __init__(self, level: int = 3, seed: int = 0):
        self.level = int(level)
        super().__init__(seed)

    def _build(self, candidates):
        lv = candidates["difficulty_level"].to_numpy(dtype="int64")
        dist = np.abs(lv - self.level) * 2 + (lv > self.level)  # equal distance: lower level first
        return candidates["item_id"].to_numpy()[dist == dist.min()]


class MostPopularPolicy(_Cached):
    """The candidate with the most Junyi history responses (``popularity``: item_id -> (unit count, template count));
    then the next one, and so on. Ties go by item_id."""
    name = "most_popular"

    def __init__(self, popularity: dict[str, tuple[int, int]]):
        super().__init__()
        self.popularity = popularity

    def _build(self, candidates):
        ids = candidates["item_id"].tolist()
        return np.array(sorted(ids, key=lambda i: (*(-x for x in self.popularity.get(i, (0, 0))), i)))

    def choose(self, observed_history, candidate_items) -> str:
        return str(_least_seen(self._order(candidate_items), observed_history)[0])


def junyi_popularity(catalog: pd.DataFrame, junyi_history: pd.DataFrame, problem_types: pd.DataFrame) -> dict:
    """item_id -> (responses on its (template, problem type) unit, responses on its template), Junyi history split.

    Junyi writes a problem type either as its id or as its position in the template (``khan_problem_types.csv``), so a
    unit counts both spellings. Items without a Khan template (GSM8K) get (0, 0)."""
    h = junyi_history[junyi_history["split"] == "history"]
    unit = h["item_id"].astype(str).value_counts().to_dict()
    template = h["item_id"].astype(str).str.split(":", n=1).str[0].value_counts().to_dict()
    index_of = dict(zip(zip(problem_types["slug"], problem_types["problem_id"]), problem_types["problem_index"]))
    out = {}
    for item, slug, ptype in zip(catalog["item_id"], catalog["khan_slug"], catalog["problem_type"]):
        if not isinstance(slug, str) or not slug:
            out[item] = (0, 0)
            continue
        ptype = str(ptype)
        keys = {f"{slug}:{ptype}"}
        if (slug, ptype) in index_of:
            keys.add(f"{slug}:{index_of[(slug, ptype)]}")
        out[item] = (sum(unit.get(k, 0) for k in keys), template.get(slug, 0))
    return out
