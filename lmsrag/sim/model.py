"""Simulated learners (card P1-08, plan item A6): the item response model, the learning rules and the calibration.

Probability that learner ``u`` answers item ``i`` of skill ``k`` correctly::

    P(u, i) = g_i + (1 - g_i - s_i) * sigmoid(a_i * (theta_{u,k} - b_i))

    b_i      = logit(1 - p_correct_hist_i)  (Junyi history accuracy of the item, P1-04), else mapped from difficulty
    a_i      ~ LogNormal(0, 0.3)
    g_i      = 0.20 for multiple choice (answer_type radio), 0.02 for typed answers; s_i = 0.05
    theta    ~ Normal(mu_k, 1), mu_k such that the initial accuracy on k equals the Junyi history accuracy on k

After each answer ``theta_{u,k} <- theta_{u,k} + eta_u * G(P)`` with ``P`` the probability before the answer and one
of three learning rules ``G`` (``flat``, ``info``, ``zpd``). The world (items and their true parameters) is fixed by
``world_seed``; the learners are drawn by ``SimEnv.reset(seed)`` in ``env.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

RULES = ("flat", "info", "zpd")
# columns of the exercise bank a policy may see; a_i, b_i, g_i and s_i are hidden
PUBLIC = ["item_id", "skill_tag", "source", "difficulty", "difficulty_level", "difficulty_tier", "difficulty_reliable",
          "answer_type", "khan_slug", "problem_type"]
N_QUAD = 64        # Gauss-Hermite nodes for E_theta[P]
MU_RANGE = (-10.0, 10.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.asarray(p, dtype="float64")
    return np.log(p) - np.log1p(-p)


def p_correct(theta, a, b, g, s):
    """The response model above; broadcasts over all arguments."""
    return g + (1.0 - g - s) * sigmoid(a * (theta - b))


def learning_gain(p, rule: str, center: float = 0.7, width: float = 0.15):
    """G(P): ``flat`` 1, ``info`` 4 P (1 - P) (peak 1 at P = 0.5), ``zpd`` exp(-(P - center)^2 / (2 width^2))."""
    p = np.asarray(p, dtype="float64")
    if rule == "flat":
        return np.ones_like(p)
    if rule == "info":
        return 4.0 * p * (1.0 - p)
    if rule == "zpd":
        return np.exp(-((p - center) ** 2) / (2.0 * width ** 2))
    raise ValueError(f"unknown learning rule {rule!r}, expected one of {RULES}")


def item_difficulty(items: pd.DataFrame, p_clip: tuple[float, float] = (0.01, 0.99)) -> np.ndarray:
    """b_i. Items with ``p_correct_hist`` use logit(1 - p). The others (Khan from hint steps, GSM8K) are placed by a
    quantile map fitted on the items that have one: the sorted ``difficulty`` values (the within-source mid-rank of
    P1-04) against the sorted b, linearly interpolated, clamped at both ends. The map is monotone, so ``difficulty``
    orders b the same way in every source; for the Junyi items ``difficulty`` is the rank of 1 - p, so the map
    returns their own b."""
    p = pd.to_numeric(items["p_correct_hist"], errors="coerce").to_numpy(dtype="float64")
    has = ~np.isnan(p)
    b = np.full(len(items), np.nan)
    b[has] = logit(1.0 - np.clip(p[has], *p_clip))
    if (~has).any():
        if has.sum() < 2:
            raise ValueError("need at least two items with p_correct_hist to map difficulty to b")
        d = items["difficulty"].to_numpy(dtype="float64")
        b[~has] = np.interp(d[~has], np.sort(d[has]), np.sort(b[has]))
    return b


def junyi_targets(junyi: pd.DataFrame, min_responses: int) -> pd.DataFrame:
    """Per skill: accuracy and count of the Junyi **history** responses; skills under ``min_responses`` are dropped."""
    h = junyi[junyi["split"] == "history"]
    t = h.groupby("skill_tag")["correct"].agg(junyi_rate="mean", junyi_n="size")
    return t[t["junyi_n"] >= min_responses]


def expected_rate(mu, theta_sd, a, b, g, s, skill_idx, n_skills, nodes=None):
    """Initial accuracy per skill: mean over its items of E_{theta ~ N(mu_k, sd)}[P], by Gauss-Hermite quadrature."""
    z, w = nodes if nodes is not None else np.polynomial.hermite_e.hermegauss(N_QUAD)
    w = w / w.sum()
    theta = mu[skill_idx][:, None] + theta_sd * z[None, :]
    ep = (p_correct(theta, a[:, None], b[:, None], g[:, None], s[:, None]) * w).sum(axis=1)
    return np.bincount(skill_idx, weights=ep, minlength=n_skills) / np.bincount(skill_idx, minlength=n_skills)


def calibrate_mu(targets, theta_sd, a, b, g, s, skill_idx, iters: int = 80):
    """mu_k by bisection (the rate rises with mu). A target outside what the model reaches on [-10, 10] (above about
    1 - s, or below the mean guess) gets the bound; the caller reports the remaining gap."""
    targets = np.asarray(targets, dtype="float64")
    k = len(targets)
    nodes = np.polynomial.hermite_e.hermegauss(N_QUAD)
    lo, hi = np.full(k, MU_RANGE[0]), np.full(k, MU_RANGE[1])
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        up = expected_rate(mid, theta_sd, a, b, g, s, skill_idx, k, nodes) < targets
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    mu = (lo + hi) / 2.0
    return mu, expected_rate(mu, theta_sd, a, b, g, s, skill_idx, k, nodes)


@dataclass(frozen=True)
class World:
    """Items with their true parameters, the skills, and mu_k. Policies get ``catalog()`` only."""
    items: pd.DataFrame          # PUBLIC columns + skill_idx, a, b, g, s; row i is item i
    skills: pd.DataFrame         # index skill_tag; skill_idx, n_items, junyi_rate, junyi_n, mu, model_rate
    params: Mapping
    seed: int

    def catalog(self) -> pd.DataFrame:
        return self.items[[c for c in PUBLIC if c in self.items.columns]].copy()


def build_world(bank: pd.DataFrame, targets: pd.DataFrame, params: Mapping, seed: int) -> World:
    """Keep the exercises whose ``skill_tag`` has a Junyi target, draw a_i with ``seed``, fit b_i, g_i, s_i and mu_k."""
    items = bank.rename(columns={"chunk_id": "item_id"}) if "item_id" not in bank.columns else bank.copy()
    # b is mapped from difficulty on the whole bank, before the skill filter, so the map does not depend on it
    items = items.assign(b=item_difficulty(items, tuple(params.get("p_clip", (0.01, 0.99)))))
    items = items[items["skill_tag"].isin(targets.index)].sort_values("item_id", kind="mergesort").reset_index(drop=True)
    if items.empty:
        raise ValueError("no exercise has a skill with a Junyi target")
    skills = targets.loc[sorted(items["skill_tag"].unique())].copy()
    skills["skill_idx"] = np.arange(len(skills))
    skills["n_items"] = items.groupby("skill_tag").size().reindex(skills.index).to_numpy()
    rng = np.random.default_rng(seed)
    guess = params.get("guess", {"radio": 0.20, "default": 0.02})
    items["skill_idx"] = skills.loc[items["skill_tag"], "skill_idx"].to_numpy()
    items["a"] = rng.lognormal(0.0, float(params.get("discrimination_log_sd", 0.3)), size=len(items))
    items["g"] = np.where(items["answer_type"] == "radio", float(guess["radio"]), float(guess["default"]))
    items["s"] = float(params.get("slip", 0.05))
    cols = [items[c].to_numpy(dtype="float64") for c in ("a", "b", "g", "s")]
    mu, rate = calibrate_mu(skills["junyi_rate"].to_numpy(), float(params.get("theta_sd", 1.0)), *cols,
                            items["skill_idx"].to_numpy())
    skills["mu"], skills["model_rate"] = mu, rate
    return World(items=items, skills=skills, params=dict(params), seed=int(seed))


def load_world(cfg: Mapping, repo: Path, seed: int | None = None) -> World:
    """World from the files in ``configs/sim.yaml``: the exercise bank and the Junyi history accuracy per skill."""
    bank = pd.read_json(repo / cfg["items"], lines=True)
    junyi = pd.read_parquet(repo / cfg["junyi"], columns=["skill_tag", "correct", "split"])
    targets = junyi_targets(junyi, int(cfg.get("min_junyi_responses", 200)))
    return build_world(bank, targets, cfg, int(cfg["world_seed"] if seed is None else seed))
