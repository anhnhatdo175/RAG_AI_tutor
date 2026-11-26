"""Proposed selection policy for the closed loop (card P2-08, Người 2); interface of ``lmsrag.sim.baselines``.

``ProposedPolicy`` is the system's method reduced to what a policy may see (``observe(u)`` and the public catalog):

1. **Mastery with item difficulty** (P2-03 (i)+(ii) in Elo form): the learner's ability on the skill ``theta_hat``
   starts at ``theta0`` and after each answer moves by ``K_n (y - p_hat)``, ``K_n = max(k_min, k0 / (1 + n))``, so a
   right answer on a hard item moves it more than on an easy one and the step shrinks with practice.
2. **Item side**: the catalog has no IRT parameter, only the corpus ``difficulty`` in [0, 1], so
   ``b_hat_i = scale * logit(clip(difficulty_i, 0.02, 0.98))``.
3. **Probability-based ZPD** (P2-04): ``p_hat(u, i) = sigmoid(theta_hat - b_hat_i)``; the policy gives the candidate
   with ``p_hat`` nearest ``target`` (0.7, middle of the ZPD band [0.6, 0.8]), plus ``repeat_penalty`` per earlier
   showing of the item, so items come back only when nothing new is close; ties go by ``item_id``.

Parameters (``closed_loop.proposed_params`` in ``configs/sim.yaml``) are chosen on ``--world tune`` only
(``python -m lmsrag.sim.tune_proposed``); the policy never reads the true a, b, g, s, P or theta.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

DEFAULTS = {"theta0": 0.0, "k0": 0.8, "k_min": 0.15, "scale": 1.0, "target": 0.7, "repeat_penalty": 0.05}


class ProposedPolicy:
    name = "proposed"

    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown ProposedPolicy parameters: {sorted(unknown)}")
        self.p = {**DEFAULTS, **params}
        self._b: dict[str, dict[str, float]] = {}

    @classmethod
    def from_config(cls, cfg: dict, catalog: pd.DataFrame | None = None) -> "ProposedPolicy":
        return cls(**(cfg.get("closed_loop", {}).get("proposed_params") or {}))

    def reset(self, seed: int) -> None:  # deterministic: nothing random to reseed
        pass

    def _item_b(self, candidates: pd.DataFrame) -> dict[str, float]:
        key = candidates["skill_tag"].iat[0]
        if key not in self._b:
            d = np.clip(candidates["difficulty"].to_numpy(dtype="float64"), 0.02, 0.98)
            b = self.p["scale"] * np.log(d / (1 - d))
            self._b[key] = dict(zip(candidates["item_id"].astype(str), b))
        return self._b[key]

    def theta_hat(self, history, b: dict[str, float]) -> float:
        th = self.p["theta0"]
        for n, obs in enumerate(history):
            bi = b.get(str(obs.item_id))
            if bi is None:  # an item of another skill: not this skill's evidence
                continue
            p = 1.0 / (1.0 + math.exp(-(th - bi)))
            k = max(self.p["k_min"], self.p["k0"] / (1.0 + n))
            th += k * (float(obs.correct) - p)
        return th

    def choose(self, observed_history, candidate_items: pd.DataFrame) -> str:
        b = self._item_b(candidate_items)
        th = self.theta_hat(observed_history, b)
        seen: dict[str, int] = {}
        for o in observed_history:
            seen[str(o.item_id)] = seen.get(str(o.item_id), 0) + 1
        best = min(b, key=lambda i: (abs(1.0 / (1.0 + math.exp(-(th - b[i]))) - self.p["target"])
                                     + self.p["repeat_penalty"] * seen.get(i, 0), i))
        return best
