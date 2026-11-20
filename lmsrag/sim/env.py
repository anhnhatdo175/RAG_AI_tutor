"""Simulation environment (card P1-08): what a policy may call, and what only the metrics may call.

    env = SimEnv(world, rule="zpd", n_learners=1000, eta=(0.02, 0.08))
    env.reset(seed)                    # draws theta_{u,k} ~ N(mu_k, 1) and eta_u ~ U(eta)
    env.catalog                        # public item table (no a, b, g, s)
    env.observe(u)                     # the learner's answers so far, nothing else
    env.step(u, item_id) -> 0 or 1     # answer, then learn
    env.true_p(u, item_id)             # metrics only
    env.true_theta(u, skill_tag)       # metrics only

Random numbers: ``reset(seed)`` draws all learners from one generator, so the learners do not depend on the learning
rule; each learner then has its own answer stream (``SeedSequence(seed).spawn``), so a learner's answers depend only
on the seed and on the items it was given, not on the order in which a policy visits learners.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .model import RULES, World, learning_gain, p_correct


class Observation(NamedTuple):
    t: int            # 0-based turn of this learner
    item_id: str
    correct: int


class SimEnv:
    def __init__(self, world: World, rule: str, n_learners: int, eta: tuple[float, float] = (0.02, 0.08),
                 theta_sd: float | None = None, zpd: dict | None = None):
        if rule not in RULES:
            raise ValueError(f"unknown learning rule {rule!r}, expected one of {RULES}")
        self.rule = rule
        self.n_learners = int(n_learners)
        self._world = world
        self._eta = (float(eta[0]), float(eta[1]))
        self._theta_sd = float(world.params.get("theta_sd", 1.0) if theta_sd is None else theta_sd)
        z = zpd or world.params.get("zpd", {"center": 0.7, "width": 0.15})
        self._center, self._width = float(z["center"]), float(z["width"])
        items = world.items
        self._row = {item: i for i, item in enumerate(items["item_id"])}
        self._k = items["skill_idx"].to_numpy()
        self._a, self._b, self._g, self._s = (items[c].to_numpy(dtype="float64") for c in ("a", "b", "g", "s"))
        self._skill = dict(zip(world.skills.index, world.skills["skill_idx"].astype(int)))
        self.catalog = world.catalog()
        self.skills = list(world.skills.index)
        self._seed = None

    @classmethod
    def from_config(cls, world: World, cfg: dict, rule: str, n_learners: int | None = None) -> "SimEnv":
        return cls(world, rule, int(cfg["n_learners"] if n_learners is None else n_learners), tuple(cfg["eta"]),
                   float(cfg.get("theta_sd", 1.0)), cfg.get("zpd"))

    # ---- policy side -------------------------------------------------------------------------------------------
    def reset(self, seed: int) -> None:
        ss = np.random.SeedSequence(int(seed))
        draw, *streams = ss.spawn(self.n_learners + 1)
        rng = np.random.default_rng(draw)
        mu = self._world.skills["mu"].to_numpy(dtype="float64")
        self._theta = mu[None, :] + self._theta_sd * rng.standard_normal((self.n_learners, len(mu)))
        self._eta_u = rng.uniform(*self._eta, size=self.n_learners)
        self._rngs = [np.random.default_rng(s) for s in streams]
        self._history: list[list[Observation]] = [[] for _ in range(self.n_learners)]
        self._seed = int(seed)

    def observe(self, u: int) -> tuple[Observation, ...]:
        """The learner's past answers in order. No parameter of the learner or of the items is returned."""
        self._check(u)
        return tuple(self._history[u])

    def step(self, u: int, item_id: str) -> int:
        """Answer ``item_id`` (Bernoulli with the current P), then theta_{u,k} += eta_u * G(P)."""
        self._check(u)
        i = self._item(item_id)
        k = self._k[i]
        p = self._p(u, i)
        correct = int(self._rngs[u].random() < p)
        self._theta[u, k] += self._eta_u[u] * float(learning_gain(p, self.rule, self._center, self._width))
        self._history[u].append(Observation(len(self._history[u]), item_id, correct))
        return correct

    # ---- metrics side: never called by a policy ----------------------------------------------------------------
    def true_p(self, u: int, item_id: str) -> float:
        self._check(u)
        return self._p(u, self._item(item_id))

    def true_theta(self, u: int, skill_tag: str) -> float:
        self._check(u)
        return float(self._theta[u, self._skill[skill_tag]])

    # ---- internals ---------------------------------------------------------------------------------------------
    def _p(self, u: int, i: int) -> float:
        return float(p_correct(self._theta[u, self._k[i]], self._a[i], self._b[i], self._g[i], self._s[i]))

    def _item(self, item_id: str) -> int:
        try:
            return self._row[item_id]
        except KeyError:
            raise KeyError(f"unknown item {item_id!r}") from None

    def _check(self, u: int) -> None:
        if self._seed is None:
            raise RuntimeError("call reset(seed) first")
        if not 0 <= u < self.n_learners:
            raise IndexError(f"learner {u} out of range 0..{self.n_learners - 1}")
