"""
LAYER 2 / stage 6 - LEARNER PROFILE VECTOR (docx 2.3.3).

  p_{u,k} = sum_{t=1..T} w(t) * v_interaction(t),   p_{u,k} <- p_{u,k} / ||p_{u,k}||_2
  w(t)    = exp(-lambda (T - t)) * [ gamma * c(t) + (1 - gamma) * (tau_base / tau(t)) ]

Two equivalent implementations are kept and unit-tested against each other:
  * ``profile_vector_direct``  - literal formula, O(T) per call (reference)
  * ``DecayedProfile``         - O(1) incremental update used online: S <- exp(-lambda) * S + w_new * v_new
Deviation from the paper formula (documented): the speed term tau_base/tau(t) is clipped to [0, 1], otherwise one lucky
2-second answer would outweigh a whole history.
v_interaction(t) is the vector of what the learner interacted with.  ASSISTments has no item text, so it is the centroid of
the corpus chunks sharing the item's (skill_tag, difficulty bucket) - see ``InteractionVectorizer``.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .. import config


def weight(c: float, tau: float, tau_base: float, age: float, lam: float, gamma: float) -> float:
    speed = min(1.0, tau_base / max(tau, 1.0))
    return math.exp(-lam * age) * (gamma * c + (1 - gamma) * speed)


def profile_vector_direct(vs: np.ndarray, c: np.ndarray, tau: np.ndarray, tau_base: float,
                          half_life: float = config.HALF_LIFE_OPPS, gamma: float = config.GAMMA) -> np.ndarray:
    """vs (T,d) interaction vectors in chronological order; c correctness; tau response time (ms)."""
    lam = math.log(2) / half_life
    T = len(vs)
    w = np.array([weight(c[t], tau[t], tau_base, T - 1 - t, lam, gamma) for t in range(T)])
    p = (w[:, None] * vs).sum(0)
    return p / max(np.linalg.norm(p), 1e-12)


class DecayedProfile:
    """Incremental form; state is a single d-vector."""

    def __init__(self, dim: int, half_life: float = config.HALF_LIFE_OPPS, gamma: float = config.GAMMA):
        self.S = np.zeros(dim)
        self.decay = math.exp(-math.log(2) / half_life)
        self.gamma = gamma

    def update(self, v: np.ndarray, c: float, tau: float, tau_base: float):
        speed = min(1.0, tau_base / max(tau, 1.0))
        self.S = self.decay * self.S + (self.gamma * c + (1 - self.gamma) * speed) * v

    @property
    def vector(self) -> np.ndarray:
        n = np.linalg.norm(self.S)
        return self.S / n if n > 1e-12 else self.S


class InteractionVectorizer:
    """v_interaction for items without text: centroid of chunks with the same (skill_tag, difficulty bucket)."""

    def __init__(self, vectors: np.ndarray, meta: list[dict]):
        groups = defaultdict(list)
        by_tag = defaultdict(list)
        for i, m in enumerate(meta):
            b = int(np.searchsorted(config.DIFFICULTY_LEVELS, m["difficulty"] - 1e-9))
            groups[(m["skill_tag"], b)].append(i)
            by_tag[m["skill_tag"]].append(i)
        self.cent = {}
        for key, ix in groups.items():
            c = vectors[ix].mean(0)
            self.cent[key] = c / np.linalg.norm(c)
        self.tag_cent = {t: (lambda c: c / np.linalg.norm(c))(vectors[ix].mean(0)) for t, ix in by_tag.items()}

    def __call__(self, skill_tag: str, bucket: int) -> np.ndarray | None:
        """Falls back to the nearest available bucket of the same skill, then to the skill centroid."""
        for b in sorted(range(3), key=lambda x: abs(x - bucket)):
            if (skill_tag, b) in self.cent:
                return self.cent[(skill_tag, b)]
        return self.tag_cent.get(skill_tag)
