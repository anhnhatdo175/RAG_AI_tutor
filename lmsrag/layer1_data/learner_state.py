"""
LAYER 1 / learner state (docx 2.2.3):   P(u) = (M_u, A_u, sigma_u, W_u, ST_u, pace_u).

The outline names the six components; W_u and ST_u are the weak / strong topic lists of docx 1.2.1, the others are defined here:
  M_u     mastery vector: skill_id -> EMA of first-try score (prior = skill's history mean for an unseen skill)
  A_u     overall mean score ("average accuracy") over all attempts
  sigma_u std of the last 30 scores (performance consistency; 0 until 5 attempts)
  W_u     weak-skill list: skills with >= 5 attempts and mastery < 0.5, weakest first
  ST_u    strong-topic list (docx 1.2.1): skills with >= 5 attempts and mastery >= 0.8, strongest first
          (the mirror image of W_u; the thresholds STRONG_THETA / STRONG_MIN_ATTEMPTS are configurable constants)
  study_time  (extra, not part of P(u) in the docx) mean response time (s, clipped to [0.5,120]) and total active time (h)
  pace_u  learning pace: OLS slope of score vs attempt index x100, i.e. score gain per 100 attempts (0.1 = +10 pts; 0 until 30 attempts)
Every field is updated in O(1) per attempt, so the offline build and the online loop use the same code path.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

import pandas as pd

WEAK_THETA, WEAK_MIN_ATTEMPTS, RECENT_N = 0.5, 5, 30
STRONG_THETA, STRONG_MIN_ATTEMPTS = 0.8, 5


@dataclass
class LearnerState:
    user_id: int
    M: dict = field(default_factory=dict)          # skill_id(str) -> mastery
    attempts: dict = field(default_factory=dict)   # skill_id(str) -> n attempts
    canonical_M: dict = field(default_factory=dict)       # canonical_skill_tag -> mastery
    canonical_attempts: dict = field(default_factory=dict)
    n: int = 0
    sum_score: float = 0.0
    recent: list = field(default_factory=list)
    rt_sum_ms: float = 0.0
    sx: float = 0.0; sy: float = 0.0; sxy: float = 0.0; sxx: float = 0.0     # running sums for the pace regression
    prof: dict = field(default_factory=dict)       # skill_id(str) -> [W_easy, W_mid, W_hard]: decayed profile weights (Layer 2)
    canonical_prof: dict = field(default_factory=dict)    # canonical_skill_tag -> [W_easy, W_mid, W_hard]
    seen: list = field(default_factory=list)       # task ids already served/answered (Layer 3/4 exclude them)

    def update(self, skill_id: int, score: float, rt_ms: float, alpha: float, prior: float, bucket: int | None = None,
               tau_base: float | None = None, gamma: float = 0.7, decay: float = 0.9170040432046712,
               task_id: str | None = None) -> None:
        k = str(int(skill_id))
        if bucket is not None:           # Layer-2 profile vector weights: W <- decay*W + [gamma*c + (1-gamma)*min(1,tau_base/tau)] e_bucket
            w = [x * decay for x in self.prof.get(k, [0.0, 0.0, 0.0])]
            w[int(bucket)] += gamma * score + (1 - gamma) * min(1.0, (tau_base or rt_ms) / max(rt_ms, 1.0))
            self.prof[k] = w
        if task_id is not None:
            self.seen = (self.seen + [task_id])[-500:]
        prev = self.M.get(k, prior)
        self.M[k] = alpha * score + (1 - alpha) * prev
        self.attempts[k] = self.attempts.get(k, 0) + 1
        self._record_observation(score, rt_ms)

    def update_canonical(self, skill_tag: str, score: float, rt_ms: float, alpha: float,
                         prior: float = 0.5, bucket: int | None = None,
                         tau_base: float | None = None, gamma: float = 0.7,
                         decay: float = 0.9170040432046712,
                         task_id: str | None = None) -> None:
        """Update real-time mastery/profile directly at the canonical exercise skill."""
        if not skill_tag or not isinstance(skill_tag, str):
            raise ValueError("skill_tag must be a non-empty string")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score must be finite and in [0, 1]")
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be finite and in [0, 1]")
        if not math.isfinite(prior) or not 0.0 <= prior <= 1.0:
            raise ValueError("prior must be finite and in [0, 1]")
        if not math.isfinite(rt_ms) or rt_ms < 0:
            raise ValueError("rt_ms must be finite and non-negative")
        if bucket is not None and bucket not in (0, 1, 2):
            raise ValueError("bucket must be 0, 1, or 2")
        if (not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0
                or not math.isfinite(decay) or not 0.0 <= decay <= 1.0):
            raise ValueError("gamma and decay must be in [0, 1]")
        if tau_base is not None and (not math.isfinite(tau_base) or tau_base < 0):
            raise ValueError("tau_base must be finite and non-negative")

        previous = self.canonical_M.get(skill_tag, prior)
        self.canonical_M[skill_tag] = alpha * score + (1.0 - alpha) * previous
        self.canonical_attempts[skill_tag] = self.canonical_attempts.get(skill_tag, 0) + 1
        if bucket is not None:
            weights = [value * decay for value in self.canonical_prof.get(skill_tag, [0.0, 0.0, 0.0])]
            speed = (
                min(1.0, tau_base / max(rt_ms, 1.0))
                if tau_base is not None
                else 0.5
            )
            weights[bucket] += gamma * score + (1.0 - gamma) * speed
            self.canonical_prof[skill_tag] = weights
        if task_id is not None:
            self.seen = (self.seen + [task_id])[-500:]
        self._record_observation(score, rt_ms)

    def _record_observation(self, score: float, rt_ms: float) -> None:
        x = self.n
        self.n += 1
        self.sum_score += score
        self.recent = (self.recent + [score])[-RECENT_N:]
        self.rt_sum_ms += min(max(rt_ms, 500.0), 120_000.0)
        self.sx += x; self.sy += score; self.sxy += x * score; self.sxx += x * x

    # ---- derived components -------------------------------------------------------------
    @property
    def A(self) -> float:
        return self.sum_score / self.n if self.n else 0.0

    @property
    def sigma(self) -> float:
        return float(np.std(self.recent)) if len(self.recent) >= 5 else 0.0

    @property
    def W(self) -> list:
        w = [(int(k), v) for k, v in self.M.items() if self.attempts.get(k, 0) >= WEAK_MIN_ATTEMPTS and v < WEAK_THETA]
        return sorted(w, key=lambda t: t[1])

    @property
    def ST(self) -> list:
        """Strong topics (docx 1.2.1): [(skill_id, mastery)], strongest first."""
        st = [(int(k), v) for k, v in self.M.items() if self.attempts.get(k, 0) >= STRONG_MIN_ATTEMPTS and v >= STRONG_THETA]
        return sorted(st, key=lambda t: -t[1])

    @property
    def canonical_W(self) -> list:
        weak = [(tag, value) for tag, value in self.canonical_M.items()
                if self.canonical_attempts.get(tag, 0) >= WEAK_MIN_ATTEMPTS and value < WEAK_THETA]
        return sorted(weak, key=lambda item: item[1])

    @property
    def canonical_ST(self) -> list:
        strong = [(tag, value) for tag, value in self.canonical_M.items()
                  if self.canonical_attempts.get(tag, 0) >= STRONG_MIN_ATTEMPTS and value >= STRONG_THETA]
        return sorted(strong, key=lambda item: -item[1])

    @property
    def study_time(self) -> dict:
        return {"mean_rt_s": self.rt_sum_ms / self.n / 1000 if self.n else 0.0, "total_time_h": self.rt_sum_ms / 3.6e6}

    @property
    def pace(self) -> float:
        if self.n < 30:
            return 0.0
        den = self.n * self.sxx - self.sx ** 2
        return 100.0 * (self.n * self.sxy - self.sx * self.sy) / den if den > 0 else 0.0

    def summary(self) -> dict:
        return {"user_id": self.user_id, "n": self.n, "A": self.A, "sigma": self.sigma,
                "n_weak": len(self.W), "n_strong": len(self.ST),
                "n_canonical_weak": len(self.canonical_W), "n_canonical_strong": len(self.canonical_ST),
                "mean_rt_s": self.study_time["mean_rt_s"], "pace_per_100": self.pace,
                "n_skills": len(self.M), "n_canonical_skills": len(self.canonical_M)}

    def context_snapshot(self) -> dict:
        """Return a compact, JSON-safe learner profile for downstream generation context."""
        return {
            "overall_accuracy": self.A,
            "consistency_sigma": self.sigma,
            "pace_per_100": self.pace,
            "mean_response_time_seconds": self.study_time["mean_rt_s"],
            "canonical_mastery": dict(self.canonical_M),
            "weak_canonical_skills": [
                {"skill_tag": tag, "mastery": mastery} for tag, mastery in self.canonical_W
            ],
            "strong_canonical_skills": [
                {"skill_tag": tag, "mastery": mastery} for tag, mastery in self.canonical_ST
            ],
        }

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "LearnerState":
        return cls(**json.loads(s))


def build_states(df: pd.DataFrame, alpha: float, priors: pd.Series, as_of=("history",), tau_base: pd.Series | None = None) -> dict[int, LearnerState]:
    """Replay the log chronologically, using ONLY rows whose split is in `as_of` (default: history -> no leakage)."""
    d = df[df.split.isin(as_of)].sort_values(["user_id", "order_id", "skill_id"])
    mean_prior = float(priors.mean())
    states: dict[int, LearnerState] = {}
    tb_med = float(tau_base.median()) if tau_base is not None else None
    for u, k, y, ms, b in zip(d.user_id.to_numpy(), d.skill_id.to_numpy(), d.correct.to_numpy(), d.ms.to_numpy(), d.bucket.to_numpy()):
        st = states.get(u)
        if st is None:
            st = states[u] = LearnerState(int(u))
        tb = float(tau_base.get(k, tb_med)) if tau_base is not None else None
        st.update(k, float(y), float(ms), alpha, float(priors.get(k, mean_prior)), bucket=int(b) if tau_base is not None else None, tau_base=tb)
    return states
