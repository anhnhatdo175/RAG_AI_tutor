"""Grade submitted tasks and apply validated feedback to learner state."""
from __future__ import annotations

import math
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from lmsrag.config import EMA_ALPHA, GAMMA, HALF_LIFE_OPPS
from lmsrag.layer1_data.learner_state import LearnerState
from lmsrag.layer1_data.state_store import StateStore


@dataclass(frozen=True)
class GradingResult:
    status: str
    score: float | None
    rubric_id: str | None = None


Grader = Callable[[dict[str, Any], Any], GradingResult]


class FeedbackLoop:
    """Validate a submission, grade it, and atomically update mastery/profile."""

    def __init__(self, state_store: StateStore, grader: Grader | None = None,
                 alpha: float = EMA_ALPHA, gamma: float = GAMMA,
                 half_life_opportunities: float = HALF_LIFE_OPPS):
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not math.isfinite(half_life_opportunities) or half_life_opportunities <= 0:
            raise ValueError("half_life_opportunities must be positive")
        self.state_store = state_store
        self.grader = grader or self._grade_answer_key
        self.alpha = alpha
        self.gamma = gamma
        self.decay = 2 ** (-1 / half_life_opportunities)

    def submit(self, *, user_id: int, event_id: str, task: dict[str, Any],
               response: Any, rt_ms: float, prior: float = 0.5,
               tau_base: float | None = None) -> dict[str, Any]:
        if not event_id or not isinstance(event_id, str):
            raise ValueError("event_id must be a non-empty string")
        if isinstance(user_id, bool) or not isinstance(user_id, int):
            raise ValueError("user_id must be an integer")
        if not isinstance(task, dict) or task.get("status") != "issued":
            raise ValueError("task must be a valid issued task")
        task_user_id = task.get("user_id")
        if isinstance(task_user_id, bool) or not isinstance(task_user_id, int):
            raise ValueError("issued task must include an integer user_id")
        if task_user_id != user_id:
            raise ValueError("task does not belong to user_id")
        if isinstance(rt_ms, bool) or not isinstance(rt_ms, (int, float)) or not math.isfinite(rt_ms) or rt_ms < 0:
            raise ValueError("rt_ms must be finite and non-negative")
        if isinstance(prior, bool) or not isinstance(prior, (int, float)) or not math.isfinite(prior) or not 0 <= prior <= 1:
            raise ValueError("prior must be finite and in [0, 1]")
        if tau_base is not None and (
            isinstance(tau_base, bool) or not isinstance(tau_base, (int, float))
            or not math.isfinite(tau_base) or tau_base <= 0
        ):
            raise ValueError("tau_base must be finite and positive when provided")
        task_id = task.get("item_id") or task.get("task_id")
        skill_tag = task.get("skill_tag")
        if not isinstance(task_id, (str, int)) or not str(task_id).strip() or not isinstance(skill_tag, str) or not skill_tag.strip():
            raise ValueError("task requires item_id/task_id and skill_tag")
        bucket = task.get("difficulty_bucket")
        if bucket is None:
            difficulty = task.get("difficulty")
            if isinstance(difficulty, bool) or not isinstance(difficulty, (int, float)) or not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
                raise ValueError("task requires difficulty or difficulty_bucket")
            bucket = min(2, max(0, int(float(difficulty) * 3)))
        if isinstance(bucket, bool) or not isinstance(bucket, int) or bucket not in (0, 1, 2):
            raise ValueError("difficulty_bucket must be 0, 1, or 2")

        result = self.grader(task, response)
        if not isinstance(result, GradingResult):
            raise TypeError("grader must return GradingResult")
        if result.status not in {"graded", "needs_review"}:
            raise ValueError("grader status must be graded or needs_review")
        if result.status != "graded":
            if result.score is not None:
                raise ValueError("needs_review result must not include a score")
            return {"status": result.status, "score": None, "state": None}
        if (isinstance(result.score, bool) or not isinstance(result.score, (int, float))
                or not math.isfinite(result.score) or not 0 <= result.score <= 1):
            raise ValueError("graded result score must be finite and in [0, 1]")

        fingerprint_payload = {
            "task_id": str(task_id),
            "skill_tag": skill_tag,
            "response": response,
            "score": float(result.score),
            "rt_ms": float(rt_ms),
            "prior": float(prior),
            "bucket": bucket,
            "tau_base": tau_base,
            "rubric_id": result.rubric_id,
            "response_type": task.get("response_type"),
            "answer_key": task.get("answer_key"),
            "gold_answer": task.get("gold_answer"),
            "task_version": task.get("task_version"),
        }
        fingerprint = hashlib.sha256(json.dumps(
            fingerprint_payload, sort_keys=True, ensure_ascii=False, default=str,
        ).encode("utf-8")).hexdigest()
        updated_state = self.state_store.apply_canonical_event(
            user_id=user_id,
            skill_tag=skill_tag,
            score=float(result.score),
            rt_ms=rt_ms,
            alpha=self.alpha,
            prior=prior,
            bucket=bucket,
            tau_base=tau_base,
            gamma=self.gamma,
            decay=self.decay,
            task_id=str(task_id),
            event_id=event_id,
            event_fingerprint=fingerprint,
        )
        return {
            "status": "graded",
            "score": float(result.score),
            "rubric_id": result.rubric_id,
            "state": updated_state,
            "context_snapshot": updated_state.context_snapshot(),
        }

    @staticmethod
    def _grade_answer_key(task: dict[str, Any], response: Any) -> GradingResult:
        if "answer_key" in task:
            expected = task["answer_key"]
        elif "gold_answer" in task:
            expected = task["gold_answer"]
        else:
            return GradingResult("needs_review", None, task.get("rubric_id"))
        response_type = task.get("response_type", "exact")
        rubric_id = task.get("rubric_id")
        if response_type == "numeric":
            tolerance_value = task.get("tolerance", 0.0)
            if (isinstance(tolerance_value, bool)
                    or not isinstance(tolerance_value, (int, float))
                    or not math.isfinite(tolerance_value) or tolerance_value < 0):
                raise ValueError("numeric tolerance must be a finite non-negative number")
            try:
                expected_number = float(expected)
                tolerance = float(tolerance_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("numeric answer key must be a finite number") from exc
            if not math.isfinite(expected_number):
                raise ValueError("numeric answer key must be a finite number")
            try:
                actual_number = float(str(response).strip())
            except (TypeError, ValueError):
                return GradingResult("graded", 0.0, rubric_id)
            if not math.isfinite(actual_number):
                return GradingResult("graded", 0.0, rubric_id)
            score = float(abs(actual_number - expected_number) <= tolerance)
            return GradingResult("graded", score, rubric_id)
        if response_type == "exact":
            normalize = lambda value: re.sub(r"\s+", " ", str(value).strip()).casefold()
            return GradingResult("graded", float(normalize(response) == normalize(expected)), rubric_id)
        return GradingResult("needs_review", None, rubric_id)


def state_after_feedback(result: dict[str, Any]) -> LearnerState | None:
    """Typed convenience accessor for callers that consume only the updated state."""
    state = result.get("state")
    return state if isinstance(state, LearnerState) else None