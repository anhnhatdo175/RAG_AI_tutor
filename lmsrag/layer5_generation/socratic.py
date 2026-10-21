"""Small state machine for stepwise Socratic interaction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SocraticSession:
    task: dict[str, Any]
    question: str
    follow_up_questions: list[str] = field(default_factory=list)
    max_steps: int = 4
    turn_index: int = 1
    hint_level: int = 0
    attempts: int = 0
    complete: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)


class SocraticTutor:
    """Manage attempts and progressive hints without exposing answer keys."""

    def start(self, generation: dict[str, Any], max_steps: int = 4) -> SocraticSession:
        if generation.get("mode") != "socratic" or generation.get("status") != "ok":
            raise ValueError("A valid Socratic generation is required")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        tutor_turn = generation.get("tutor_turn") or {}
        question = tutor_turn.get("question")
        follow_up_questions = tutor_turn.get("follow_up_questions", [])
        task = generation.get("task")
        if not isinstance(question, str) or not question.strip() or not isinstance(task, dict):
            raise ValueError("Generation must include a task and initial Socratic question")
        if not isinstance(follow_up_questions, list) or any(
            not isinstance(item, str) or not item.strip() for item in follow_up_questions
        ):
            raise ValueError("follow_up_questions must be a list of non-empty questions")
        if len(follow_up_questions) + 1 > max_steps:
            raise ValueError("Socratic questions exceed max_steps")
        return SocraticSession(
            task=task,
            question=question,
            follow_up_questions=follow_up_questions,
            max_steps=max_steps,
        )

    def submit(self, session: SocraticSession, score: float) -> dict[str, Any]:
        if session.complete:
            raise ValueError("Socratic session is already complete")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not 0 <= score <= 1):
            raise ValueError("score must be in [0, 1]")

        session.attempts += 1
        session.history.append({
            "turn_index": session.turn_index,
            "question": session.question,
            "score": float(score),
        })
        if score >= 1.0:
            if session.follow_up_questions:
                if session.attempts >= session.max_steps:
                    session.complete = True
                    return {
                        "status": "limit_reached",
                        "complete": True,
                        "message": "Đã hết lượt cho phép; bước hiện tại đã đúng nhưng còn bước chưa hỏi.",
                    }
                session.question = session.follow_up_questions.pop(0)
                session.turn_index += 1
                return {
                    "status": "next_step",
                    "complete": False,
                    "question": session.question,
                    "turn_index": session.turn_index,
                }
            session.complete = True
            return {
                "status": "correct",
                "complete": True,
                "message": "Đúng rồi. Hãy giải thích ngắn gọn vì sao cách làm này hợp lý.",
            }

        hints = session.task.get("hints", [])
        if session.attempts >= session.max_steps:
            session.complete = True
            return {
                "status": "limit_reached",
                "complete": True,
                "message": "Mình sẽ dừng lượt gợi ý tại đây để tránh làm thay bạn.",
            }

        if session.hint_level < len(hints):
            hint = hints[session.hint_level]
            session.hint_level += 1
        else:
            hint = "Hãy xem lại mục tiêu của bước hiện tại và thử chia bài toán thành thao tác nhỏ hơn."
        session.turn_index += 1
        return {
            "status": "hint",
            "complete": False,
            "hint": hint,
            "question": session.question,
            "turn_index": session.turn_index,
        }