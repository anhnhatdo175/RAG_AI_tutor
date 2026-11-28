"""Request and response models for the Moodle-facing REST API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

ExternalId = StrictStr | StrictInt


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecommendRequest(APIModel):
    user_id: ExternalId
    skill_tag: str | None = Field(default=None, min_length=1, max_length=200)
    k: int = Field(default=5, ge=1, le=50)


class RecommendationReason(APIModel):
    mastery: float
    p_hat: float
    zpd_score: float
    cf_score: float
    cold_start: bool


class Citation(APIModel):
    chunk_id: str
    source: str


class RecommendedTask(APIModel):
    item_id: str
    skill_tag: str
    prompt: str
    difficulty: float
    response_type: str
    score: float
    reason: RecommendationReason
    citations: list[Citation]


class RecommendResponse(APIModel):
    recommendations: list[RecommendedTask]


class GenerateRequest(APIModel):
    teacher_id: ExternalId
    course_id: ExternalId
    skill_tag: str = Field(min_length=1, max_length=200)
    difficulty: float = Field(ge=0.0, le=1.0)
    count: int = Field(default=1, ge=1, le=10)
    mode: Literal["practice", "socratic"] = "practice"
    subject: str = Field(default="Mathematics", min_length=1, max_length=100)
    grade: str | None = Field(default=None, min_length=1, max_length=100)


class TaskPublic(APIModel):
    task_id: str
    item_id: str
    skill_tag: str
    difficulty: float
    prompt: str
    response_type: str
    hints: list[str]
    personalization_reason: str
    supporting_doc_ids: list[str]
    tutor_turn: dict[str, Any] | None = None
    status: Literal["draft", "needs_review", "approved", "rejected"]


class GenerateResponse(APIModel):
    tasks: list[TaskPublic]


class TaskEdit(APIModel):
    prompt: str | None = Field(default=None, min_length=1, max_length=10000)
    hints: list[str] | None = Field(default=None, max_length=10)
    personalization_reason: str | None = Field(default=None, max_length=2000)
    answer_key: str | None = Field(default=None, max_length=2000)


class ReviewRequest(APIModel):
    action: Literal["approve", "edit", "reject"]
    edit: TaskEdit | None = None

    @model_validator(mode="after")
    def validate_edit(self) -> "ReviewRequest":
        if self.action == "edit" and self.edit is None:
            raise ValueError("edit details are required when action is edit")
        if self.action != "edit" and self.edit is not None:
            raise ValueError("edit details are only valid when action is edit")
        return self


class ReviewResponse(APIModel):
    task: TaskPublic


class SubmitRequest(APIModel):
    user_id: ExternalId
    task_id: str = Field(min_length=1, max_length=64)
    answer: Any
    idempotency_key: str = Field(min_length=1, max_length=200)
    rt_ms: float = Field(default=0.0, ge=0.0)


class SubmitResponse(APIModel):
    status: Literal["graded", "needs_review", "processing"]
    correct: bool | None
    score: float | None
    feedback: str
    mastery_before: dict[str, float] | None = None
    mastery_after: dict[str, float] | None = None


class ProfileResponse(APIModel):
    mastery: dict[str, float]
    variance: float
    attempts_by_skill: dict[str, int]
    weak_skills: list[dict[str, Any]]
    strong_skills: list[dict[str, Any]]
    total_attempts: int


class TutorRequest(APIModel):
    user_id: ExternalId
    task_id: str = Field(min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=4000)


class TutorResponse(APIModel):
    session_id: str
    status: str
    complete: bool
    turn_index: int
    question: str | None = None
    hint: str | None = None
    message: str | None = None


class SyncEvent(APIModel):
    user_id: ExternalId
    task_id: str = Field(min_length=1, max_length=64)
    answer: Any
    idempotency_key: str = Field(min_length=1, max_length=200)
    rt_ms: float = Field(default=0.0, ge=0.0)


class SyncRequest(APIModel):
    events: list[SyncEvent] = Field(max_length=500)


class SyncResponse(APIModel):
    accepted: int
    results: list[SubmitResponse]


class HealthResponse(APIModel):
    status: Literal["ok"]
