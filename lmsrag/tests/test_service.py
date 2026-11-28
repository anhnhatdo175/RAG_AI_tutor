from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from lmsrag.application import LearningActivity
from lmsrag.layer5_generation.generator import GenerationValidationError
from lmsrag.layer6_feedback.feedback import FeedbackLoop
from lmsrag.service.app import create_app
from lmsrag.service.db import (
    DatabaseStateStore,
    ServiceDatabase,
    applied_feedback_events,
    decision_log,
    learners,
    submissions,
    tutor_turns,
)
from lmsrag.service.engine import ServiceEngine

SERVICE_TOKEN = "test-service-token-that-is-at-least-32-characters"
LEARNER_SALT = "test-learner-salt-that-is-at-least-32-characters"


class FakeEncoder:
    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in queries], dtype="float32")


class FakeRecommender:
    exercise_bank = [{
        "item_id": "bank-item-1",
        "doc_id": "doc-1",
        "subject": "Mathematics",
        "grade": "8",
        "skill_tag": "Unit-Rate",
        "difficulty": 0.6,
        "statement": "A source exercise.",
        "response_type": "exact",
        "mapping_status": "mapped",
    }]
    chunk_ids_by_doc = {"doc-1": [0]}

    def __init__(self) -> None:
        self.store = SimpleNamespace(
            meta=[{
                "chunk_id": "chunk-1",
                "doc_id": "doc-1",
                "source": "test-source",
                "skill_tag": "Unit-Rate",
                "text": "Grounding text for the test.",
            }],
            vectors=np.asarray([[1.0, 0.0]], dtype="float32"),
        )

    def recommend(self, **_kwargs: object) -> list[dict[str, object]]:
        assert _kwargs["target_skill"] == "Unit-Rate"
        return [{
            **self.exercise_bank[0],
            "ranking_score": 0.9,
            "mastery": 0.5,
            "cf_predicted_success": 0.7,
            "zpd_score": 0.8,
            "cf_score": 0.9,
            "mastery_source": "neutral_cold_start",
        }]


class FakePipeline:
    def __init__(self, state_store: DatabaseStateStore) -> None:
        self.state_store = state_store
        self.feedback = FeedbackLoop(state_store)
        self.recommender = FakeRecommender()
        self.encoder = FakeEncoder()
        self.fail_generation = False
        self.generation_calls: list[dict[str, object]] = []

    def prepare_activity(self, **kwargs: object) -> LearningActivity:
        self.generation_calls.append(kwargs)
        if self.fail_generation:
            raise GenerationValidationError("model JSON remained invalid after repair")
        mode = str(kwargs["mode"])
        generation = {
            "schema_version": "1.0",
            "status": "ok",
            "mode": mode,
            "task": {
                "item_id": "new_task",
                "skill_tag": "Unit-Rate",
                "difficulty": 0.6,
                "prompt": "What is the answer?",
                "response_type": "exact",
                "hints": ["Check the arithmetic."],
                "personalization_reason": "Practice a mapped skill.",
                "supporting_doc_ids": ["doc-1"],
            },
            "tutor_turn": (
                {"question": "What operation should you use?", "follow_up_questions": []}
                if mode == "socratic" else None
            ),
            "safety": {"contains_final_answer": False, "grounded": True},
        }
        return LearningActivity(
            task_id=None,
            item_id=None,
            generation=generation,
            internal_task={
                "item_id": "GEN-test-task",
                "answer_key": "42",
                "skill_tag": "Unit-Rate",
                "difficulty": 0.6,
                "response_type": "exact",
                "reference_item_id": f"bank-item-{len(self.generation_calls)}",
            },
            stage_latency_ms={"retrieval": 1.0, "generation": 2.0},
        )

    def close(self) -> None:
        return None


@pytest.fixture
def service():
    raw_engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    database = ServiceDatabase(raw_engine)
    state_store = DatabaseStateStore(database)
    pipeline = FakePipeline(state_store)
    service_engine = ServiceEngine(database, pipeline, LEARNER_SALT)
    app = create_app(service_engine=service_engine, service_token=SERVICE_TOKEN)
    with TestClient(app) as client:
        yield client, pipeline, database
    database.close()


def headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {SERVICE_TOKEN}"}


def assert_no_answer_key(response) -> None:
    assert "answer_key" not in response.text


def decision_count(database: ServiceDatabase, endpoint: str) -> int:
    with database.engine.connect() as connection:
        return int(connection.execute(
            select(func.count()).select_from(decision_log).where(
                decision_log.c.endpoint == endpoint
            )
        ).scalar_one())


def test_health_recommendation_and_bearer_auth(service) -> None:
    client, _, database = service
    health = client.get("/health")
    unauthorized = client.post("/recommend", json={"user_id": "moodle-user"})
    assert health.status_code == 200
    assert unauthorized.status_code == 401
    assert_no_answer_key(health)
    assert_no_answer_key(unauthorized)

    response = client.post(
        "/recommend",
        headers=headers(),
        json={"user_id": "moodle-user", "skill_tag": "Unit-Rate", "k": 3},
    )
    assert response.status_code == 200
    recommendation = response.json()["recommendations"][0]
    assert recommendation["item_id"] == "bank-item-1"
    assert recommendation["citations"] == [{"chunk_id": "chunk-1", "source": "test-source"}]
    assert recommendation["reason"]["cold_start"] is True
    assert_no_answer_key(response)
    assert decision_count(database, "health") == 1
    assert decision_count(database, "recommend") == 1
    assert decision_count(database, "/recommend") == 1


def test_review_gate_submission_idempotency_and_private_identity(service) -> None:
    client, _, database = service
    generated = client.post(
        "/generate",
        headers=headers(),
        json={
            "teacher_id": "teacher-1",
            "course_id": "course-1",
            "skill_tag": "Unit-Rate",
            "difficulty": 0.6,
            "count": 1,
        },
    )
    assert generated.status_code == 200
    task = generated.json()["tasks"][0]
    assert task["status"] == "draft"
    assert_no_answer_key(generated)

    submission = {
        "user_id": "moodle-user-42",
        "task_id": task["task_id"],
        "answer": "42",
        "idempotency_key": "attempt-1",
    }
    unapproved_submission = client.post("/submit", headers=headers(), json=submission)
    assert unapproved_submission.status_code == 409
    assert_no_answer_key(unapproved_submission)

    approved = client.post(
        f"/tasks/{task['task_id']}/review",
        headers=headers(),
        json={"action": "approve"},
    )
    assert approved.status_code == 200
    assert approved.json()["task"]["status"] == "approved"
    assert_no_answer_key(approved)

    cold_profile = client.get("/profile/moodle-user-42", headers=headers())
    assert cold_profile.status_code == 200
    assert_no_answer_key(cold_profile)
    assert cold_profile.json()["mastery"]["Unit-Rate"] == 0.5
    assert cold_profile.json()["variance"] == 0.25

    first = client.post("/submit", headers=headers(), json=submission)
    replay = client.post("/submit", headers=headers(), json=submission)
    assert first.status_code == replay.status_code == 200
    assert_no_answer_key(first)
    assert_no_answer_key(replay)
    assert first.json() == replay.json()
    assert first.json()["score"] == 1.0
    assert first.json()["mastery_after"]["Unit-Rate"] > first.json()["mastery_before"]["Unit-Rate"]

    changed_answer = {**submission, "answer": "not-42"}
    replay_conflict = client.post("/submit", headers=headers(), json=changed_answer)
    assert replay_conflict.status_code == 409
    assert_no_answer_key(replay_conflict)
    profile = client.get("/profile/moodle-user-42", headers=headers())
    assert profile.status_code == 200
    assert_no_answer_key(profile)
    assert profile.json()["total_attempts"] == 1
    assert profile.json()["attempts_by_skill"] == {"Unit-Rate": 1}

    with database.engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(submissions)).scalar_one() == 1
        assert connection.execute(
            select(func.count()).select_from(applied_feedback_events)
        ).scalar_one() == 1
        learner_hashes = connection.execute(
            select(learners.c.learner_hash)
        ).scalars().all()
        assert learner_hashes
        assert "moodle-user-42" not in learner_hashes
    assert decision_count(database, "generate") == 1
    assert decision_count(database, "review") == 1
    assert decision_count(database, "profile") == 2
    assert decision_count(database, "submit") == 2
    assert decision_count(database, "/submit") == 2


def test_malformed_generation_is_held_for_teacher_review(service) -> None:
    client, pipeline, database = service
    pipeline.fail_generation = True
    response = client.post(
        "/generate",
        headers=headers(),
        json={
            "teacher_id": "teacher-2",
            "course_id": "course-2",
            "skill_tag": "Unit-Rate",
            "difficulty": 0.6,
            "count": 1,
        },
    )
    assert response.status_code == 200
    assert_no_answer_key(response)
    task = response.json()["tasks"][0]
    assert task["status"] == "needs_review"

    blocked = client.post(
        "/submit",
        headers=headers(),
        json={
            "user_id": "moodle-user-2",
            "task_id": task["task_id"],
            "answer": "42",
            "idempotency_key": "attempt-needs-review",
        },
    )
    assert blocked.status_code == 409
    assert_no_answer_key(blocked)
    assert decision_count(database, "generate") == 1
    assert decision_count(database, "/submit") == 1


def test_socratic_task_returns_a_persisted_turn(service) -> None:
    client, _, database = service
    generated = client.post(
        "/generate",
        headers=headers(),
        json={
            "teacher_id": "teacher-3",
            "course_id": "course-3",
            "skill_tag": "Unit-Rate",
            "difficulty": 0.6,
            "mode": "socratic",
        },
    )
    assert_no_answer_key(generated)
    task = generated.json()["tasks"][0]
    assert generated.status_code == 200
    assert "What operation" in task["tutor_turn"]["question"]
    review = client.post(
        f"/tasks/{task['task_id']}/review",
        headers=headers(),
        json={"action": "approve"},
    )
    assert review.status_code == 200
    assert_no_answer_key(review)

    turn = client.post(
        "/tutor",
        headers=headers(),
        json={
            "user_id": "moodle-user-3",
            "task_id": task["task_id"],
            "session_id": "session-3",
            "message": "42",
        },
    )
    assert turn.status_code == 200
    assert_no_answer_key(turn)
    assert turn.json()["status"] == "correct"
    assert turn.json()["complete"] is True
    with database.engine.connect() as connection:
        assert connection.execute(
            select(func.count()).select_from(tutor_turns)
        ).scalar_one() == 2
        assert connection.execute(
            select(func.count()).select_from(decision_log).where(
                decision_log.c.endpoint == "tutor"
            )
        ).scalar_one() == 1
    assert decision_count(database, "generate") == 1
    assert decision_count(database, "review") == 1


def test_sync_accepts_submissions_and_persists_profile(service) -> None:
    client, _, database = service
    created = client.post(
        "/generate",
        headers=headers(),
        json={
            "teacher_id": "teacher-sync",
            "course_id": "course-sync",
            "skill_tag": "Unit-Rate",
            "difficulty": 0.6,
        },
    )
    assert created.status_code == 200
    assert_no_answer_key(created)
    task = created.json()["tasks"][0]
    approved = client.post(
        f"/tasks/{task['task_id']}/review",
        headers=headers(),
        json={"action": "approve"},
    )
    assert approved.status_code == 200
    assert_no_answer_key(approved)

    synced = client.post(
        "/sync",
        headers=headers(),
        json={
            "events": [{
                "user_id": "offline-user",
                "task_id": task["task_id"],
                "answer": "42",
                "idempotency_key": "offline-attempt",
                "rt_ms": 1200,
            }]
        },
    )
    assert synced.status_code == 200
    assert_no_answer_key(synced)
    assert synced.json()["accepted"] == 1
    assert synced.json()["rejected"] == 0
    assert synced.json()["results"][0]["idempotency_key"] == "offline-attempt"
    assert synced.json()["results"][0]["score"] == 1.0
    profile = client.get("/profile/offline-user", headers=headers())
    assert profile.status_code == 200
    assert_no_answer_key(profile)
    assert profile.json()["attempts_by_skill"] == {"Unit-Rate": 1}
    assert decision_count(database, "generate") == 1
    assert decision_count(database, "review") == 1
    assert decision_count(database, "sync") == 1
    assert decision_count(database, "profile") == 1


def generate_task(client, *, mode: str = "practice", teacher: str = "teacher-x") -> dict:
    response = client.post(
        "/generate",
        headers=headers(),
        json={
            "teacher_id": teacher,
            "course_id": "course-x",
            "skill_tag": "Unit-Rate",
            "difficulty": 0.6,
            "mode": mode,
        },
    )
    assert response.status_code == 200
    assert_no_answer_key(response)
    return response.json()["tasks"][0]


def approve(client, task_id: str):
    response = client.post(
        f"/tasks/{task_id}/review",
        headers=headers(),
        json={"action": "approve"},
    )
    assert_no_answer_key(response)
    return response


def test_exception_new_learner_starts_neutral_with_largest_variance(service) -> None:
    client, _, database = service
    recommended = client.post(
        "/recommend",
        headers=headers(),
        json={"user_id": 987654, "skill_tag": "Unit-Rate"},
    )
    profile = client.get("/profile/987654", headers=headers())
    assert recommended.status_code == profile.status_code == 200
    reason = recommended.json()["recommendations"][0]["reason"]
    assert reason["cold_start"] is True
    assert reason["mastery"] == 0.5
    assert profile.json()["mastery"] == {"Unit-Rate": 0.5}
    assert profile.json()["variance"] == 0.25
    assert profile.json()["total_attempts"] == 0
    assert_no_answer_key(recommended)
    assert_no_answer_key(profile)
    assert decision_count(database, "recommend") == 1
    assert decision_count(database, "profile") == 1


def test_exception_invalid_llm_json_is_logged_with_its_failure(service) -> None:
    client, pipeline, database = service
    pipeline.fail_generation = True
    task = generate_task(client)
    assert task["status"] == "needs_review"
    assert task["prompt"] == ""
    with database.engine.connect() as connection:
        reasons = connection.execute(
            select(decision_log.c.reasons).where(decision_log.c.endpoint == "generate")
        ).scalar_one()
    assert reasons["status"] == "needs_review"
    assert "invalid after repair" in reasons["failure"]


def test_exception_needs_review_task_is_withheld_until_teacher_fixes_it(service) -> None:
    client, pipeline, _ = service
    pipeline.fail_generation = True
    task = generate_task(client)

    premature = approve(client, task["task_id"])
    assert premature.status_code == 409

    edited = client.post(
        f"/tasks/{task['task_id']}/review",
        headers=headers(),
        json={
            "action": "edit",
            "edit": {
                "prompt": "Một xe đi 120 km trong 2 giờ. Vận tốc là bao nhiêu km/h?",
                "answer_key": "60",
            },
        },
    )
    assert edited.status_code == 200
    assert edited.json()["task"]["status"] == "draft"
    assert_no_answer_key(edited)

    blocked = client.post("/submit", headers=headers(), json={
        "user_id": "learner-review", "task_id": task["task_id"],
        "answer": "60", "idempotency_key": "draft-attempt",
    })
    assert blocked.status_code == 409

    assert approve(client, task["task_id"]).status_code == 200
    graded = client.post("/submit", headers=headers(), json={
        "user_id": "learner-review", "task_id": task["task_id"],
        "answer": "60", "idempotency_key": "approved-attempt",
    })
    assert graded.status_code == 200
    assert graded.json()["correct"] is True
    assert_no_answer_key(graded)


def test_exception_duplicate_submission_does_not_update_mastery_twice(service) -> None:
    client, _, database = service
    task = generate_task(client)
    assert approve(client, task["task_id"]).status_code == 200
    payload = {
        "user_id": "learner-dup", "task_id": task["task_id"],
        "answer": "41", "idempotency_key": "same-key", "rt_ms": 900,
    }
    responses = [client.post("/submit", headers=headers(), json=payload) for _ in range(3)]
    assert {response.status_code for response in responses} == {200}
    assert responses[0].json() == responses[1].json() == responses[2].json()
    assert responses[0].json()["correct"] is False
    profile = client.get("/profile/learner-dup", headers=headers()).json()
    assert profile["total_attempts"] == 1
    with database.engine.connect() as connection:
        replays = connection.execute(
            select(decision_log.c.reasons).where(decision_log.c.endpoint == "submit")
        ).scalars().all()
    assert sum(bool(reason.get("idempotent_replay")) for reason in replays) == 2


def test_rejected_task_cannot_be_submitted_or_reviewed_again(service) -> None:
    client, _, _ = service
    task = generate_task(client)
    rejected = client.post(
        f"/tasks/{task['task_id']}/review",
        headers=headers(),
        json={"action": "reject"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["task"]["status"] == "rejected"
    assert approve(client, task["task_id"]).status_code == 409
    submitted = client.post("/submit", headers=headers(), json={
        "user_id": "learner-r", "task_id": task["task_id"],
        "answer": "42", "idempotency_key": "rejected-attempt",
    })
    assert submitted.status_code == 409
    assert_no_answer_key(submitted)


def test_edit_action_requires_edit_fields(service) -> None:
    client, _, database = service
    task = generate_task(client)
    response = client.post(
        f"/tasks/{task['task_id']}/review",
        headers=headers(),
        json={"action": "edit"},
    )
    assert response.status_code == 422
    assert decision_count(database, "/tasks/{task_id}/review") == 1


def test_unknown_task_and_skill_return_not_found(service) -> None:
    client, _, database = service
    missing = "00000000-0000-0000-0000-000000000000"
    responses = [
        client.post(f"/tasks/{missing}/review", headers=headers(), json={"action": "reject"}),
        client.post("/submit", headers=headers(), json={
            "user_id": "u", "task_id": missing, "answer": "1", "idempotency_key": "k",
        }),
        client.post("/tutor", headers=headers(), json={
            "user_id": "u", "task_id": missing, "session_id": "s", "message": "1",
        }),
        client.post("/recommend", headers=headers(), json={
            "user_id": "u", "skill_tag": "No-Such-Skill",
        }),
        client.post("/generate", headers=headers(), json={
            "teacher_id": "t", "course_id": "c", "skill_tag": "Unit-Rate", "difficulty": 0.05,
        }),
    ]
    assert [response.status_code for response in responses] == [404] * 5
    for response in responses:
        assert_no_answer_key(response)
    assert decision_count(database, "/submit") == 1
    assert decision_count(database, "/tutor") == 1


def test_tutor_rejects_practice_tasks_and_completed_sessions(service) -> None:
    client, _, _ = service
    practice = generate_task(client)
    assert approve(client, practice["task_id"]).status_code == 200
    wrong_mode = client.post("/tutor", headers=headers(), json={
        "user_id": "u", "task_id": practice["task_id"], "session_id": "s", "message": "42",
    })
    assert wrong_mode.status_code == 409

    socratic = generate_task(client, mode="socratic", teacher="teacher-s")
    assert approve(client, socratic["task_id"]).status_code == 200
    body = {"user_id": "u", "task_id": socratic["task_id"], "session_id": "s2", "message": "42"}
    assert client.post("/tutor", headers=headers(), json=body).json()["complete"] is True
    again = client.post("/tutor", headers=headers(), json=body)
    assert again.status_code == 409
    assert_no_answer_key(again)


def test_sync_rejects_bad_events_without_blocking_good_ones(service) -> None:
    client, _, database = service
    task = generate_task(client)
    assert approve(client, task["task_id"]).status_code == 200
    events = {"events": [
        {"user_id": "s1", "task_id": task["task_id"], "answer": "42", "idempotency_key": "e1"},
        {"user_id": "s2", "task_id": "missing-task", "answer": "42", "idempotency_key": "e2"},
        {"user_id": "s3", "task_id": task["task_id"], "answer": "7", "idempotency_key": "e3"},
    ]}
    first = client.post("/sync", headers=headers(), json=events)
    replay = client.post("/sync", headers=headers(), json=events)
    assert first.status_code == replay.status_code == 200
    assert first.json() == replay.json()
    assert first.json()["accepted"] == 2
    assert first.json()["rejected"] == 1
    statuses = {item["idempotency_key"]: item["status"] for item in first.json()["results"]}
    assert statuses == {"e1": "graded", "e2": "rejected", "e3": "graded"}
    assert first.json()["results"][1]["error"] == "Task was not found"
    assert_no_answer_key(first)
    with database.engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(submissions)).scalar_one() == 2
    assert client.get("/profile/s1", headers=headers()).json()["total_attempts"] == 1


def test_every_protected_endpoint_requires_the_bearer_token(service) -> None:
    client, _, database = service
    calls = [
        ("post", "/recommend", {"user_id": "u"}),
        ("post", "/generate", {"teacher_id": "t", "course_id": "c",
                               "skill_tag": "Unit-Rate", "difficulty": 0.6}),
        ("post", "/tasks/x/review", {"action": "approve"}),
        ("post", "/submit", {"user_id": "u", "task_id": "x", "answer": 1,
                             "idempotency_key": "k"}),
        ("get", "/profile/u", None),
        ("post", "/tutor", {"user_id": "u", "task_id": "x", "session_id": "s",
                            "message": "m"}),
        ("post", "/sync", {"events": []}),
    ]
    for method, path, body in calls:
        kwargs = {"json": body} if body is not None else {}
        wrong = getattr(client, method)(
            path, headers={"Authorization": "Bearer wrong-token"}, **kwargs
        )
        missing = getattr(client, method)(path, **kwargs)
        assert wrong.status_code == missing.status_code == 401, path
    with database.engine.connect() as connection:
        logged = connection.execute(
            select(func.count()).select_from(decision_log)
        ).scalar_one()
    assert logged == 2 * len(calls)


def test_unexpected_errors_return_500_and_are_still_logged(service) -> None:
    client, pipeline, database = service

    def broken_encoder(_queries):
        raise RuntimeError("encoder crashed")

    pipeline.encoder.encode_queries = broken_encoder
    safe_client = TestClient(client.app, raise_server_exceptions=False)
    response = safe_client.post(
        "/recommend",
        headers=headers(),
        json={"user_id": "u", "skill_tag": "Unit-Rate"},
    )
    assert response.status_code == 500
    assert "encoder crashed" not in response.text
    with database.engine.connect() as connection:
        reasons = connection.execute(
            select(decision_log.c.reasons).where(decision_log.c.endpoint == "/recommend")
        ).scalar_one()
    assert reasons == {"http_status": 500}


def test_external_ids_are_validated(service) -> None:
    client, _, _ = service
    for user_id in ["", " ", "x" * 201, True]:
        response = client.post(
            "/recommend",
            headers=headers(),
            json={"user_id": user_id, "skill_tag": "Unit-Rate"},
        )
        assert response.status_code == 422, user_id


def test_generate_batch_uses_a_different_source_exercise_per_task(service) -> None:
    client, pipeline, _ = service
    response = client.post(
        "/generate",
        headers=headers(),
        json={"teacher_id": "t", "course_id": "c", "skill_tag": "Unit-Rate",
              "difficulty": 0.6, "count": 3},
    )
    assert response.status_code == 200
    excluded = [set(call["exclude_item_ids"]) for call in pipeline.generation_calls]
    assert excluded == [set(), {"bank-item-1"}, {"bank-item-1", "bank-item-2"}]
