"""Moodle-facing operations that adapt the existing Layer 1-6 pipeline."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import insert, select, update

from lmsrag import config
from lmsrag.application import LearningPipeline, NoLearningContentError
from lmsrag.layer1_data.learner_state import LearnerState
from lmsrag.layer5_generation.generator import GenerationValidationError
from lmsrag.layer5_generation.socratic import SocraticTutor
from lmsrag.layer6_feedback.feedback import GradingResult
from lmsrag.service.db import (
    DatabaseStateStore,
    ServiceDatabase,
    assignments,
    chunks,
    insert_ignore,
    insert_ignore_many,
    submissions,
    task_citations,
    tasks,
    tutor_turns,
)


class ServiceNotFoundError(LookupError):
    """A requested task or assignment does not exist."""


class ServiceConflictError(RuntimeError):
    """The request conflicts with persisted task or submission state."""


class ServiceEngine:
    def __init__(
        self,
        database: ServiceDatabase,
        pipeline: LearningPipeline,
        learner_id_salt: str,
    ):
        if not isinstance(learner_id_salt, str) or len(learner_id_salt) < 32:
            raise ValueError("LEARNER_ID_SALT must contain at least 32 characters")
        self.database = database
        self.pipeline = pipeline
        self.state_store = pipeline.state_store
        self.learner_id_salt = learner_id_salt.encode("utf-8")
        self.socratic_tutor = SocraticTutor()
        self._submission_locks = [threading.Lock() for _ in range(64)]
        self.database.create_schema()
        chunk_rows = [
            {
                "chunk_id": str(item["chunk_id"]),
                "doc_id": str(item["doc_id"]),
                "source": str(item["source"]),
                "skill_tag": str(item["skill_tag"]),
                "text": str(item["text"]),
                "metadata": item,
            }
            for item in self.pipeline.recommender.store.meta
        ]
        self._chunk_meta_by_id = {
            str(item["chunk_id"]): item
            for item in self.pipeline.recommender.store.meta
        }
        with self.database.engine.begin() as connection:
            insert_ignore_many(
                connection,
                chunks,
                chunk_rows,
                [chunks.c.chunk_id],
            )

    @classmethod
    def from_artifacts(
        cls,
        database: ServiceDatabase,
        learner_id_salt: str,
    ) -> "ServiceEngine":
        state_store = DatabaseStateStore(database)
        pipeline = LearningPipeline.from_artifacts(
            state_store=state_store,
            initialize_state_snapshot=False,
            task_database=os.getenv(
                "SERVICE_TASK_DATABASE",
                str(config.ART / "service_issued_tasks.sqlite"),
            ),
        )
        return cls(database, pipeline, learner_id_salt)

    def close(self) -> None:
        self.pipeline.close()
        self.database.close()

    def learner_identity(self, external_id: str | int) -> tuple[str, int]:
        learner_hash = self.hash_external_id("learner", external_id)
        runtime_id = int(learner_hash[:15], 16)
        if not isinstance(self.state_store, DatabaseStateStore):
            raise RuntimeError("Service pipeline requires DatabaseStateStore")
        self.state_store.register_identity(runtime_id, learner_hash)
        return learner_hash, runtime_id

    def hash_external_id(self, namespace: str, external_id: str | int) -> str:
        if isinstance(external_id, bool) or not isinstance(external_id, (str, int)):
            raise ValueError("External identifiers must be strings or integers")
        normalized = str(external_id).strip()
        if not normalized or len(normalized) > 200:
            raise ValueError("External identifiers must contain 1 to 200 characters")
        message = f"{namespace}:{normalized}".encode("utf-8")
        return hmac.new(self.learner_id_salt, message, hashlib.sha256).hexdigest()

    def _generation_identity(
        self,
        teacher_hash: str,
        course_hash: str,
        task_nonce: str,
    ) -> int:
        synthetic = hmac.new(
            self.learner_id_salt,
            f"generation:{teacher_hash}:{course_hash}:{task_nonce}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        runtime_id = int(synthetic[:15], 16)
        if not isinstance(self.state_store, DatabaseStateStore):
            raise RuntimeError("Service pipeline requires DatabaseStateStore")
        self.state_store.register_identity(runtime_id, synthetic)
        return runtime_id

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _row_dict(row: Any) -> dict[str, Any]:
        return dict(row._mapping)

    def _task_row(self, task_id: str) -> dict[str, Any]:
        with self.database.engine.connect() as connection:
            row = connection.execute(
                select(tasks).where(tasks.c.task_id == task_id)
            ).first()
        if row is None:
            raise ServiceNotFoundError("Task was not found")
        return self._row_dict(row)

    @staticmethod
    def _public_task(row: dict[str, Any]) -> dict[str, Any]:
        generation = row.get("generation") or {}
        generated_task = generation.get("task") or {}
        return {
            "task_id": row["task_id"],
            "item_id": row["item_id"],
            "skill_tag": row["skill_tag"],
            "difficulty": row["difficulty"],
            "prompt": row["prompt"],
            "response_type": row["response_type"],
            "hints": row.get("hints") or [],
            "personalization_reason": generated_task.get("personalization_reason", ""),
            "supporting_doc_ids": generated_task.get("supporting_doc_ids", []),
            "tutor_turn": generation.get("tutor_turn"),
            "status": row["status"],
        }

    def _chunks_for_doc(self, doc_id: str) -> list[dict[str, str]]:
        recommender = self.pipeline.recommender
        indices = recommender.chunk_ids_by_doc.get(str(doc_id), [])
        citations = []
        for index in indices:
            metadata = recommender.store.meta[index]
            citations.append({
                "chunk_id": str(metadata["chunk_id"]),
                "source": str(metadata["source"]),
            })
        return citations

    def recommend(
        self,
        user_id: str | int,
        skill_tag: str | None,
        k: int,
    ) -> list[dict[str, Any]]:
        started = time.perf_counter()
        learner_hash, runtime_id = self.learner_identity(user_id)
        state = self.state_store.get(runtime_id) or LearnerState(runtime_id)
        bank = self.pipeline.recommender.exercise_bank
        groups = sorted({
            (str(item.get("subject")), str(item.get("grade")), str(item.get("skill_tag")))
            for item in bank
            if item.get("mapping_status") == "mapped"
            and (skill_tag is None or item.get("skill_tag") == skill_tag)
        })
        if skill_tag is not None and not groups:
            raise ServiceNotFoundError("No mapped exercises match the requested skill")

        query_ms = 0.0
        ranking_ms = 0.0
        ranked: list[tuple[float, dict[str, Any], list[dict[str, str]]]] = []
        for subject, grade, skill in groups:
            query = skill.replace("-", " ")
            stage_started = time.perf_counter()
            vectors = self.pipeline.encoder.encode_queries([query])
            query_vector = vectors[0]
            query_ms += (time.perf_counter() - stage_started) * 1000

            stage_started = time.perf_counter()
            candidates = self.pipeline.recommender.recommend(
                user_id=runtime_id,
                state=state,
                query_vector=query_vector,
                target_subject=subject,
                target_grade=grade,
                target_skill=skill,
                top_k=k,
            )
            ranking_ms += (time.perf_counter() - stage_started) * 1000
            for item in candidates:
                citations = self._chunks_for_doc(str(item["doc_id"]))
                ranked.append((float(item["ranking_score"]), item, citations))

        ranked.sort(key=lambda result: result[0], reverse=True)
        response = []
        cited_chunks: list[dict[str, str]] = []
        scores: dict[str, Any] = {}
        reasons: dict[str, Any] = {}
        for score, item, citations in ranked[:k]:
            response.append({
                "item_id": str(item["item_id"]),
                "skill_tag": str(item["skill_tag"]),
                "prompt": str(item.get("statement", item.get("prompt", ""))),
                "difficulty": float(item["difficulty"]),
                "response_type": str(item.get("response_type", "exact")),
                "score": score,
                "reason": {
                    "mastery": float(item["mastery"]),
                    "p_hat": float(item["cf_predicted_success"]),
                    "zpd_score": float(item["zpd_score"]),
                    "cf_score": float(item["cf_score"]),
                    "cold_start": state.n == 0,
                },
                "citations": citations,
            })
            cited_chunks.extend(citations)
            scores[str(item["item_id"])] = score
            reasons[str(item["item_id"])] = {
                "mastery": item["mastery"],
                "p_hat": item["cf_predicted_success"],
                "zpd_score": item["zpd_score"],
                "cf_score": item["cf_score"],
                "mastery_source": item["mastery_source"],
            }

        total_ms = (time.perf_counter() - started) * 1000
        self.database.add_decision(
            decision_id=str(uuid.uuid4()),
            learner_hash=learner_hash,
            endpoint="recommend",
            query=skill_tag,
            retrieved_chunks=cited_chunks,
            scores=scores,
            reasons=reasons,
            model=None,
            stage_latency_ms={
                "query_embedding": query_ms,
                "ranking": ranking_ms,
            },
            latency_ms=total_ms,
        )
        return response

    def _resolve_grade(
        self,
        *,
        skill_tag: str,
        subject: str,
        difficulty: float,
        requested_grade: str | None,
    ) -> str:
        candidates = [
            item for item in self.pipeline.recommender.exercise_bank
            if item.get("mapping_status") == "mapped"
            and item.get("skill_tag") == skill_tag
            and item.get("subject") == subject
            and (requested_grade is None or str(item.get("grade")) == requested_grade)
            and abs(float(item.get("difficulty", -1.0)) - difficulty) <= 0.10
        ]
        if not candidates:
            raise NoLearningContentError(
                "No mapped source exercise matches the requested skill, subject, grade, and difficulty."
            )
        if requested_grade is not None:
            return requested_grade
        counts = Counter(str(item["grade"]) for item in candidates)
        return sorted(counts, key=lambda grade: (-counts[grade], grade))[0]

    def generate(
        self,
        *,
        teacher_id: str | int,
        course_id: str | int,
        skill_tag: str,
        difficulty: float,
        count: int,
        mode: str,
        subject: str,
        grade: str | None,
    ) -> list[dict[str, Any]]:
        started = time.perf_counter()
        teacher_hash = self.hash_external_id("teacher", teacher_id)
        course_hash = self.hash_external_id("course", course_id)
        target_grade = self._resolve_grade(
            skill_tag=skill_tag,
            subject=subject,
            difficulty=difficulty,
            requested_grade=grade,
        )
        generated = []
        used_source_items: set[str] = set()
        for _ in range(count):
            task_nonce = str(uuid.uuid4())
            runtime_id = self._generation_identity(teacher_hash, course_hash, task_nonce)
            activity = None
            failure: str | None = None
            try:
                activity = self.pipeline.prepare_activity(
                    user_id=runtime_id,
                    query=skill_tag.replace("-", " "),
                    target_subject=subject,
                    target_grade=target_grade,
                    target_skill=skill_tag,
                    mode=mode,
                    language="vi",
                    allow_new_task_generation=True,
                    target_difficulty=difficulty,
                    issue_task=False,
                    exclude_item_ids=frozenset(used_source_items),
                )
                if activity.internal_task is not None:
                    used_source_items.add(str(activity.internal_task.get("reference_item_id", "")))
            except (GenerationValidationError, NoLearningContentError) as exc:
                failure = str(exc)

            generation = activity.generation if activity is not None else {
                "schema_version": "1.0",
                "status": "needs_review",
                "mode": mode,
                "task": None,
                "tutor_turn": None,
                "safety": {"contains_final_answer": False, "grounded": False},
            }
            internal_task = activity.internal_task if activity is not None else None
            successful = (
                failure is None
                and activity is not None
                and generation.get("status") == "ok"
                and internal_task is not None
            )
            task_id = str(uuid.uuid4())
            generated_task = generation.get("task") if successful else {}
            status = "draft" if successful else "needs_review"
            item_id = (
                str(internal_task["item_id"])
                if successful
                else f"REVIEW-{task_id}"
            )
            prompt = str(generated_task.get("prompt", ""))
            citations: list[dict[str, str]] = []
            for doc_id in generated_task.get("supporting_doc_ids", []):
                citations.extend(self._chunks_for_doc(str(doc_id)))

            task_values = {
                "task_id": task_id,
                "item_id": item_id,
                "skill_tag": skill_tag,
                "difficulty": difficulty,
                "prompt": prompt,
                "response_type": str(generated_task.get("response_type", "exact")),
                "answer_key": str(internal_task["answer_key"]) if successful else None,
                "hints": generated_task.get("hints", []),
                "generation": generation,
                "mode": mode,
                "status": status,
                "created_by_hash": teacher_hash,
                "created_at": self._now(),
                "updated_at": self._now(),
            }
            with self.database.engine.begin() as connection:
                connection.execute(insert(tasks).values(**task_values))
                for citation in {item["chunk_id"]: item for item in citations}.values():
                    source_meta = self._chunk_meta_by_id.get(citation["chunk_id"])
                    if source_meta is not None:
                        connection.execute(insert(task_citations).values(
                            task_id=task_id,
                            chunk_id=citation["chunk_id"],
                            source=citation["source"],
                            doc_id=str(source_meta["doc_id"]),
                        ))
                connection.execute(insert(assignments).values(
                    task_id=task_id,
                    teacher_hash=teacher_hash,
                    course_hash=course_hash,
                    status=status,
                    created_at=self._now(),
                ))

            stage_latency = activity.stage_latency_ms if activity is not None else {}
            if failure:
                stage_latency = {**stage_latency, "generation_error": 0.0}
            self.database.add_decision(
                decision_id=str(uuid.uuid4()),
                learner_hash=None,
                endpoint="generate",
                query=skill_tag,
                retrieved_chunks=citations,
                scores={},
                reasons={
                    "status": status,
                    "failure": failure,
                    "target_difficulty": difficulty,
                    "target_grade": target_grade,
                },
                model=config.GENERATION_MODEL,
                stage_latency_ms=stage_latency,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            row = self._task_row(task_id)
            generated.append(self._public_task(row))
        return generated

    def review(
        self,
        task_id: str,
        *,
        action: str,
        edit: dict[str, Any] | None,
    ) -> dict[str, Any]:
        row = self._task_row(task_id)
        if row["status"] not in {"draft", "needs_review"}:
            raise ServiceConflictError("Only draft or needs_review tasks can be reviewed")
        values: dict[str, Any] = {"updated_at": self._now()}
        assignment_status = None
        if action == "approve":
            if not row.get("answer_key") and not (edit and edit.get("answer_key")):
                raise ServiceConflictError("A task needs an answer key before it can be approved")
            values["status"] = "approved"
            assignment_status = "active"
        elif action == "reject":
            values["status"] = "rejected"
            assignment_status = "rejected"
        elif action == "edit":
            if not edit:
                raise ValueError("Edit fields are required for an edit action")
            generation = dict(row["generation"] or {})
            generated_task = dict(generation.get("task") or {})
            for key in ("prompt", "hints", "personalization_reason"):
                if key in edit:
                    generated_task[key] = edit[key]
            generation["task"] = generated_task
            if "prompt" in edit:
                values["prompt"] = edit["prompt"]
            if "hints" in edit:
                values["hints"] = edit["hints"]
            if "answer_key" in edit:
                values["answer_key"] = str(edit["answer_key"])
            values["generation"] = generation
            values["status"] = "draft"
            assignment_status = "draft"
        else:
            raise ValueError("action must be approve, edit, or reject")

        with self.database.engine.begin() as connection:
            connection.execute(update(tasks).where(tasks.c.task_id == task_id).values(**values))
            if assignment_status is not None:
                connection.execute(
                    update(assignments)
                    .where(assignments.c.task_id == task_id)
                    .values(status=assignment_status)
                )
        updated = self._task_row(task_id)
        self.database.add_decision(
            decision_id=str(uuid.uuid4()),
            learner_hash=None,
            endpoint="review",
            query=None,
            retrieved_chunks=[],
            scores={},
            reasons={"action": action, "status": updated["status"]},
            model=None,
            stage_latency_ms={},
            latency_ms=0.0,
        )
        return self._public_task(updated)

    @staticmethod
    def _submission_response(row: dict[str, Any]) -> dict[str, Any]:
        score = row["score"]
        return {
            "status": row["status"],
            "correct": None if score is None else bool(score >= 1.0),
            "score": score,
            "feedback": row["feedback"] or "",
            "mastery_before": row["mastery_before"],
            "mastery_after": row["mastery_after"],
        }

    def submit(
        self,
        *,
        user_id: str | int,
        task_id: str,
        answer: Any,
        idempotency_key: str,
        rt_ms: float,
    ) -> dict[str, Any]:
        lock_key = hashlib.sha256(
            f"{user_id}:{idempotency_key}".encode("utf-8")
        ).digest()
        lock_index = int.from_bytes(lock_key[:2], "big") % len(self._submission_locks)
        lock = self._submission_locks[lock_index]
        with lock:
            return self._submit_locked(
                user_id=user_id,
                task_id=task_id,
                answer=answer,
                idempotency_key=idempotency_key,
                rt_ms=rt_ms,
            )

    def _submit_locked(
        self,
        *,
        user_id: str | int,
        task_id: str,
        answer: Any,
        idempotency_key: str,
        rt_ms: float,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        learner_hash, runtime_id = self.learner_identity(user_id)
        task_row = self._task_row(task_id)
        if task_row["status"] != "approved":
            raise ServiceConflictError("Task is not approved for learner submissions")
        with self.database.engine.connect() as connection:
            assignment = connection.execute(
                select(assignments.c.id).where(
                    assignments.c.task_id == task_id,
                    assignments.c.status == "active",
                )
            ).first()
        if assignment is None:
            raise ServiceConflictError("Task is not assigned to an active course")

        request_fingerprint = hashlib.sha256(json.dumps(
            {"task_id": task_id, "answer": answer, "rt_ms": rt_ms},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
            allow_nan=False,
        ).encode("utf-8")).hexdigest()
        now = self._now()
        cached_response = None
        with self.database.engine.begin() as connection:
            insert_ignore(
                connection,
                submissions,
                {
                    "task_id": task_id,
                    "learner_hash": learner_hash,
                    "idempotency_key": idempotency_key,
                    "request_hash": request_fingerprint,
                    "answer": answer,
                    "status": "processing",
                    "score": None,
                    "feedback": None,
                    "mastery_before": None,
                    "mastery_after": None,
                    "created_at": now,
                    "updated_at": now,
                },
                [submissions.c.learner_hash, submissions.c.idempotency_key],
            )
            stored = connection.execute(
                select(submissions).where(
                    submissions.c.learner_hash == learner_hash,
                    submissions.c.idempotency_key == idempotency_key,
                )
            ).first()
            submission = self._row_dict(stored)
            if submission["request_hash"] != request_fingerprint or submission["task_id"] != task_id:
                raise ServiceConflictError("Idempotency key was already used for a different submission")
            if submission["status"] != "processing":
                cached_response = self._submission_response(submission)

        if cached_response is not None:
            self.database.add_decision(
                decision_id=str(uuid.uuid4()),
                learner_hash=learner_hash,
                endpoint="submit",
                query=None,
                retrieved_chunks=[],
                scores={"score": cached_response["score"]},
                reasons={"task_id": task_id, "status": cached_response["status"], "idempotent_replay": True},
                model=None,
                stage_latency_ms={"idempotency_lookup": (time.perf_counter() - started) * 1000},
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            return cached_response

        state_before = self.state_store.get(runtime_id) or LearnerState(runtime_id)
        mastery_before = {
            task_row["skill_tag"]: float(state_before.canonical_M.get(task_row["skill_tag"], 0.5))
        }
        issued_task = {
            "item_id": task_row["item_id"],
            "task_id": task_id,
            "skill_tag": task_row["skill_tag"],
            "difficulty": task_row["difficulty"],
            "response_type": task_row["response_type"],
            "answer_key": task_row["answer_key"],
            "user_id": runtime_id,
            "status": "issued",
        }
        event_id = hashlib.sha256(
            f"{learner_hash}:{idempotency_key}".encode("utf-8")
        ).hexdigest()
        grade_started = time.perf_counter()
        grading: GradingResult = self.pipeline.feedback.grader(issued_task, answer)
        grading_ms = (time.perf_counter() - grade_started) * 1000
        feedback_ms = 0.0
        if grading.status == "needs_review":
            response = {
                "status": "needs_review",
                "correct": None,
                "score": None,
                "feedback": "Bài cần giáo viên xem xét; tiến độ chưa được cập nhật.",
                "mastery_before": mastery_before,
                "mastery_after": mastery_before,
            }
        else:
            feedback_started = time.perf_counter()
            result = self.pipeline.feedback.submit(
                user_id=runtime_id,
                event_id=event_id,
                task=issued_task,
                response=answer,
                rt_ms=rt_ms,
            )
            feedback_ms = (time.perf_counter() - feedback_started) * 1000
            state_after = result["state"]
            mastery_after = {
                task_row["skill_tag"]: float(
                    state_after.canonical_M.get(task_row["skill_tag"], 0.5)
                )
            }
            score = float(result["score"])
            response = {
                "status": "graded",
                "correct": score >= 1.0,
                "score": score,
                "feedback": (
                    "Đúng rồi." if score >= 1.0
                    else "Câu trả lời chưa đúng. Hãy xem lại gợi ý."
                ),
                "mastery_before": mastery_before,
                "mastery_after": mastery_after,
            }

        with self.database.engine.begin() as connection:
            connection.execute(
                update(submissions)
                .where(
                    submissions.c.learner_hash == learner_hash,
                    submissions.c.idempotency_key == idempotency_key,
                )
                .values(
                    status=response["status"],
                    score=response["score"],
                    feedback=response["feedback"],
                    mastery_before=response["mastery_before"],
                    mastery_after=response["mastery_after"],
                    updated_at=self._now(),
                )
            )
        self.database.add_decision(
            decision_id=str(uuid.uuid4()),
            learner_hash=learner_hash,
            endpoint="submit",
            query=None,
            retrieved_chunks=[],
            scores={"score": response["score"]},
            reasons={"task_id": task_id, "status": response["status"]},
            model=None,
            stage_latency_ms={
                "grading": grading_ms,
                "mastery_update": feedback_ms,
            },
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return response

    def profile(self, user_id: str | int) -> dict[str, Any]:
        learner_hash, runtime_id = self.learner_identity(user_id)
        state = self.state_store.get(runtime_id) or LearnerState(runtime_id)
        mapped_skills = {
            str(item["skill_tag"])
            for item in self.pipeline.recommender.exercise_bank
            if item.get("mapping_status") == "mapped"
        }
        response = {
            "mastery": {
                skill: float(state.canonical_M.get(skill, 0.5))
                for skill in sorted(mapped_skills | set(state.canonical_M))
            },
            "variance": 0.25 if state.n == 0 else float(state.sigma ** 2),
            "attempts_by_skill": {
                key: int(value) for key, value in state.canonical_attempts.items()
            },
            "weak_skills": [
                {"skill_tag": tag, "mastery": float(mastery)}
                for tag, mastery in state.canonical_W
            ],
            "strong_skills": [
                {"skill_tag": tag, "mastery": float(mastery)}
                for tag, mastery in state.canonical_ST
            ],
            "total_attempts": int(state.n),
        }
        self.database.add_decision(
            decision_id=str(uuid.uuid4()),
            learner_hash=learner_hash,
            endpoint="profile",
            query=None,
            retrieved_chunks=[],
            scores={},
            reasons={"total_attempts": state.n},
            model=None,
            stage_latency_ms={},
            latency_ms=0.0,
        )
        return response

    def tutor(
        self,
        *,
        user_id: str | int,
        task_id: str,
        session_id: str,
        message: str,
    ) -> dict[str, Any]:
        learner_hash, runtime_id = self.learner_identity(user_id)
        session_hash = self.hash_external_id("session", session_id)
        row = self._task_row(task_id)
        if row["status"] != "approved" or row["mode"] != "socratic":
            raise ServiceConflictError("Task is not an approved Socratic activity")
        generation = row["generation"]
        if not isinstance(generation, dict):
            raise ServiceConflictError("Socratic generation data is unavailable")
        with self.database.engine.connect() as connection:
            events = connection.execute(
                select(tutor_turns)
                .where(
                    tutor_turns.c.session_id == session_hash,
                    tutor_turns.c.learner_hash == learner_hash,
                    tutor_turns.c.task_id == task_id,
                )
                .order_by(tutor_turns.c.id)
            ).fetchall()
        session = self.socratic_tutor.start(generation)
        for event in events:
            event = self._row_dict(event)
            if event["role"] == "student":
                if session.complete:
                    raise ServiceConflictError("Socratic session is already complete")
                self.socratic_tutor.submit(session, float(event["score"] or 0.0))
        if session.complete:
            raise ServiceConflictError("Socratic session is already complete")

        internal_task = {
            "item_id": row["item_id"],
            "task_id": task_id,
            "skill_tag": row["skill_tag"],
            "difficulty": row["difficulty"],
            "response_type": row["response_type"],
            "answer_key": row["answer_key"],
            "user_id": runtime_id,
            "status": "issued",
        }
        grading = self.pipeline.feedback.grader(internal_task, message)
        if grading.status != "graded" or grading.score is None:
            raise ServiceConflictError("Task answer requires teacher review")
        submitted_turn_index = session.turn_index
        outcome = self.socratic_tutor.submit(session, float(grading.score))
        now = self._now()
        assistant_text = (
            outcome.get("hint")
            or outcome.get("question")
            or outcome.get("message")
            or ""
        )
        with self.database.engine.begin() as connection:
            connection.execute(insert(tutor_turns).values(
                session_id=session_hash,
                learner_hash=learner_hash,
                task_id=task_id,
                turn_index=submitted_turn_index,
                role="student",
                content=message,
                score=float(grading.score),
                response=None,
                created_at=now,
            ))
            connection.execute(insert(tutor_turns).values(
                session_id=session_hash,
                learner_hash=learner_hash,
                task_id=task_id,
                turn_index=submitted_turn_index,
                role="assistant",
                content=assistant_text,
                score=None,
                response=outcome,
                created_at=self._now(),
            ))
        self.database.add_decision(
            decision_id=str(uuid.uuid4()),
            learner_hash=learner_hash,
            endpoint="tutor",
            query=None,
            retrieved_chunks=[],
            scores={"score": float(grading.score)},
            reasons={"task_id": task_id, "status": outcome["status"]},
            model=config.GENERATION_MODEL,
            stage_latency_ms={},
            latency_ms=0.0,
        )
        return {
            "session_id": session_id,
            "turn_index": session.turn_index,
            **outcome,
        }

    def sync(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply Moodle attempts one by one; a rejected event never blocks the rest.

        Each result carries the event's idempotency key so the Moodle task can
        mark accepted attempts as sent and keep rejected ones for a teacher.
        """
        started = time.perf_counter()
        results = []
        for event in events:
            try:
                outcome = self.submit(
                    user_id=event["user_id"],
                    task_id=event["task_id"],
                    answer=event["answer"],
                    idempotency_key=event["idempotency_key"],
                    rt_ms=float(event.get("rt_ms", 0.0)),
                )
                outcome = {**outcome, "error": None}
            except (ServiceNotFoundError, ServiceConflictError, ValueError) as exc:
                outcome = {
                    "status": "rejected",
                    "correct": None,
                    "score": None,
                    "feedback": "",
                    "mastery_before": None,
                    "mastery_after": None,
                    "error": str(exc),
                }
            results.append({"idempotency_key": event["idempotency_key"], **outcome})
        accepted = sum(result["status"] != "rejected" for result in results)
        self.database.add_decision(
            decision_id=str(uuid.uuid4()),
            learner_hash=None,
            endpoint="sync",
            query=None,
            retrieved_chunks=[],
            scores={"accepted": accepted, "rejected": len(results) - accepted},
            reasons={"event_count": len(events)},
            model=None,
            stage_latency_ms={"submission_processing": (time.perf_counter() - started) * 1000},
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return results

