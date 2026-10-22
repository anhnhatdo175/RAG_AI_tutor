"""Application orchestration for one terminal-based learning turn."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lmsrag import config, data
from lmsrag.layer1_data.learner_state import LearnerState
from lmsrag.layer1_data.state_store import StateStore
from lmsrag.layer3_retrieval.retriever import (
    AdaptiveRetriever,
    CrossEncoderReranker,
    load_query_encoder,
)
from lmsrag.layer4_recommender import HybridRecommender
from lmsrag.layer5_generation import GenerationContext, LLMGenerator
from lmsrag.layer6_feedback import FeedbackLoop
from lmsrag.skill_mapping import SkillMapper
from lmsrag.layer2_embedding.index import load_and_verify


@dataclass(frozen=True)
class LearningActivity:
    task_id: str | None
    item_id: str | None
    generation: dict[str, Any]
    status: str = "generated"


class NoLearningContentError(RuntimeError):
    """Raised when the current filters leave no grounded lesson or exercise."""


class _IssuedTaskStore:
    def __init__(self, path: str | Path):
        self.db = sqlite3.connect(str(path), timeout=30)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS issued_learning_tasks ("
            "task_id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, item_id TEXT NOT NULL, "
            "payload TEXT NOT NULL, status TEXT NOT NULL, event_id TEXT, "
            "response TEXT, rt_ms REAL, created_at REAL NOT NULL, updated_at REAL NOT NULL)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS issued_learning_tasks_user_status "
            "ON issued_learning_tasks(user_id, status, created_at)"
        )
        self.db.commit()

    def get_open(self, user_id: int) -> tuple[str, str, dict[str, Any]] | None:
        row = self.db.execute(
            "SELECT task_id,item_id,payload,status FROM issued_learning_tasks "
            "WHERE user_id=? AND status IN ('issued','submitting') "
            "ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[2])
        payload["status"] = row[3]
        return str(row[0]), str(row[1]), payload

    def issue(
        self, user_id: int, item_id: str, task_payload: dict[str, Any],
        generation: dict[str, Any],
    ) -> str:
        task_id = str(uuid.uuid4())
        now = time.time()
        payload = json.dumps(
            {"issued_task": task_payload, "generation": generation},
            ensure_ascii=False,
            allow_nan=False,
        )
        with self.db:
            self.db.execute(
                "INSERT INTO issued_learning_tasks "
                "(task_id,user_id,item_id,payload,status,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (task_id, user_id, item_id, payload, "issued", now, now),
            )
        return task_id

    def get(self, task_id: str, user_id: int) -> tuple[dict[str, Any], str, str | None, str | None, float | None]:
        row = self.db.execute(
            "SELECT payload,status,event_id,response,rt_ms FROM issued_learning_tasks "
            "WHERE task_id=? AND user_id=?",
            (task_id, user_id),
        ).fetchone()
        if row is None:
            raise ValueError("Task does not exist or does not belong to this learner")
        return json.loads(row[0]), str(row[1]), row[2], row[3], row[4]

    def begin_submission(
        self, task_id: str, event_id: str, response: Any, rt_ms: float,
    ) -> None:
        with self.db:
            cursor = self.db.execute(
                "UPDATE issued_learning_tasks SET status='submitting',event_id=?,response=?,"
                "rt_ms=?,updated_at=? WHERE task_id=? AND status='issued'",
                (event_id, json.dumps(response, ensure_ascii=False), rt_ms, time.time(), task_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Task is no longer in the issued state")

    def complete(self, task_id: str, status: str) -> None:
        with self.db:
            cursor = self.db.execute(
                "UPDATE issued_learning_tasks SET status=?,updated_at=? "
                "WHERE task_id=? AND status='submitting'",
                (status, time.time(), task_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Could not persist the task's completed status")

    def close(self) -> None:
        self.db.close()


class LearningPipeline:
    """Connect retrieval, recommendation, generation, grading, and live state."""

    def __init__(
        self,
        *,
        state_store: StateStore,
        retriever: AdaptiveRetriever,
        recommender: HybridRecommender,
        encoder: Any,
        generator: LLMGenerator | None = None,
        task_database: str | Path | None = None,
    ):
        self.state_store = state_store
        self.retriever = retriever
        self.recommender = recommender
        self.encoder = encoder
        self.generator = generator or LLMGenerator()
        self.feedback = FeedbackLoop(state_store)
        self.issued_tasks = _IssuedTaskStore(task_database or state_store.path)

    @classmethod
    def from_artifacts(cls) -> "LearningPipeline":
        """Load built artifacts and prepare a separate persistent online state DB."""
        bank_path = config.ART / "exercise_bank.jsonl"
        live_path = config.ART / "learner_state_live.sqlite"
        required = [
            config.ART / "index_manifest.json",
            config.CHUNKS_JSONL,
            config.EMB_NPY,
            config.SKILL_CATALOG_JSON,
            bank_path,
            config.ART / "learner_state.sqlite",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing pipeline artifacts. Build Layer 1 and Layer 2 first: "
                + ", ".join(missing)
            )

        store, _ = load_and_verify()
        encoder = load_query_encoder()
        mapper = SkillMapper()
        logs = data.load_logs(mapper)
        bank = [
            json.loads(line)
            for line in bank_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        catalog = json.loads(config.SKILL_CATALOG_JSON.read_text(encoding="utf-8"))
        recommender = HybridRecommender(store, bank, catalog, logs)
        retriever = AdaptiveRetriever(store, encoder, CrossEncoderReranker())

        state_store = StateStore(live_path)
        try:
            if state_store.count() == 0:
                state_store.initialize_from_snapshot(config.ART / "learner_state.sqlite")
            return cls(
                state_store=state_store,
                retriever=retriever,
                recommender=recommender,
                encoder=encoder,
            )
        except Exception:
            state_store.close()
            raise

    def get_pending_activity(self, user_id: int) -> LearningActivity | None:
        pending = self.issued_tasks.get_open(user_id)
        if pending is None:
            return None
        task_id, item_id, payload = pending
        return LearningActivity(
            task_id=task_id,
            item_id=item_id,
            generation=payload["generation"],
            status=payload["status"],
        )

    def prepare_activity(
        self,
        *,
        user_id: int,
        query: str,
        target_subject: str,
        target_grade: str,
        target_skill: str,
        mode: str = "practice",
        language: str = "vi",
    ) -> LearningActivity:
        if not query.strip():
            raise ValueError("query must not be empty")
        pending = self.get_pending_activity(user_id)
        if pending is not None:
            return pending

        state = self.state_store.get(user_id) or LearnerState(user_id=user_id)
        query_vectors = np.asarray(self.encoder.encode_queries([query]), dtype="float32")
        if query_vectors.ndim != 2 or query_vectors.shape[0] != 1:
            raise ValueError("Query encoder must return one vector for one query")
        query_vector = query_vectors[0]
        documents = self.retriever.retrieve(
            query,
            target_subject=target_subject,
            target_grade=target_grade,
            target_skill=target_skill,
            top_k=5,
        )
        if not documents:
            raise NoLearningContentError(
                "No indexed learning material matches the subject, grade, and skill."
            )
        recommendations = self.recommender.recommend(
            user_id=user_id,
            state=state,
            query_vector=query_vector,
            target_subject=target_subject,
            target_grade=target_grade,
            target_skill=target_skill,
            top_k=5,
        )
        if not recommendations:
            raise NoLearningContentError(
                "No unused mapped exercise matches the subject, grade, and skill."
            )

        selected = recommendations[0]
        if not selected.get("gold_answer"):
            raise ValueError(
                f"Exercise {selected.get('item_id')!r} has no answer key and cannot be auto-graded."
            )
        generation_candidate = {
            field: selected[field]
            for field in (
                "item_id",
                "doc_id",
                "skill_tag",
                "mapping_status",
                "subject",
                "grade",
                "difficulty",
                "statement",
                "response_type",
            )
            if field in selected
        }
        context = GenerationContext.from_sources(
            retrieved_docs=documents,
            candidate_tasks=[generation_candidate],
            learner_state=state,
            constraints={
                "mode": mode,
                "target_skill": target_skill,
                "target_subject": target_subject,
                "target_grade": target_grade,
                "language": language,
                "allow_new_task_generation": False,
            },
            top_k=5,
            top_n=1,
        )
        generation = self.generator.generate(context)
        if generation["status"] == "insufficient_context":
            return LearningActivity(None, None, generation)
        generated_task = generation["task"]
        item_id = str(selected["item_id"])
        if generated_task["item_id"] != item_id:
            raise ValueError("Generated task does not match the selected exercise")

        issued_task = {
            **generated_task,
            "item_id": item_id,
            "skill_tag": selected["skill_tag"],
            "difficulty": selected["difficulty"],
            "response_type": selected.get("response_type", "exact"),
            "answer_key": selected["gold_answer"],
            "user_id": user_id,
            "status": "issued",
        }
        task_id = self.issued_tasks.issue(
            user_id, item_id, issued_task, generation
        )
        return LearningActivity(task_id, item_id, generation)

    def submit_answer(
        self,
        *,
        user_id: int,
        task_id: str,
        response: Any = None,
        rt_ms: float | None = None,
    ) -> dict[str, Any]:
        payload, status, event_id, saved_response, saved_rt_ms = self.issued_tasks.get(
            task_id, user_id
        )
        task = payload["issued_task"]
        if status == "issued":
            if response is None or rt_ms is None:
                raise ValueError("A response and response time are required")
            event_id = str(uuid.uuid4())
            self.issued_tasks.begin_submission(task_id, event_id, response, rt_ms)
        elif status == "submitting":
            if event_id is None or saved_rt_ms is None:
                raise RuntimeError("Interrupted submission is missing recovery data")
            if response is not None and json.dumps(response, ensure_ascii=False) != saved_response:
                raise ValueError("A submission is already being processed with another response")
            response = json.loads(saved_response) if saved_response is not None else None
            rt_ms = float(saved_rt_ms)
        else:
            raise ValueError("Task has already been submitted")

        if event_id is None or rt_ms is None:
            raise RuntimeError("Submission recovery data is incomplete")
        result = self.feedback.submit(
            user_id=user_id,
            event_id=event_id,
            task=task,
            response=response,
            rt_ms=float(rt_ms),
        )
        final_status = "graded" if result["status"] == "graded" else "needs_review"
        self.issued_tasks.complete(task_id, final_status)
        return result

    def close(self) -> None:
        self.issued_tasks.close()
        self.state_store.close()
