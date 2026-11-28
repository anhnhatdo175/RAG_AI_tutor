"""SQLAlchemy Core persistence for the Moodle-facing service."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Index,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from lmsrag.layer1_data.learner_state import LearnerState

metadata = MetaData()

learners = Table(
    "learners",
    metadata,
    Column("learner_hash", String(64), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

tasks = Table(
    "tasks",
    metadata,
    Column("task_id", String(64), primary_key=True),
    Column("item_id", String(200), nullable=False),
    Column("skill_tag", String(200), nullable=False),
    Column("difficulty", Float, nullable=False),
    Column("prompt", Text, nullable=False),
    Column("response_type", String(40), nullable=False),
    Column("answer_key", Text),
    Column("hints", JSON, nullable=False),
    Column("generation", JSON, nullable=False),
    Column("mode", String(20), nullable=False),
    Column("status", String(24), nullable=False),
    Column("created_by_hash", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

task_citations = Table(
    "task_citations",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("task_id", String(64), ForeignKey("tasks.task_id", ondelete="CASCADE"), nullable=False),
    Column("chunk_id", String(300), nullable=False),
    Column("source", String(80), nullable=False),
    Column("doc_id", String(300), nullable=False),
    UniqueConstraint("task_id", "chunk_id", name="uq_task_citations_task_chunk"),
)

assignments = Table(
    "assignments",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("task_id", String(64), ForeignKey("tasks.task_id", ondelete="CASCADE"), nullable=False),
    Column("teacher_hash", String(64), nullable=False),
    Column("course_hash", String(64), nullable=False),
    Column("status", String(24), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("task_id", "course_hash", name="uq_assignments_task_course"),
)

submissions = Table(
    "submissions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("task_id", String(64), ForeignKey("tasks.task_id"), nullable=False),
    Column("learner_hash", String(64), ForeignKey("learners.learner_hash"), nullable=False),
    Column("idempotency_key", String(200), nullable=False),
    Column("request_hash", String(64), nullable=False),
    Column("answer", JSON, nullable=False),
    Column("status", String(24), nullable=False),
    Column("score", Float),
    Column("feedback", Text),
    Column("mastery_before", JSON),
    Column("mastery_after", JSON),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("learner_hash", "idempotency_key", name="uq_submissions_learner_idempotency"),
)

mastery_state = Table(
    "mastery_state",
    metadata,
    Column("learner_hash", String(64), ForeignKey("learners.learner_hash"), primary_key=True),
    Column("state_json", JSON, nullable=False),
    Column("version", Integer, nullable=False, default=1),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

applied_feedback_events = Table(
    "applied_feedback_events",
    metadata,
    Column("learner_hash", String(64), ForeignKey("learners.learner_hash"), primary_key=True),
    Column("event_id", String(200), primary_key=True),
    Column("event_fingerprint", String(64), nullable=False),
    Column("task_id", String(200)),
    Column("applied_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("learner_hash", "task_id", name="uq_feedback_learner_task"),
)

tutor_turns = Table(
    "tutor_turns",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(200), nullable=False),
    Column("learner_hash", String(64), ForeignKey("learners.learner_hash"), nullable=False),
    Column("task_id", String(64), ForeignKey("tasks.task_id"), nullable=False),
    Column("turn_index", Integer, nullable=False),
    Column("role", String(20), nullable=False),
    Column("content", Text, nullable=False),
    Column("score", Float),
    Column("response", JSON),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

decision_log = Table(
    "decision_log",
    metadata,
    Column("decision_id", String(64), primary_key=True),
    Column("learner_hash", String(64), ForeignKey("learners.learner_hash")),
    Column("endpoint", String(40), nullable=False),
    Column("query", Text),
    Column("retrieved_chunks", JSON, nullable=False),
    Column("scores", JSON, nullable=False),
    Column("reasons", JSON, nullable=False),
    Column("model", String(200)),
    Column("stage_latency_ms", JSON, nullable=False),
    Column("latency_ms", Float, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

chunks = Table(
    "chunks",
    metadata,
    Column("chunk_id", String(300), primary_key=True),
    Column("doc_id", String(300), nullable=False),
    Column("source", String(80), nullable=False),
    Column("skill_tag", String(200), nullable=False),
    Column("text", Text, nullable=False),
    Column("metadata", JSON, nullable=False),
)

Index("ix_tasks_status", tasks.c.status)
Index("ix_assignments_course_status", assignments.c.course_hash, assignments.c.status)
Index("ix_submissions_task_learner", submissions.c.task_id, submissions.c.learner_hash)
Index(
    "ix_tutor_turns_session",
    tutor_turns.c.session_id,
    tutor_turns.c.learner_hash,
    tutor_turns.c.task_id,
)
Index("ix_decision_log_created_at", decision_log.c.created_at)
Index("ix_chunks_doc_id", chunks.c.doc_id)


def create_database(url: str, *, echo: bool = False) -> "ServiceDatabase":
    if not isinstance(url, str) or not url.strip():
        raise ValueError("DATABASE_URL must be a non-empty SQLAlchemy URL")
    engine = create_engine(url, future=True, pool_pre_ping=True, echo=echo)
    return ServiceDatabase(engine)


class ServiceDatabase:
    def __init__(self, engine: Engine):
        self.engine = engine

    def create_schema(self) -> None:
        metadata.create_all(self.engine)

    def health(self) -> bool:
        from sqlalchemy import text

        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True

    def close(self) -> None:
        self.engine.dispose()

    def ensure_learner(self, learner_hash: str) -> None:
        if not isinstance(learner_hash, str) or len(learner_hash) != 64:
            raise ValueError("learner_hash must be a SHA-256 hex digest")
        now = datetime.now(timezone.utc)
        with self.engine.begin() as connection:
            insert_ignore(
                connection,
                learners,
                {"learner_hash": learner_hash, "created_at": now},
                [learners.c.learner_hash],
            )

    def add_decision(
        self,
        *,
        decision_id: str,
        learner_hash: str | None,
        endpoint: str,
        query: str | None,
        retrieved_chunks: list[dict[str, Any]],
        scores: dict[str, Any],
        reasons: dict[str, Any],
        model: str | None,
        stage_latency_ms: dict[str, float],
        latency_ms: float,
    ) -> None:
        with self.engine.begin() as connection:
            connection.execute(insert(decision_log).values(
                decision_id=decision_id,
                learner_hash=learner_hash,
                endpoint=endpoint,
                query=query,
                retrieved_chunks=retrieved_chunks,
                scores=scores,
                reasons=reasons,
                model=model,
                stage_latency_ms=stage_latency_ms,
                latency_ms=latency_ms,
                created_at=datetime.now(timezone.utc),
            ))


def insert_ignore(
    connection: Connection,
    table: Table,
    values: dict[str, Any],
    conflict_columns: list[Any],
) -> None:
    if connection.dialect.name == "sqlite":
        statement = sqlite_insert(table).values(**values).on_conflict_do_nothing(
            index_elements=conflict_columns
        )
    elif connection.dialect.name == "postgresql":
        statement = postgresql_insert(table).values(**values).on_conflict_do_nothing(
            index_elements=conflict_columns
        )
    else:
        raise RuntimeError(f"Unsupported service database dialect: {connection.dialect.name}")
    connection.execute(statement)


def insert_ignore_many(
    connection: Connection,
    table: Table,
    rows: list[dict[str, Any]],
    conflict_columns: list[Any],
) -> None:
    if not rows:
        return
    if connection.dialect.name == "sqlite":
        statement = sqlite_insert(table).on_conflict_do_nothing(
            index_elements=conflict_columns
        )
    elif connection.dialect.name == "postgresql":
        statement = postgresql_insert(table).on_conflict_do_nothing(
            index_elements=conflict_columns
        )
    else:
        raise RuntimeError(f"Unsupported service database dialect: {connection.dialect.name}")
    connection.execute(statement, rows)


class DatabaseStateStore:
    """StateStore-compatible adapter that persists only pseudonymous learner keys."""

    def __init__(self, database: ServiceDatabase):
        self.database = database
        self._runtime_hashes: dict[int, str] = {}

    def register_identity(self, runtime_id: int, learner_hash: str) -> None:
        previous = self._runtime_hashes.get(runtime_id)
        if previous is not None and previous != learner_hash:
            raise RuntimeError("Pseudonymous learner key collision")
        self.database.ensure_learner(learner_hash)
        self._runtime_hashes[runtime_id] = learner_hash

    def _learner_hash(self, user_id: int) -> str:
        learner_hash = self._runtime_hashes.get(int(user_id))
        if learner_hash is None:
            raise ValueError("Learner identity must be registered before state access")
        return learner_hash

    def count(self) -> int:
        with self.database.engine.connect() as connection:
            return int(connection.execute(select(func.count()).select_from(mastery_state)).scalar_one())

    def initialize_from_snapshot(self, snapshot_path: str) -> int:
        _ = snapshot_path
        raise RuntimeError("Service state must not import offline learner IDs")

    def get(self, user_id: int) -> LearnerState | None:
        learner_hash = self._learner_hash(user_id)
        with self.database.engine.connect() as connection:
            row = connection.execute(
                select(mastery_state.c.state_json).where(mastery_state.c.learner_hash == learner_hash)
            ).first()
        if row is None:
            return None
        value = row.state_json
        return LearnerState.from_json(value if isinstance(value, str) else json.dumps(value))

    def apply_canonical_event(
        self,
        user_id: int,
        skill_tag: str,
        score: float,
        rt_ms: float,
        alpha: float,
        prior: float = 0.5,
        **kw: Any,
    ) -> LearnerState:
        learner_hash = self._learner_hash(user_id)
        event_id = kw.pop("event_id", None)
        fingerprint = kw.pop("event_fingerprint", None)
        task_id = kw.get("task_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be a non-empty string")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("event_fingerprint must be a SHA-256 hex digest")

        with self.database.engine.begin() as connection:
            existing = connection.execute(
                select(
                    applied_feedback_events.c.event_fingerprint,
                    applied_feedback_events.c.task_id,
                ).where(
                    applied_feedback_events.c.learner_hash == learner_hash,
                    applied_feedback_events.c.event_id == event_id,
                )
            ).first()
            if existing is not None:
                if existing.event_fingerprint != fingerprint or existing.task_id != task_id:
                    raise ValueError("event_id was already used with a different submission")
                return self._get_on(connection, learner_hash)

            if task_id is not None:
                applied_task = connection.execute(
                    select(applied_feedback_events.c.event_id).where(
                        applied_feedback_events.c.learner_hash == learner_hash,
                        applied_feedback_events.c.task_id == task_id,
                    )
                ).first()
                if applied_task is not None:
                    raise ValueError("task was already applied to learner state")

            connection.execute(insert(applied_feedback_events).values(
                learner_hash=learner_hash,
                event_id=event_id,
                event_fingerprint=fingerprint,
                task_id=task_id,
                applied_at=datetime.now(timezone.utc),
            ))
            state = self._get_on(connection, learner_hash)
            if state is None:
                state = LearnerState(int(user_id))
            state.update_canonical(
                skill_tag=skill_tag,
                score=score,
                rt_ms=rt_ms,
                alpha=alpha,
                prior=prior,
                **kw,
            )
            state_json = state.to_json()
            current = connection.execute(
                select(mastery_state.c.version).where(mastery_state.c.learner_hash == learner_hash)
            ).first()
            if current is None:
                connection.execute(insert(mastery_state).values(
                    learner_hash=learner_hash,
                    state_json=json.loads(state_json),
                    version=1,
                    updated_at=datetime.now(timezone.utc),
                ))
            else:
                connection.execute(
                    update(mastery_state)
                    .where(mastery_state.c.learner_hash == learner_hash)
                    .values(
                        state_json=json.loads(state_json),
                        version=int(current.version) + 1,
                        updated_at=datetime.now(timezone.utc),
                    )
                )
            return state

    @staticmethod
    def _get_on(
        connection: Connection,
        learner_hash: str,
    ) -> LearnerState | None:
        row = connection.execute(
            select(mastery_state.c.state_json).where(mastery_state.c.learner_hash == learner_hash)
        ).first()
        if row is None:
            return None
        value = row.state_json
        return LearnerState.from_json(value if isinstance(value, str) else json.dumps(value))

    def close(self) -> None:
        self.database.close()
