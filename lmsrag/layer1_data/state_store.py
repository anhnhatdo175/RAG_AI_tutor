"""LAYER 1 / state management: persistent per-learner state in SQLite (stdlib, transactional, versioned rows)."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .learner_state import LearnerState


class StateStore:
    def __init__(self, path):
        self.path = Path(path)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS learner_state (user_id INTEGER PRIMARY KEY, state TEXT NOT NULL,"
                        " version INTEGER NOT NULL DEFAULT 1, updated_at REAL NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS applied_feedback_events ("
                        "user_id INTEGER NOT NULL, event_id TEXT NOT NULL, event_fingerprint TEXT, "
                        "task_id TEXT, applied_at REAL NOT NULL, "
                "PRIMARY KEY(user_id, event_id))")
        event_columns = {row[1] for row in self.db.execute("PRAGMA table_info(applied_feedback_events)")}
        if "event_fingerprint" not in event_columns:
            self.db.execute("ALTER TABLE applied_feedback_events ADD COLUMN event_fingerprint TEXT")
        if "task_id" not in event_columns:
            self.db.execute("ALTER TABLE applied_feedback_events ADD COLUMN task_id TEXT")
        self.db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS applied_feedback_task_once "
            "ON applied_feedback_events(user_id, task_id) WHERE task_id IS NOT NULL"
        )
        self.db.commit()

    def bulk_put(self, states: dict[int, LearnerState]):
        now = time.time()
        with self.db:
            self.db.executemany(
                "INSERT INTO learner_state(user_id,state,version,updated_at) VALUES(?,?,1,?) "
                "ON CONFLICT(user_id) DO UPDATE SET state=excluded.state, version=version+1, updated_at=excluded.updated_at",
                [(int(u), s.to_json(), now) for u, s in states.items()])

    def initialize_from_snapshot(self, snapshot_path: str | Path) -> int:
        """Copy offline baseline states into an empty live store exactly once."""
        source_path = Path(snapshot_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Learner state snapshot not found: {source_path}")
        if source_path.resolve() == self.path.resolve():
            raise ValueError("Live learner state store and snapshot must be different files")
        if self.count():
            raise RuntimeError("Cannot initialize a learner state store that already contains data")

        source = sqlite3.connect(str(source_path))
        try:
            rows = source.execute(
                "SELECT user_id,state,version,updated_at FROM learner_state"
            ).fetchall()
        finally:
            source.close()
        with self.db:
            self.db.executemany(
                "INSERT INTO learner_state(user_id,state,version,updated_at) VALUES(?,?,?,?)",
                rows,
            )
        return len(rows)

    def get(self, user_id: int) -> LearnerState | None:
        row = self.db.execute("SELECT state FROM learner_state WHERE user_id=?", (int(user_id),)).fetchone()
        return LearnerState.from_json(row[0]) if row else None

    def apply_event(self, user_id: int, skill_id: int, score: float, rt_ms: float, alpha: float, prior: float, **kw) -> LearnerState:
        """Atomic read-modify-write used by the online feedback loop (Layer 6)."""
        with self.db:                                       # one transaction
            st = self.get(user_id) or LearnerState(int(user_id))
            st.update(skill_id, score, rt_ms, alpha, prior, **kw)
            self.db.execute(
                "INSERT INTO learner_state(user_id,state,version,updated_at) VALUES(?,?,1,?) "
                "ON CONFLICT(user_id) DO UPDATE SET state=excluded.state, version=version+1, updated_at=excluded.updated_at",
                (int(user_id), st.to_json(), time.time()))
        return st

    def apply_canonical_event(self, user_id: int, skill_tag: str, score: float, rt_ms: float,
                              alpha: float, prior: float = 0.5, **kw) -> LearnerState:
        """Persist a graded interaction once, without guessing a source-specific skill ID."""
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            event_id = kw.pop("event_id", None)
            event_fingerprint = kw.pop("event_fingerprint", None)
            if event_id is not None:
                if not isinstance(event_id, str) or not event_id:
                    raise ValueError("event_id must be a non-empty string")
                existing_event = self.db.execute(
                    "SELECT event_fingerprint,task_id FROM applied_feedback_events WHERE user_id=? AND event_id=?",
                    (int(user_id), event_id),
                ).fetchone()
                if existing_event:
                    if existing_event[0] != event_fingerprint or existing_event[1] != kw.get("task_id"):
                        raise ValueError("event_id was already used with a different submission")
                    existing = self.get(user_id)
                    return existing or LearnerState(int(user_id))
                task_id = kw.get("task_id")
                if task_id is not None:
                    if not isinstance(task_id, str) or not task_id:
                        raise ValueError("task_id must be a non-empty string")
                    applied_task = self.db.execute(
                        "SELECT 1 FROM applied_feedback_events WHERE user_id=? AND task_id=?",
                        (int(user_id), task_id),
                    ).fetchone()
                    if applied_task:
                        raise ValueError("task was already applied to learner state")
                self.db.execute(
                    "INSERT INTO applied_feedback_events(user_id,event_id,event_fingerprint,task_id,applied_at) VALUES(?,?,?,?,?)",
                    (int(user_id), event_id, event_fingerprint, kw.get("task_id"), time.time()),
                )
            st = self.get(user_id) or LearnerState(int(user_id))
            st.update_canonical(skill_tag, score, rt_ms, alpha, prior, **kw)
            self.db.execute(
                "INSERT INTO learner_state(user_id,state,version,updated_at) VALUES(?,?,1,?) "
                "ON CONFLICT(user_id) DO UPDATE SET state=excluded.state, version=version+1, updated_at=excluded.updated_at",
                (int(user_id), st.to_json(), time.time()))
        return st

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM learner_state").fetchone()[0]

    def close(self):
        self.db.close()
