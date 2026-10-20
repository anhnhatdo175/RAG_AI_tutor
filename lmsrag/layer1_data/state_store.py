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
        """Persist a graded interaction without guessing a source-specific skill ID."""
        with self.db:
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
