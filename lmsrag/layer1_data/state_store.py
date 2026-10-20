"""LAYER 1 / state management: persistent per-learner state in SQLite (stdlib, transactional, versioned rows)."""
from __future__ import annotations

import sqlite3
import time

from .learner_state import LearnerState


class StateStore:
    def __init__(self, path):
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

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM learner_state").fetchone()[0]

    def close(self):
        self.db.close()
