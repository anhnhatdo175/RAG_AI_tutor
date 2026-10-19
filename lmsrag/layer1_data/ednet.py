"""
LAYER 1 / source 3 - EdNet-KT1 (docx 2.8.1).   Raw EdNet files -> the common log schema of ``data.finalize_logs``.

Files: KT1/u<id>.csv  (timestamp [ms], solving_id, question_id, user_answer, elapsed_time [ms])
       contents/questions.csv (question_id, bundle_id, explanation_id, correct_answer, part, tags, deployed_at)

  correct   = user_answer == correct_answer
  skill     = first tag of the question (EdNet tags are numeric ids with no public names); ``part`` (TOEIC part 1-7) is the fallback
  problem   = question_id

!! EdNet is TOEIC (English).  GSM8K / MATH cannot be its corpus.  Until a TOEIC grammar corpus is added (docx 2.8.1 says
   EdNet is a benchmark dataset, so the corpus has to be built or licensed), every EdNet skill maps to tag None:
   the log loads, the learner state P(u) and the EMA tracking (docx 2.9.4) work, retrieval/ZPD evaluation (2.9.3) does not.
   ``ednet_skill_names`` lets you supply a {tag id -> SkillMapper name} table once such a corpus exists.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data import finalize_logs
from ..skill_mapping import SkillMapper


def read_ednet(root: Path, max_users: int | None = None, ednet_skill_names: dict[str, str] | None = None) -> pd.DataFrame:
    root = Path(root)
    qpath = next(iter(sorted(root.rglob("questions.csv"))), None)
    if qpath is None:
        raise FileNotFoundError(f"no questions.csv under {root} (EdNet content file: bit.ly/ednet-content)")
    q = pd.read_csv(qpath)
    for c in ("question_id", "correct_answer", "part", "tags"):
        if c not in q:
            raise KeyError(f"questions.csv lacks '{c}'; columns: {list(q.columns)}")
    q["skill_key"] = q.tags.fillna("").astype(str).str.split(";").str[0].replace("", np.nan).fillna("part" + q.part.astype(str))
    qmap = q.set_index("question_id")[["correct_answer", "skill_key"]]
    files = sorted(p for p in root.rglob("u*.csv") if p.stem[1:].isdigit())
    if max_users:
        files = files[:max_users]
    if not files:
        raise FileNotFoundError(f"no KT1 user files (u<id>.csv) under {root}")
    parts = []
    for f in files:
        d = pd.read_csv(f)
        d["user_id"] = int(f.stem[1:]); parts.append(d)
    df = pd.concat(parts, ignore_index=True).join(qmap, on="question_id")
    df = df[df.correct_answer.notna()].copy()
    df["correct"] = (df.user_answer == df.correct_answer).astype(int)
    df["ms"] = pd.to_numeric(df.elapsed_time, errors="coerce").fillna(10_000).clip(500, 120_000)
    df["order_id"] = df.timestamp.astype("int64")
    keys = sorted(df.skill_key.unique()); sid = {k: i + 1 for i, k in enumerate(keys)}
    df["skill_id"] = df.skill_key.map(sid).astype(int)
    names = ednet_skill_names or {}
    df["skill_name"] = df.skill_key.map(lambda k: names.get(k, f"ednet:{k}"))      # unknown -> mapper.to_tag gives None
    df["problem_id"] = df.question_id
    return df[["order_id", "user_id", "problem_id", "correct", "skill_id", "skill_name", "ms"]]


def load_ednet(mapper: SkillMapper, root: Path, min_user_rows: int = 20, max_users: int | None = None, ednet_skill_names: dict[str, str] | None = None) -> pd.DataFrame:
    return finalize_logs(read_ednet(root, max_users, ednet_skill_names), mapper, min_user_rows)
