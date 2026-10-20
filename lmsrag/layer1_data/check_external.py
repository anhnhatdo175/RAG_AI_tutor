"""
Sanity-check REAL Junyi / EdNet files before running the pipeline on them.

    python -m lmsrag.layer1_data.check_external junyi <folder with junyi_ProblemLog_original.csv>
    python -m lmsrag.layer1_data.check_external ednet <folder with KT1/ and questions.csv>

Prints: columns found, rows, learners, how many events survive the exercise->corpus join (Junyi), correct rate, split sizes.
Paste the output back so the adapters can be adjusted to the real schema.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .. import config
from ..skill_mapping import SkillMapper


def main(kind: str, root: str, nrows: int | None = 2_000_000):
    mapper = SkillMapper()
    if kind == "junyi":
        import pandas as pd
        from . import junyi
        log = next(iter(sorted(Path(root).rglob("*ProblemLog*.csv"))), None)
        print("problem log:", log)
        if log is not None:
            print("columns:", list(pd.read_csv(log, nrows=2).columns))
        ex2tag = junyi.exercise_to_tag()
        print(f"corpus exercises with a Khan counterpart: {len(ex2tag)}  (tags: {len(set(ex2tag.values()))})")
        raw_ex = pd.read_csv(log, usecols=["exercise"], nrows=nrows).exercise
        names = set(raw_ex.unique())
        print(f"Junyi exercises in log: {len(names)} | joined to corpus: {len(names & set(ex2tag))} | events joined: {raw_ex.isin(ex2tag).mean():.1%}")
        df = junyi.load_junyi(mapper, root, nrows=nrows)
    elif kind == "ednet":
        from . import ednet
        df = ednet.load_ednet(mapper, root, max_users=2000)
        print("NOTE: no TOEIC corpus -> tag is None for all skills; only state / EMA tracking can be evaluated.")
    else:
        raise SystemExit("kind must be junyi or ednet")
    print(f"rows {len(df):,} | learners {df.user_id.nunique():,} | skills {df.skill_id.nunique()} | correct rate {df.correct.mean():.3f}")
    print("split:", df.split.value_counts().to_dict(), "| tagged events:", f"{df.tag.notna().mean():.1%}", "| item difficulty known:", f"{df.item_known.mean():.1%}")
    ok = df[df.tag.notna()].groupby("tag").problem_id.nunique().sort_values(ascending=False)
    print("items (exercises) per tag:", ok.head(12).to_dict())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2_000_000)
