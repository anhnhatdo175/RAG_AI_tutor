"""Export a non-student-level manifest to request exact ASSISTments problem text.

Usage: ``python -m lmsrag.data_tools.assistments_problem_manifest``
The result contains item IDs and aggregate skill coverage only; it excludes learner IDs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .. import config
from ..data import read_assistments
from ..skill_mapping import SkillMapper


def _joined_values(values: pd.Series) -> str:
    return " | ".join(sorted({str(value).strip() for value in values if pd.notna(value)}))


def export_manifest(output: Path | None = None) -> pd.DataFrame:
    frame = read_assistments()
    required = {"problem_id", "assistment_id", "template_id", "skill_id", "skill_name", "user_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ASSISTments log is missing columns needed for the manifest: {sorted(missing)}")
    mapper = SkillMapper()
    frame["canonical_skill_tag"] = frame.skill_name.map(mapper.to_tag).fillna("")
    frame["problem_id"] = frame.problem_id.astype("string").str.strip()
    frame = frame[frame.problem_id.notna() & frame.problem_id.ne("")]

    rows = []
    for problem_id, group in frame.groupby("problem_id", sort=True):
        rows.append({
            "problem_id": str(problem_id),
            "assistment_ids": _joined_values(group.assistment_id),
            "template_ids": _joined_values(group.template_id),
            "skill_ids": _joined_values(group.skill_id),
            "skill_names": _joined_values(group.skill_name),
            "canonical_skill_tags": _joined_values(group.canonical_skill_tag),
            "logged_interactions": int(len(group)),
            "distinct_learners": int(group.user_id.nunique()),
            "problem_text": "",
            "body_source_or_url": "",
            "review_status": "needs_official_problem_body",
        })
    result = pd.DataFrame(rows)
    if result.problem_id.duplicated().any():
        raise AssertionError("Manifest must contain exactly one row per problem_id")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False, encoding="utf-8-sig")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=config.ART / "assistments_problem_body_manifest.csv")
    args = parser.parse_args(argv)
    manifest = export_manifest(args.output)
    print(f"Wrote {len(manifest):,} unique ASSISTments problem IDs to {args.output}")
    print(f"Mapped canonical-skill rows: {(manifest.canonical_skill_tags.str.len() > 0).sum():,}")
    print(f"Rows without a canonical-skill tag: {(manifest.canonical_skill_tags.str.len() == 0).sum():,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
