"""Build reproducible learner profiles and a synthetic ASSISTments knowledge base.

The source file is the official 2009-2010 ASSISTments Skill Builder export.
This script intentionally creates a small, auditable study slice:

* five real learners with enough observations and four skills in common;
* skill-level profiles from the selected learners' observed attempts;
* multiple synthetic learning documents for every skill/difficulty pair;
* a manifest linking documents to the learners for whom each difficulty is
  recommended.

The generated material is a deterministic scaffold for retrieval experiments.
It must be reviewed by a subject-matter expert before being presented as
authoritative instructional content.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_USERS = ["78523", "78561", "78571", "78544", "78557"]
DEFAULT_MIN_ROWS = 100
DEFAULT_MIN_COMMON_SKILLS = 6
DEFAULT_MIN_ATTEMPTS_PER_COMMON_SKILL = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("skill_builder_data.csv"),
    )
    parser.add_argument(
        "--users",
        default=",".join(DEFAULT_USERS),
        help="Five comma-separated ASSISTments user_id values.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=DEFAULT_MIN_ROWS,
        help="Minimum valid skill-tagged attempts per selected user.",
    )
    parser.add_argument(
        "--min-attempts-per-skill",
        type=int,
        default=DEFAULT_MIN_ATTEMPTS_PER_COMMON_SKILL,
        help="Minimum attempts per common skill for every selected user.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent,
    )
    return parser.parse_args()


def difficulty_for_accuracy(accuracy: float) -> tuple[str, str]:
    if accuracy < 0.55:
        return "beginner", "struggling"
    if accuracy < 0.78:
        return "intermediate", "developing"
    return "advanced", "proficient"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def load_selected_rows(
    input_path: Path, selected_users: set[str]
) -> tuple[dict[str, dict[str, list[dict[str, str]]]], dict[str, str], dict[str, int]]:
    rows_by_user_skill: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    skill_names: dict[str, str] = {}
    counters = {"total_rows": 0, "missing_skill_id": 0, "selected_rows": 0}

    with input_path.open("r", encoding="latin-1", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "user_id",
            "skill_id",
            "skill_name",
            "correct",
            "attempt_count",
            "ms_first_response",
        }
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Missing required ASSISTments columns: {sorted(missing_columns)}"
            )

        for row in reader:
            counters["total_rows"] += 1
            user_id = row.get("user_id", "").strip()
            skill_id = row.get("skill_id", "").strip()
            if not skill_id:
                counters["missing_skill_id"] += 1
            if user_id not in selected_users or not skill_id:
                continue
            try:
                int(row["correct"])
                int(row["attempt_count"])
                int(row["ms_first_response"])
            except (TypeError, ValueError):
                continue
            counters["selected_rows"] += 1
            rows_by_user_skill[user_id][skill_id].append(row)
            skill_names.setdefault(skill_id, row.get("skill_name", "").strip())

    return rows_by_user_skill, skill_names, counters


def validate_selection(
    rows_by_user_skill: dict[str, dict[str, list[dict[str, str]]]],
    selected_users: list[str],
    min_rows: int,
    min_common_skills: int,
    min_attempts_per_skill: int,
) -> list[str]:
    missing = [user for user in selected_users if user not in rows_by_user_skill]
    if missing:
        raise ValueError(f"Selected users have no valid rows: {missing}")
    undersized = {
        user: sum(len(rows) for rows in rows_by_user_skill[user].values())
        for user in selected_users
    }
    undersized = {
        user: count for user, count in undersized.items() if count < min_rows
    }
    if undersized:
        raise ValueError(
            f"Selected users below --min-rows={min_rows}: {undersized}"
        )
    common = set.intersection(
        *(set(rows_by_user_skill[user]) for user in selected_users)
    )
    common = {
        skill_id
        for skill_id in common
        if min(
            len(rows_by_user_skill[user][skill_id]) for user in selected_users
        )
        >= min_attempts_per_skill
    }
    if len(common) < min_common_skills:
        raise ValueError(
            f"Only {len(common)} common skills with at least "
            f"{min_attempts_per_skill} attempts per user found; "
            f"need {min_common_skills}."
        )
    return sorted(common, key=int)


def build_profiles(
    rows_by_user_skill: dict[str, dict[str, list[dict[str, str]]]],
    selected_users: list[str],
    common_skills: list[str],
    skill_names: dict[str, str],
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for user_id in selected_users:
        skill_profiles: list[dict[str, Any]] = []
        for skill_id in common_skills:
            rows = rows_by_user_skill[user_id][skill_id]
            count = len(rows)
            correct = sum(int(row["correct"]) for row in rows)
            accuracy = correct / count
            avg_attempts = sum(int(row["attempt_count"]) for row in rows) / count
            avg_time_sec = (
                sum(int(row["ms_first_response"]) for row in rows) / count / 1000
            )
            difficulty, mastery = difficulty_for_accuracy(accuracy)
            skill_profiles.append(
                {
                    "skill_id": skill_id,
                    "skill_name": skill_names[skill_id],
                    "questions_attempted": count,
                    "correct_answers": correct,
                    "accuracy": round(accuracy, 4),
                    "avg_attempt_count": round(avg_attempts, 3),
                    "avg_time_sec": round(avg_time_sec, 3),
                    "mastery_level": mastery,
                    "recommended_material_difficulty": difficulty,
                }
            )
        profiles[user_id] = {
            "source": "ASSISTments 2009-2010 Skill Builder",
            "user_id": user_id,
            "skills": skill_profiles,
        }
    return profiles


CONTENT_TYPES = {
    "concept": "Concept explanation",
    "worked_example": "Worked example",
    "misconception": "Misconception and feedback",
    "strategy": "Problem-solving strategy",
    "transfer_practice": "Transfer practice",
}


def material_content(
    skill_id: str, skill_name: str, difficulty: str, content_type: str
) -> str:
    examples = {
        "70": {
            "concept": "Find a percentage of a quantity by converting the percent to a decimal or fraction and multiplying.",
            "example": "25% of 80 = 0.25 × 80 = 20.",
            "practice": "Calculate 15% of 60 and explain why the answer is smaller than 60.",
        },
        "297": {
            "concept": "The area of a trapezoid is one half of the sum of the parallel bases multiplied by the height.",
            "example": "For bases 8 and 12 and height 5, area = (8 + 12) × 5 / 2 = 50 square units.",
            "practice": "Find the area of a trapezoid with bases 6 and 10 and height 4.",
        },
        "303": {
            "concept": "The volume of a cylinder is pi times the radius squared times the height: V = pi r²h.",
            "example": "With radius 3 and height 5, V = pi × 3² × 5 = 45 pi cubic units.",
            "practice": "Write the exact volume of a cylinder with radius 2 and height 7.",
        },
        "307": {
            "concept": "The volume of a rectangular prism is length multiplied by width multiplied by height.",
            "example": "A prism with dimensions 4, 3, and 5 has volume 4 × 3 × 5 = 60 cubic units.",
            "practice": "Find the volume of a prism measuring 8 by 2 by 3 units.",
        },
        "11": {
            "concept": "Use a Venn diagram to represent sets, their intersection, their union, and elements that belong to only one set.",
            "example": "If 12 students like A, 9 like B, and 4 like both, then the union has 12 + 9 - 4 = 17 students.",
            "practice": "Draw two sets with 8 elements in A, 6 in B, and 3 in both. Find the union.",
        },
        "317": {
            "concept": "The greatest common factor (GCF) is the largest positive integer that divides each number without a remainder.",
            "example": "The factors common to 18 and 24 include 1, 2, 3, and 6, so GCF(18, 24) = 6.",
            "practice": "Find the GCF of 36 and 48 using a factor list or prime factorization.",
        },
    }
    item = examples.get(
        skill_id,
        {
            "concept": f"Apply the standard method for the ASSISTments skill {skill_name}.",
            "example": "Identify the known quantities, select the relevant formula, and verify the units.",
            "practice": "Solve a new problem for this skill and check the result.",
        },
    )
    level_guidance = {
        "beginner": "Use a worked example, name each quantity, and check units after every step.",
        "intermediate": "Solve a two-stage variation and compare the result with an estimate.",
        "advanced": "Handle an unfamiliar representation, justify the formula, and check reasonableness.",
    }[difficulty]
    checking_note = (
        "Check whether the answer has the correct units and is reasonable."
        if skill_id in {"297", "303", "307"}
        else "Check the result against the definition and a simple example."
    )
    key_points = {
        "11": (
            "Separate only-A, only-B, and both regions before adding.",
            "Subtract the intersection once when computing a union.",
        ),
        "70": (
            "Translate percent language before choosing an operation.",
            "Use a benchmark such as 10%, 1%, or 50% to estimate.",
        ),
        "297": (
            "The height is perpendicular to the parallel bases.",
            "The slanted side is not automatically the height.",
        ),
        "303": (
            "The radius is half the diameter and is squared.",
            "Keep pi symbolic when an exact answer is requested.",
        ),
        "307": (
            "Multiply the three dimensions once, in any order.",
            "Volume is measured in cubic units.",
        ),
        "317": (
            "A common factor must divide every number in the group.",
            "Prime factorization exposes shared factors systematically.",
        ),
    }.get(skill_id, ("Identify the given quantities.", "Check the result."))
    type_sections = {
        "concept": f"""## Core concept
{item["concept"]}

## Study checklist
1. Name the quantities or sets in the problem.
2. Select the definition or formula before substituting numbers.
3. Show intermediate steps and check the result.

## Key distinctions
- {key_points[0]}
- {key_points[1]}
""",
        "worked_example": f"""## Worked example
{item["example"]}

## Why the method works
Start from the definition of {skill_name.lower()}, substitute only known
quantities, and preserve the meaning of the units or set relationships.
{checking_note}
""",
        "misconception": f"""## Common misconception
A learner may choose a related operation without first identifying the
quantities in the question. For this skill, do not skip the representation:
{item["concept"]}

## Corrective feedback
Ask the learner to point to the step where the definition or formula was
applied, then redo that step with a smaller example.
""",
        "strategy": f"""## Problem-solving strategy
1. Restate what the question asks.
2. Mark the known quantities and the unknown.
3. Select the definition or formula: {item["concept"]}
4. Calculate and verify the result.

## Strategy reminders
- {key_points[0]}
- {key_points[1]}
""",
        "transfer_practice": f"""## Transfer task
{item["practice"]}

## Variation
Change one quantity or the representation and solve again. Explain which
step stays the same and which step changes.

## Self-check rubric
- Did you identify the correct quantities?
- Did you use the correct operation or formula?
- Did you show an intermediate step?
- {checking_note}
""",
    }
    return f"""# {skill_name} ({difficulty})

<!-- content_type: {content_type} -->
## Learning objective
Build reliable understanding of {skill_name.lower()} at the {difficulty} level.

## Level guidance
{level_guidance}

{type_sections[content_type]}

## Practice prompt
{item["practice"]}

## Retrieval and tutoring notes
- Explain the meaning of each variable before calculating.
- {checking_note}
- Ask the learner to show intermediate steps rather than returning only a number.
- If the learner makes an arithmetic error, revisit the corresponding step without
  changing the target skill.
"""


def write_materials(
    output: Path,
    profiles: dict[str, dict[str, Any]],
    common_skills: list[str],
    skill_names: dict[str, str],
) -> list[dict[str, Any]]:
    materials_dir = output / "materials"
    materials_dir.mkdir(parents=True, exist_ok=True)
    for existing_material in materials_dir.glob("assistments-skill-*.md"):
        existing_material.unlink()
    needed: dict[tuple[str, str], list[str]] = defaultdict(list)
    for user_id, profile in profiles.items():
        for skill in profile["skills"]:
            needed[(skill["skill_id"], skill["recommended_material_difficulty"])].append(
                user_id
            )

    manifest: list[dict[str, Any]] = []
    for skill_id in common_skills:
        skill_name = skill_names[skill_id]
        for difficulty in ("beginner", "intermediate", "advanced"):
            for content_type in CONTENT_TYPES:
                doc_id = (
                    f"assistments-skill-{skill_id}-{difficulty}-{content_type}"
                )
                filename = f"{doc_id}.md"
                frontmatter = {
                    "doc_id": doc_id,
                    "source": "synthetic_template_v3",
                    "source_dataset": "ASSISTments 2009-2010 Skill Builder",
                    "skill_id": skill_id,
                    "skill_name": skill_name,
                    "difficulty": difficulty,
                    "content_type": content_type,
                    "review_status": "needs_subject_matter_review",
                }
                content = "---\n" + yaml_like(frontmatter) + "---\n\n"
                content += material_content(
                    skill_id, skill_name, difficulty, content_type
                )
                (materials_dir / filename).write_text(content, encoding="utf-8")
                manifest.append(
                    {
                        **frontmatter,
                        "file": f"materials/{filename}",
                        "tags": [
                            f"skill-{skill_id}",
                            slug(skill_name),
                            difficulty,
                            content_type,
                        ],
                    }
                )
    return manifest


def yaml_like(values: dict[str, Any]) -> str:
    lines = []
    for key, value in values.items():
        if isinstance(value, list):
            lines.append(f"{key}: [{', '.join(json.dumps(str(item)) for item in value)}]")
        else:
            lines.append(f"{key}: {json.dumps(str(value), ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    selected_users = [value.strip() for value in args.users.split(",") if value.strip()]
    if len(selected_users) != 5 or len(set(selected_users)) != 5:
        raise ValueError("--users must contain exactly five distinct user IDs.")

    rows, skill_names, counters = load_selected_rows(args.input, set(selected_users))
    common_skills = validate_selection(
        rows,
        selected_users,
        args.min_rows,
        DEFAULT_MIN_COMMON_SKILLS,
        args.min_attempts_per_skill,
    )
    profiles = build_profiles(rows, selected_users, common_skills, skill_names)
    manifest = write_materials(args.output, profiles, common_skills, skill_names)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    selection = {
        "input": str(args.input.name),
        "selected_user_ids": selected_users,
        "common_skills": [
            {"skill_id": skill_id, "skill_name": skill_names[skill_id]}
            for skill_id in common_skills
        ],
        "data_quality": counters,
    }
    print(json.dumps(selection, indent=2, ensure_ascii=False))
    print(f"Generated {len(profiles)} profiles and {len(manifest)} materials.")


if __name__ == "__main__":
    main()
