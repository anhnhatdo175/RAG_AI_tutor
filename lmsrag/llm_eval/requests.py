"""Build the frozen P3-02 request set from the approved skill map."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from lmsrag import settings
from lmsrag.labels.rubric import target_level
from lmsrag.skill_mapping import SkillMapper

_REQUIRED_CONFIG = {"skill_count", "dev_skill_count", "mastery_levels", "modes", "skill_map", "output_dir"}


def eligible_skill_tags(skill_map: Path) -> list[str]:
    """Return canonical skills with at least one usable, mapped Khan template."""
    tags: set[str] = set()
    with skill_map.open("r", encoding="utf-8-sig", newline="") as source:
        for row_number, row in enumerate(csv.DictReader(source), start=2):
            tag = (row.get("canonical_tag") or "").strip()
            if not tag or tag.startswith(("khan:", "junyi:")):
                continue
            if not (row.get("source_dataset") or "").strip():
                continue
            if (row.get("method") or "").strip().casefold() == "template_only":
                continue
            if (row.get("template_status") or "").strip().casefold() != "usable":
                continue
            usable_count = int((row.get("usable_problems") or "0").strip())
            if usable_count < 0:
                raise ValueError(f"{skill_map}:{row_number}: usable_problems must be non-negative")
            if usable_count:
                tags.add(tag)
    if not tags:
        raise ValueError(f"No mapped canonical skills have usable Khan templates in {skill_map}")
    return sorted(tags)


def _validate_config(config: dict[str, Any]) -> tuple[int, int, tuple[float, ...], tuple[str, ...]]:
    missing = _REQUIRED_CONFIG - config.keys()
    if missing:
        raise ValueError(f"Missing llm_eval configuration keys: {sorted(missing)}")

    skill_count = config["skill_count"]
    dev_skill_count = config["dev_skill_count"]
    if (isinstance(skill_count, bool) or not isinstance(skill_count, int) or skill_count < 1
            or isinstance(dev_skill_count, bool) or not isinstance(dev_skill_count, int)
            or not 1 <= dev_skill_count < skill_count):
        raise ValueError("skill_count and dev_skill_count must be positive, with dev_skill_count < skill_count")

    mastery_levels = config["mastery_levels"]
    if not isinstance(mastery_levels, list) or not mastery_levels:
        raise ValueError("mastery_levels must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in mastery_levels):
        raise ValueError("mastery_levels must contain finite values in [0, 1]")
    if len(set(mastery_levels)) != len(mastery_levels):
        raise ValueError("mastery_levels must not contain duplicates")

    modes = config["modes"]
    if not isinstance(modes, list) or not modes or any(
        mode not in {"practice", "socratic"} for mode in modes
    ) or len(set(modes)) != len(modes):
        raise ValueError("modes must be a non-empty list of unique practice/socratic values")
    normalized_mastery = tuple(float(value) for value in mastery_levels)
    normalized_modes = tuple(modes)
    if skill_count != 60 or dev_skill_count != 10:
        raise ValueError("P3-02 requires exactly 60 skills with 10 dev skills")
    if normalized_mastery != (0.3, 0.55, 0.8) or normalized_modes != ("practice", "socratic"):
        raise ValueError("P3-02 requires mastery levels 0.3/0.55/0.8 and practice/socratic modes")
    return skill_count, dev_skill_count, normalized_mastery, normalized_modes


def _learner_profile(skill_tag: str, mastery: float) -> dict[str, Any]:
    weak = ([{"skill_tag": skill_tag, "mastery": mastery}] if mastery < 0.5 else [])
    strong = ([{"skill_tag": skill_tag, "mastery": mastery}] if mastery >= 0.8 else [])
    return {
        "overall_accuracy": mastery,
        "consistency_sigma": 0.0,
        "pace_per_100": 0.0,
        "mean_response_time_seconds": 30.0,
        "canonical_mastery": {skill_tag: mastery},
        "weak_canonical_skills": weak,
        "strong_canonical_skills": strong,
    }


def _common_error(skill_tag: str) -> str:
    tag = skill_tag.casefold()
    if "addition-and-subtraction-fractions" in tag:
        return "I added the numerators and denominators separately instead of finding a common denominator."
    if "fraction" in tag:
        return "I treated the numerator and denominator as separate whole numbers and used the wrong operation."
    if "percent" in tag:
        return "I used the percent as a whole number instead of converting it to a proportion out of 100."
    if "integer" in tag:
        return "I ignored the negative sign and calculated as if both values were positive."
    if "decimal" in tag:
        return "I lined up the digits rather than the decimal points."
    if "equation" in tag or "solving" in tag or "inequalit" in tag:
        return "I moved a term across the equals sign without changing its sign."
    if "square-root" in tag or "exponent" in tag:
        return "I applied the operation to only one part of the expression and did not check the exponent or root rule."
    if "area" in tag or "circumference" in tag or "perimeter" in tag or "volume" in tag:
        return "I chose a formula for a different measurement than the one the question asks for."
    if "mean" in tag or "median" in tag or "mode" in tag:
        return "I used the rule for a different measure of center."
    if "slope" in tag or "proportion" in tag or "linear" in tag:
        return "I reversed the order of the quantities in the ratio."
    if "probability" in tag:
        return "I counted favorable outcomes but forgot to divide by the total number of possible outcomes."
    return "I chose a reasonable operation but applied it to the wrong values from the question."


def _student_turns(skill_tag: str, skill_name: str) -> list[dict[str, Any]]:
    return [
        {
            "turn_index": 1,
            "scenario": "common_error",
            "student_message": _common_error(skill_tag),
        },
        {
            "turn_index": 2,
            "scenario": "stuck",
            "student_message": "I do not know where to start.",
        },
        {
            "turn_index": 3,
            "scenario": "correct_intermediate_step",
            "student_message": (
                f"I identified the relevant quantities and wrote the initial relationship for {skill_name}. "
                "What should I do next?"
            ),
        },
    ]


def build_requests(
    skill_tags: list[str],
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    """Build requests with a skill-level, seeded dev/eval split."""
    skill_count, dev_skill_count, mastery_levels, modes = _validate_config(config)
    unique_tags = sorted(set(skill_tags))
    if len(unique_tags) < skill_count:
        raise ValueError(f"Need {skill_count} eligible skills; found {len(unique_tags)}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    rng = random.Random(seed)
    selected = rng.sample(unique_tags, skill_count)
    rng.shuffle(selected)
    split_by_skill = {
        **{tag: "dev" for tag in selected[:dev_skill_count]},
        **{tag: "eval" for tag in selected[dev_skill_count:]},
    }
    mapper = SkillMapper()

    requests: list[dict[str, Any]] = []
    for skill_tag in selected:
        skill_name = mapper.topic_label(skill_tag)
        split = split_by_skill[skill_tag]
        for mastery in mastery_levels:
            for mode in modes:
                request_id = f"{split}-{skill_tag}-m{round(mastery * 100):02d}-{mode}"
                request: dict[str, Any] = {
                    "request_id": request_id,
                    "split": split,
                    "skill_tag": skill_tag,
                    "mastery": mastery,
                    "target_level": target_level(mastery),
                    "mode": mode,
                    "learner_profile": _learner_profile(skill_tag, mastery),
                }
                if mode == "socratic":
                    request["student_turns"] = _student_turns(skill_tag, skill_name)
                requests.append(request)
    return requests


def write_requests(requests: list[dict[str, Any]], output_path: Path) -> bool:
    """Write a deterministic JSONL file; refuse to silently change a frozen set."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for request in requests
    )
    if output_path.exists():
        existing = output_path.read_text(encoding="utf-8")
        if existing != content:
            raise FileExistsError(f"{output_path} already exists with different contents")
        return False
    output_path.write_text(content, encoding="utf-8", newline="\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="llm_eval", help="YAML file under configs/")
    args = parser.parse_args()

    config = settings.load(args.config)
    requests = build_requests(
        eligible_skill_tags(config["skill_map"]),
        config,
        settings.data_seed(),
    )
    output_path = Path(config["output_dir"]) / "requests_v1.jsonl"
    action = "Created" if write_requests(requests, output_path) else "Verified"
    print(f"{action} {len(requests)} requests ({output_path})")


if __name__ == "__main__":
    main()
