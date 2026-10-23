"""
LAYER 1 / sources (docx 2.2.1):  document corpus D, exercise bank T, interaction log.

  D  - worked solutions / lessons  -> indexed by Layer 2 (see layer2_embedding.build)
  T  - exercise bank: practice items with gold answer, skill_tag and difficulty (built here from the SAME real records)
  L  - interaction log (ASSISTments)
A registry records provenance (url, licence, sha256, row counts) so every number in a report can be traced to a file.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .. import config
from ..layer2_embedding import clean, parse
from ..skill_mapping import SkillMapper

PROVENANCE = {
    "gsm8k_train.jsonl": dict(url="https://github.com/openai/grade-school-math", licence="MIT"),
    "gsm8k_test.jsonl": dict(url="https://github.com/openai/grade-school-math", licence="MIT"),
    "math_test.jsonl": dict(url="https://github.com/hendrycks/math (copy: QwenLM/Qwen2.5-Math evaluation data)", licence="MIT"),
    "skill_builder_data.csv": dict(url="https://sites.google.com/site/assistmentsdata/ (2009-2010 skill builder)",
                                   licence="academic research use - verify the original terms before redistributing"),
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def source_registry() -> dict:
    files = [*config.GSM8K_FILES, *config.MATH_FILES, config.ASSISTMENTS_CSV]
    reg = {}
    for f in files:
        if f.exists():
            n = sum(1 for _ in f.open("rb"))
            reg[f.name] = {"path": str(f.relative_to(config.REPO)), "bytes": f.stat().st_size, "lines": n, "sha256": _sha(f),
                           **PROVENANCE.get(f.name, {})}
    return reg


import re

# a task shown to a learner must be answerable from its text alone
NEEDS_VISUAL = re.compile(r"\b(table|chart|graph|figure|diagram|picture|histogram|plot|shown|illustrated|below|above|following)\b", re.I)


def build_exercise_bank(mapper: SkillMapper) -> list[dict]:
    """Build the stable exercise catalog linked to D through doc_id and chunk `<doc_id>#0`."""
    recs, _ = clean.clean_records(parse.iter_records())
    bank = []
    for r in recs:
        if not r.answer or r.content_type != "worked_example" or NEEDS_VISUAL.search(r.question):
            continue
        tag, label, conf = mapper.tag_text(r.question.lower(), r.subject, r.level, r.is_gsm8k)
        bank.append(dict(item_id=f"T-{r.doc_id}", doc_id=r.doc_id, statement=r.question, gold_answer=r.answer,
                         canonical_skill_id=tag, skill_tag=tag or "unmapped", topic_label=label or "unmapped",
                         tag_confidence=conf or "none", mapping_status="mapped" if tag else "unmapped",
                         difficulty=r.difficulty, source=r.source, subject=parse.infer_subject(r),
                         source_subject=r.subject, grade=parse.infer_grade(r), level=r.level))
    item_ids = [item["item_id"] for item in bank]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Exercise catalog contains duplicate item_id values")
    return bank


def filter_exercise_bank(bank: list[dict], target_subject: str, target_grade: str,
                         skill_tag: str | None = None) -> list[dict]:
    """Return mapped exercise candidates matching subject/grade and optional skill."""
    if not target_subject or not target_grade:
        raise ValueError("target_subject and target_grade must be non-empty")
    return [
        item for item in bank
        if item.get("mapping_status", "mapped") == "mapped"
        and item.get("skill_tag") not in (None, "", "unmapped")
        and item.get("subject") == target_subject
        and item.get("grade") == target_grade
        and (skill_tag is None or item.get("skill_tag") == skill_tag)
    ]


def save_bank(bank: list[dict]) -> dict:
    path = config.ART / "exercise_bank.jsonl"
    path.write_text("\n".join(json.dumps(b, ensure_ascii=False) for b in bank), encoding="utf-8")
    mapped = [b for b in bank if b["skill_tag"] != "unmapped"]
    return {"n_items": len(bank), "n_mapped": len(mapped),
            "by_difficulty": {str(k): v for k, v in sorted(Counter(b["difficulty"] for b in bank).items())},
            "by_source": dict(Counter(b["source"] for b in bank)),
            "mapped_by_skill_top10": dict(Counter(b["skill_tag"] for b in mapped).most_common(10))}
