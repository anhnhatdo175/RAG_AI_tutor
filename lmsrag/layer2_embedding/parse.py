"""
LAYER 2 / stage 1 - PARSE.  Raw files -> uniform ParsedRecord (no cleaning yet).

Supported: GSM8K jsonl, MATH jsonl, Khan-style jsonl ({"text", ...}), Markdown/plain-text lessons.
Every record keeps provenance (source, split, original id) so each chunk can be traced back to a file.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .. import config

DIFFICULTY_LEVELS = config.DIFFICULTY_LEVELS


@dataclass
class ParsedRecord:
    doc_id: str
    source: str                     # GSM8K | MATH | Khan_Academy_<x> | Lesson_<x>
    subject: str | None
    level: int | None               # MATH level 1-5, else None
    question: str
    solution: str
    answer: str
    difficulty: float               # 0.3 / 0.6 / 0.9  (basic / intermediate / advanced)
    is_gsm8k: bool = False
    content_type: str = "worked_example"
    extra: dict = field(default_factory=dict)


def math_difficulty(level: int) -> float:
    return DIFFICULTY_LEVELS[0] if level <= 2 else DIFFICULTY_LEVELS[1] if level == 3 else DIFFICULTY_LEVELS[2]


def gsm_difficulty(n_steps: int) -> float:
    return DIFFICULTY_LEVELS[0] if n_steps <= 2 else DIFFICULTY_LEVELS[1] if n_steps <= 4 else DIFFICULTY_LEVELS[2]


def infer_grade(r: "ParsedRecord") -> str:
    """Grade band stored in chunk metadata (docx 2.4.2).  Explicit `grade` in the record wins, then source/level rules."""
    g = (r.extra or {}).get("grade")
    if g:
        return str(g)
    if r.is_gsm8k:
        return config.GRADE_GSM8K
    if r.level is not None:
        return config.GRADE_BY_MATH_LEVEL.get(int(r.level), config.GRADE_DEFAULT)
    return config.GRADE_DEFAULT


def _lines(path: Path) -> Iterator[str]:
    # NOT str.splitlines(): it also splits on U+2028/U+0085 that occur inside JSON strings
    for ln in path.read_text(encoding="utf-8").split("\n"):
        if ln.strip():
            yield ln


def parse_gsm8k(path: Path) -> Iterator[ParsedRecord]:
    split = path.stem.split("_")[-1]
    for i, ln in enumerate(_lines(path)):
        r = json.loads(ln)
        body, _, final = r["answer"].partition("####")
        n_steps = body.count("<<")
        yield ParsedRecord(f"gsm8k-{split}-{i}", "GSM8K", "word_problem", None, r["question"], body, final.strip(),
                           gsm_difficulty(n_steps), is_gsm8k=True, extra={"split": split, "n_steps": n_steps})


def parse_math(path: Path) -> Iterator[ParsedRecord]:
    split = path.stem.split("_")[-1]
    for ln in _lines(path):
        r = json.loads(ln)
        lvl = int(str(r["level"]).split()[-1])
        uid = str(r.get("unique_id", "")).replace("/", "_").replace(".json", "") or f"{split}-{abs(hash(r['problem']))}"
        yield ParsedRecord(f"math-{uid}", "MATH", r["subject"], lvl, r["problem"], r["solution"], str(r.get("answer", "")),
                           math_difficulty(lvl), extra={"split": split})


def parse_khan_jsonl(path: Path) -> Iterator[ParsedRecord]:
    """{"text", "title"?, "difficulty"?, "content_type"? ("scaffold_rule" | "worked_example"), "subject"?,
        "solution"?, "answer"? (worked_example items, e.g. rendered khan-exercises)}"""
    for i, ln in enumerate(_lines(path)):
        r = json.loads(ln)
        yield ParsedRecord(f"khan-{path.stem}-{i}", f"Khan_Academy_{path.stem}", r.get("subject"), None, r["text"],
                           r.get("solution", ""), str(r.get("answer", "")),
                           float(r.get("difficulty", 0.6)), content_type=r.get("content_type", "scaffold_rule"),
                           extra={"title": r.get("title"), **r.get("extra", {})})


def parse_markdown(path: Path) -> Iterator[ParsedRecord]:
    """One record per top-level section (# / ##) so long lessons are split on their own structure first."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^(?=#{1,2}\s)", text)
    for i, part in enumerate(p for p in parts if p.strip()):
        title = part.splitlines()[0].lstrip("# ").strip()
        yield ParsedRecord(f"lesson-{path.stem}-{i}", f"Lesson_{path.stem}", None, None, part, "", "", 0.6,
                           content_type="scaffold_rule", extra={"title": title})


def iter_records() -> Iterator[ParsedRecord]:
    for f in config.GSM8K_FILES:
        if f.exists():
            yield from parse_gsm8k(f)
    for f in config.MATH_FILES:
        if f.exists():
            yield from parse_math(f)
    if config.KHAN_DIR.exists():
        for f in sorted(config.KHAN_DIR.glob("*.jsonl")):
            yield from parse_khan_jsonl(f)
        for f in sorted(config.KHAN_DIR.glob("*.md")):
            yield from parse_markdown(f)
