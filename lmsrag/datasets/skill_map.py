"""Read the skill dictionary v1 (``skill_map_v1.csv``, card P1-05) for the ASSISTments adapters.

The dictionary is the join key between log skills and the corpus: a skill's ``canonical_tag`` is written to its log
rows, and an empty tag means the skill is unmapped (its rows are dropped, with the dictionary's ``note`` as reason).
Skills absent from the dictionary fall back to ``SkillMapper``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..skill_mapping import SkillMapper


def skill_key(name: str) -> str:
    """Case- and whitespace-insensitive key: the 2009 log has labels such as 'Circumference ' (trailing space)."""
    return " ".join(str(name).split()).casefold()


class MapTagger:
    """``to_tag`` and ``unmapped_reason`` with the dictionary first and ``SkillMapper`` as fallback."""

    def __init__(self, path: Path, mapper: SkillMapper | None = None):
        self.mapper = mapper or SkillMapper()
        m = pd.read_csv(path, dtype=str, keep_default_na=False)
        m = m[m["source_skill"] != ""]
        self.tags: dict[str, str | None] = {}
        self.notes: dict[str, str] = {}
        for skill, rows in m.groupby(m["source_skill"].map(skill_key)):
            tags = sorted(set(rows["canonical_tag"]) - {""})
            if len(tags) > 1:
                raise ValueError(f"{path.name}: skill {skill!r} has several canonical tags {tags}")
            self.tags[skill] = tags[0] if tags else None
            self.notes[skill] = next((n for n in rows["note"] if n), "")

    def to_tag(self, log_skill: str) -> str | None:
        key = skill_key(log_skill)
        return self.tags[key] if key in self.tags else self.mapper.to_tag(log_skill)

    def unmapped_reason(self, log_skill: str) -> str:
        key = skill_key(log_skill)
        return self.notes.get(key) or self.mapper.unmapped_reason(log_skill)


def tagger(cfg_dataset: dict, repo: Path, mapper: SkillMapper | None = None) -> tuple[MapTagger | SkillMapper, str]:
    """The tagger an adapter should use, and its name for the cleaning log."""
    path = cfg_dataset.get("skill_map")
    if path and (repo / path).is_file():
        return MapTagger(repo / path, mapper), Path(path).name
    return mapper or SkillMapper(), "SkillMapper"
