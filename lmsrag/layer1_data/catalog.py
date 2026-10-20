"""Stable skill catalog and source-skill mapping table."""
from __future__ import annotations

import json

import pandas as pd

from .. import config
from ..skill_mapping import SkillMapper, slugify


def build_skill_catalog(df: pd.DataFrame, mapper: SkillMapper) -> dict:
    source_skills = (
        df[["skill_id", "skill_name"]]
        .drop_duplicates()
        .sort_values(["skill_id", "skill_name"])
    )
    mappings = []
    for row in source_skills.itertuples(index=False):
        skill_id, skill_name = int(row.skill_id), str(row.skill_name)
        canonical_tag = mapper.to_tag(skill_name)
        canonical_name = mapper.canonical_name(skill_name)
        mappings.append({
            "source_system": "ASSISTments",
            "source_skill_id": skill_id,
            "source_skill_key": f"ASSISTments:{skill_id}",
            "source_skill_name": skill_name,
            "canonical_skill_id": canonical_tag,
            "canonical_skill_tag": canonical_tag,
            "topic_label": mapper.topic_label(canonical_tag) if canonical_tag else None,
            "mapping_status": "mapped" if canonical_tag else "unmapped",
            "confidence": mapper.confidence(skill_name),
            "alias_of": canonical_name if canonical_name != skill_name else None,
            "reason": None if canonical_tag else mapper.unmapped_reason(skill_name),
        })

    canonical_skills = [
        {
            "canonical_skill_id": slugify(name),
            "canonical_skill_tag": slugify(name),
            "canonical_name": name,
            "topic_label": rule["topic_label"],
            "subjects": rule["subjects"],
            "confidence": rule["confidence"],
            "aliases": sorted(rule["aliases"]),
        }
        for name, rule in sorted(mapper.canonical.items())
    ]
    return {
        "schema_version": 1,
        "canonical_skills": canonical_skills,
        "source_skill_mappings": mappings,
    }


def save_skill_catalog(catalog: dict) -> None:
    config.SKILL_CATALOG_JSON.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
