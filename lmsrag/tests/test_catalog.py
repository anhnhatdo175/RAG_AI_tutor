import pandas as pd

from lmsrag.layer1_data.catalog import build_skill_catalog
from lmsrag.layer1_data.sources import filter_exercise_bank
from lmsrag.skill_mapping import SkillMapper


def test_skill_catalog_maps_source_ids_to_canonical_tags():
    df = pd.DataFrame([
        {"skill_id": 10, "skill_name": "Percent Of"},
        {"skill_id": 11, "skill_name": "Percents"},
        {"skill_id": 12, "skill_name": "Venn Diagram"},
    ])

    catalog = build_skill_catalog(df, SkillMapper())
    mappings = {row["source_skill_id"]: row for row in catalog["source_skill_mappings"]}

    assert mappings[10]["canonical_skill_tag"] == "Percent-Of"
    assert mappings[11]["canonical_skill_tag"] == "Percent-Of"
    assert mappings[11]["alias_of"] == "Percent Of"
    assert mappings[12]["canonical_skill_tag"] is None
    assert mappings[12]["mapping_status"] == "unmapped"


def test_skill_mapper_normalizes_source_labels_and_applies_specific_rules():
    mapper = SkillMapper()

    assert mapper.to_tag(" Circumference  ") == "Circumference"
    assert mapper.to_tag("sLoPe") == "Finding-Slope-From-Equation"
    assert mapper.to_tag("Parts of a Polyomial, Terms, Coefficient, Monomial, Exponent, Variable") == "Polynomial-Vocabulary"
    assert mapper.tag_text("Find the x-intercepts of the line.", "Algebra", 2, False)[0] == "Intercept"
    assert mapper.tag_text("Convert 12 inches to feet.", "Prealgebra", 2, False)[0] == "Unit-Conversion-Within-a-System"
    assert mapper.tag_text("Find the coefficient of x in this polynomial.", "Algebra", 2, False)[0] == "Polynomial-Vocabulary"


def test_unmapped_source_skills_have_explanations():
    exported = SkillMapper().export(["Venn Diagram", "Reading a Ruler or Scale", "Unknown Skill"])
    catalog = build_skill_catalog(
        pd.DataFrame([{"skill_id": 1, "skill_name": "Pattern Finding "}]),
        SkillMapper(),
    )

    assert exported["assistments"]["Venn Diagram"]["reason"].startswith("diagram/visual")
    assert "visual scale" in exported["assistments"]["Reading a Ruler or Scale"]["reason"]
    assert exported["assistments"]["Unknown Skill"]["reason"] == "no rule written"
    assert catalog["source_skill_mappings"][0]["reason"].startswith("too generic")


def test_exercise_bank_filter_matches_exact_subject_grade_and_skill():
    bank = [
        {"item_id": "T-1", "subject": "Mathematics", "grade": "lower_secondary", "skill_tag": "fractions"},
        {"item_id": "T-2", "subject": "Mathematics", "grade": "upper_secondary", "skill_tag": "fractions"},
        {"item_id": "T-3", "subject": "English", "grade": "lower_secondary", "skill_tag": "fractions"},
    ]

    result = filter_exercise_bank(bank, "Mathematics", "lower_secondary", "fractions")
    assert [item["item_id"] for item in result] == ["T-1"]
