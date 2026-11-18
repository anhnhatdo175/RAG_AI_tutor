"""Build the skill dictionary v1 from the P1-05 review sheets (step 2, after Người 1 has filled them).

Inputs (``lmsrag/data/labels/``): the three ``a4_review_*.csv`` sheets produced by ``review_draft`` and filled by
hand. Outputs (``lmsrag/data/mappings/``):

* ``skill_map_v1.csv``: one row per accepted (ASSISTments skill, Khan template) pair, plus a row with a reason for
  every other skill of the two ASSISTments logs (``khan_slug`` empty), plus the templates with no ASSISTments skill
  (``source_skill`` empty, tag ``khan:<slug>``). ``canonical_tag`` is the tag the adapters write to log rows and the
  corpus writes to chunks; empty means the skill is unmapped. Columns follow card P1-05 (``source_dataset,
  source_skill, canonical_tag, khan_slug, method, confidence, note``), then provenance.
* ``skill_map_v1_rejected.csv``: the draft pairs the reviewer dropped (used to sample negatives for the κ sheets).

The script refuses to write anything when a sheet has an unknown decision, a slug that is not in
``khan_templates.csv`` or a skill name that is not in the ASSISTments logs.

    python -m lmsrag.labels.build_skill_map
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..datasets.skill_map import skill_key
from ..skill_mapping import CANONICAL, SkillMapper, slugify

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "lmsrag" / "data" / "labels"
MAPPINGS = REPO / "lmsrag" / "data" / "mappings"
KHAN = REPO / "lmsrag" / "data" / "khan_v2"
TABLES = REPO / "reports" / "tables"
DRAFT = MAPPINGS / "assistments_to_khan_draft.csv"

MAP_VERSION = "1.0"
DECISIONS = {"A": {"keep", "drop", "fix", "none"}, "B": {"add", "none"}, "C": {"keep", "add", "none"}}
# card P1-05 columns first, then provenance
COLUMNS = ["source_dataset", "source_skill", "canonical_tag", "khan_slug", "method", "confidence", "note",
           "template_status", "usable_problems", "junyi_responses", "junyi_accuracy", "review", "review_row",
           "map_version"]
# how each kind of row was decided (the κ audit reports accuracy per method)
METHOD = {"A:keep": "manual_review", "D:keep": "manual_review", "A:drop": "manual_review",
          "D:drop": "manual_review", "A:fix_replaced": "manual_fix", "A:fix": "manual_fix", "B:add": "manual_add", "C:add": "manual_add",
          "C:keep": "rule_confirmed", "A:none": "no_template", "B:none": "no_template",
          "A:all_dropped": "no_template", "C:none": "template_only", "unreviewed": "not_reviewed"}
DOUBT = ("phân vân", "phan van")


def read_sheet(path: Path) -> pd.DataFrame:
    """Read a review sheet as text; Excel sometimes saves it in the ANSI code page instead of UTF-8."""
    for enc in ("utf-8-sig", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"{path.name}: cannot decode")
    return df.apply(lambda s: s.str.strip())


def split_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(";") if v.strip()]


def rule_skill(tag: str) -> str | None:
    """'Polynomial-Factors' -> 'Polynomial Factors' (the canonical log skill whose tag it is)."""
    return next((name for name in CANONICAL if slugify(name) == tag), None)


def collect(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame) -> tuple[list[dict], list[dict], list[str]]:
    """Turn the three sheets into accepted rows, rejected pairs and a list of errors."""
    rows, rejected, errors = [], [], []

    def bad_decision(sheet, r):
        if r["decision"] not in DECISIONS[sheet]:
            errors.append(f"{sheet} row {r['row']}: decision {r['decision']!r} not in {sorted(DECISIONS[sheet])}")
            return True
        return False

    for _, r in a.iterrows():
        if bad_decision("A", r):
            continue
        base = {"source_skill": r["assistments_skill"], "origin": "draft_v0_manual", "review_row": int(r["row"]),
                "note": r["note"]}
        slug, d, sheet = r["khan_slug"], r["decision"], r.get("sheet") or "A"
        if not slug:  # no_khan_template row: drop and none both mean "no template"
            rows.append({**base, "khan_slug": "", "review": "A:none"})
        elif d == "keep":
            rows.append({**base, "khan_slug": slug, "review": f"{sheet}:keep"})
        elif d == "drop":
            rejected.append({**base, "khan_slug": slug, "review": f"{sheet}:drop"})
        elif d == "none":
            errors.append(f"A row {r['row']}: 'none' is for rows without a template; use drop")
        else:  # fix
            new = split_list(r["khan_slugs"])
            if not new:
                errors.append(f"A row {r['row']}: fix needs khan_slugs")
            if slug not in new:
                rejected.append({**base, "khan_slug": slug, "review": "A:fix_replaced"})
            rows += [{**base, "khan_slug": s, "review": "A:fix"} for s in new]

    for _, r in b.iterrows():
        if bad_decision("B", r):
            continue
        base = {"source_skill": r["assistments_skill"], "origin": "review_B", "review_row": int(r["row"]),
                "note": r["note"]}
        if r["decision"] == "add":
            new = split_list(r["khan_slugs"])
            if not new:
                errors.append(f"B row {r['row']}: add needs khan_slugs")
            rows += [{**base, "khan_slug": s, "review": "B:add"} for s in new]
        else:
            rows.append({**base, "khan_slug": "", "review": "B:none"})

    for _, r in c.iterrows():
        if bad_decision("C", r):
            continue
        base = {"khan_slug": r["khan_slug"], "origin": "review_C", "review_row": int(r["row"]), "note": r["note"]}
        if r["decision"] == "add":
            skills = split_list(r["assistments_skill"])
            if not skills:
                errors.append(f"C row {r['row']}: add needs assistments_skill")
            rows += [{**base, "source_skill": s, "review": "C:add"} for s in skills]
        elif r["decision"] == "keep":
            tag = r["tag_now"].removeprefix("rule:")
            skill = rule_skill(tag) if r["tag_now"].startswith("rule:") else None
            if skill is None:
                errors.append(f"C row {r['row']}: keep needs a rule:<tag> row with a known tag, got {r['tag_now']!r}")
            else:
                rows.append({**base, "source_skill": skill, "review": "C:keep"})
        else:
            rows.append({**base, "source_skill": "", "canonical_tag": f"khan:{r['khan_slug']}", "review": "C:none"})

    # a skill whose every draft pair was dropped stays in the dictionary, with no template
    kept = {r["source_skill"] for r in rows}
    for r in rejected:
        if r["source_skill"] not in kept:
            kept.add(r["source_skill"])
            rows.append({**r, "khan_slug": "", "review": "A:all_dropped",
                         "note": "các template của bản nháp không luyện đúng kỹ năng (skill_map_v1_rejected.csv)"})
    return rows, rejected, errors


def unreviewed_rows(log_skills: dict[str, str], reviewed: set[str], mapper: SkillMapper) -> list[dict]:
    """Skills of the two logs that no sheet covers (not in the draft, < 1,000 rows in 2012): listed with a reason."""
    out = []
    for key, name in sorted(log_skills.items()):
        if key in reviewed:
            continue
        tag, why = mapper.to_tag(name), mapper.unmapped_reason(name)
        scope = "không có trong bản nháp và dưới 1.000 dòng ở 2012 nên chưa rà template"
        reason = (f"{scope}; tag {tag} qua quy tắc văn bản" if tag else
                  f"{scope}; SkillMapper chưa có quy tắc" if why == "no rule written" else
                  f"{scope}; không có tag: {why}")
        out.append({"source_skill": name, "khan_slug": "", "review": "unreviewed", "review_row": 0, "note": reason,
                    "origin": "log"})
    return out


def apply_recheck(a: pd.DataFrame, d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Sheet D (second look at dropped pairs) overrides sheet A on the rows it decides."""
    a, errors = a.copy(), []
    a["sheet"] = "A"
    for _, r in d[d["decision"] != ""].iterrows():
        if r["decision"] not in {"keep", "drop"}:
            errors.append(f"D row {r['row']}: decision {r['decision']!r} not in ['drop', 'keep']")
            continue
        i = a.index[a["row"] == r["a_row"]]
        if len(i) != 1 or a.at[i[0], "khan_slug"] != r["khan_slug"]:
            errors.append(f"D row {r['row']}: a_row {r['a_row']} does not match sheet A")
            continue
        a.at[i[0], "decision"], a.at[i[0], "sheet"] = r["decision"], "D"
        a.at[i[0], "note"] = "; ".join(x for x in (f"phiếu D: {r['decision']}", r["note"]) if x)
    return a, errors


def build(labels: Path = LABELS, out_dir: Path = MAPPINGS) -> pd.DataFrame:
    a = read_sheet(labels / "a4_review_A_draft_pairs.csv")
    b = read_sheet(labels / "a4_review_B_new_2012_skills.csv")
    c = read_sheet(labels / "a4_review_C_untagged_templates.csv")
    d_errors = []
    if (labels / "a4_review_D_recheck_drops.csv").is_file():
        a, d_errors = apply_recheck(a, read_sheet(labels / "a4_review_D_recheck_drops.csv"))
    rows, rejected, errors = collect(a, b, c)
    errors = d_errors + errors

    templates = pd.read_csv(KHAN / "khan_templates.csv").set_index("slug")
    datasets: dict[str, set] = {}  # skill key -> logs it appears in
    log_skills: dict[str, str] = {}  # skill key -> label, whitespace trimmed
    for ds in ("assist2009", "assist2012"):
        for s in pd.read_csv(TABLES / f"a1_skill_counts_{ds}.csv")["source_skill"]:
            datasets.setdefault(skill_key(s), set()).add(ds)
            log_skills.setdefault(skill_key(s), " ".join(str(s).split()))
    known = set(datasets) | {skill_key(s) for s in pd.read_csv(DRAFT)["assistments_skill"]}
    for r in rows + rejected:
        where = f"{r['review'].split(':')[0]} row {r['review_row']}"
        if r["khan_slug"] and r["khan_slug"] not in templates.index:
            errors.append(f"{where}: unknown khan slug {r['khan_slug']!r}")
        if r["source_skill"] and skill_key(r["source_skill"]) not in known:
            errors.append(f"{where}: unknown ASSISTments skill {r['source_skill']!r}")
    if errors:
        raise SystemExit("skill_map_v1 not written:\n  " + "\n  ".join(errors))

    mapper = SkillMapper()
    rows += unreviewed_rows(log_skills, {skill_key(r["source_skill"]) for r in rows}, mapper)
    junyi = pd.read_csv(KHAN / "junyi_exercise_stats_survey.csv").set_index("exercise")
    usable = set(templates.index[templates["status"] == "usable"])
    has_usable = {skill_key(r["source_skill"]) for r in rows if r["khan_slug"] in usable}

    def tag_for(skill: str) -> str:
        """SkillMapper's tag when it has one (keeps the log and GSM8K/MATH tags unchanged); a new tag only for a
        skill that gained a usable Khan template in the review; otherwise empty, i.e. unmapped."""
        return mapper.to_tag(skill) or (slugify(skill) if skill_key(skill) in has_usable else "")

    def frame(records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        if "canonical_tag" not in df.columns:
            df["canonical_tag"] = None
        df["canonical_tag"] = [t if isinstance(t, str) and t else tag_for(s) if s else ""
                               for t, s in zip(df["canonical_tag"], df["source_skill"])]
        sets = [datasets.get(skill_key(s), set()) for s in df["source_skill"]]
        df["source_dataset"] = ["both" if len(d) == 2 else next(iter(d), "") for d in sets]
        df["method"] = df["review"].map(METHOD)
        df["confidence"] = ["low" if m == "not_reviewed" else
                            "medium" if any(w in str(n).casefold() for w in DOUBT) else "high"
                            for m, n in zip(df["method"], df["note"])]
        slug = df["khan_slug"]
        df["template_status"] = slug.map(templates["status"]).fillna("no_khan_template")
        df["usable_problems"] = slug.map(templates["usable_problems"]).fillna(0).astype(int)
        df["junyi_responses"] = slug.map(junyi["responses"]).astype("Int64")
        df["junyi_accuracy"] = slug.map(junyi["accuracy"]).round(4)
        df["map_version"] = MAP_VERSION
        df = df.drop_duplicates(["source_skill", "khan_slug"])
        return df[COLUMNS].sort_values(["source_skill", "khan_slug"], ignore_index=True)

    v1, rej = frame(rows), frame(rejected)
    missing = set(datasets) - {skill_key(s) for s in v1["source_skill"]}
    assert not missing, f"skills of the logs missing from v1: {sorted(missing)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    v1.to_csv(out_dir / "skill_map_v1.csv", index=False, encoding="utf-8")
    rej.to_csv(out_dir / "skill_map_v1_rejected.csv", index=False, encoding="utf-8")
    return v1


def main() -> int:
    v1 = build()
    pairs = v1[(v1["source_skill"] != "") & (v1["khan_slug"] != "")]
    usable = pairs[pairs["template_status"] == "usable"]
    print(f"skill_map_v1.csv: {len(v1)} rows, {len(pairs)} (skill, template) pairs, {len(usable)} with a usable template")
    print(f"  skills: {pairs['source_skill'].nunique()} with a template, "
          f"{v1.loc[v1['khan_slug'] == '', 'source_skill'].nunique()} without; "
          f"templates with no skill: {(v1['source_skill'] == '').sum()}")
    print("  by method:", v1["method"].value_counts().to_dict())
    print("  skills with a canonical tag:", v1.loc[v1["canonical_tag"].str.len() > 0, "source_skill"].nunique(),
          "of", v1.loc[v1["source_skill"] != "", "source_skill"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
