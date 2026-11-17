"""Review sheets for the draft skill dictionary (plan card P1-05, step 1: Người 1 reviews the draft).

    python -m lmsrag.labels.review_draft

Writes three CSV files (UTF-8 with BOM so Excel shows Vietnamese and TeX correctly) in ``lmsrag/data/labels/``:

* ``a4_review_A_draft_pairs.csv``   every (ASSISTments skill, Khan template) pair of the draft, with the template's
  title and one rendered problem; pairs the plan flags as doubtful come first (priority 1), then by response count.
* ``a4_review_B_new_2012_skills.csv``  ASSISTments 2012 skills with >= 1,000 rows that the draft does not cover, each
  with three candidate templates ranked by word overlap (a suggestion only).
* ``a4_review_C_untagged_templates.csv``  usable templates that no dictionary entry, Junyi topic or text rule tags.

The reviewer fills ``decision`` (keep / drop / fix / add / none) and, when needed, ``khan_slugs`` and ``note``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
KHAN = REPO / "lmsrag" / "data" / "khan_v2"
DRAFT = REPO / "lmsrag" / "data" / "mappings" / "assistments_to_khan_draft.csv"
TABLES = REPO / "reports" / "tables"
OUT = REPO / "lmsrag" / "data" / "labels"
MIN_ROWS_2012 = 1000

# pairs the plan (card P1-05) asks to look at closely: (skill, slug prefix)
DOUBTFUL = [("Unit Rate", "rate_problems_2"), ("Volume Cylinder", "solid_geometry"), ("Volume Sphere", "solid_geometry"),
            ("Surface Area Cylinder", "solid_geometry"), ("Estimation", "rounding_"),
            ("Write Linear Equation from Graph", "slope_intercept_form"), ("Table", "reading_tables_")]
STOP = {"and", "or", "of", "the", "a", "an", "from", "with", "to", "in", "on", "by", "for", "non", "than", "more", "fewer",
        # generic words in ASSISTments skill names that say nothing about the math
        "properties", "property", "classification", "picking", "choice", "choices", "finding", "using", "side",
        "standard", "shape", "write", "writing", "problem"}


def _words(text: str) -> set[str]:
    w = re.findall(r"[a-z]+", str(text).lower())
    return {x[:-1] if len(x) > 3 and x.endswith("s") else x for x in w if x not in STOP}


def samples() -> tuple[dict, dict]:
    """slug -> (title, question, answer) from the corpus problems, falling back to any successful render."""
    title, sample = {}, {}
    for path in (KHAN / "khan_problems.jsonl", KHAN / "khan_rendered.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "fail" in r or r["slug"] in sample:
                continue
            q = (r.get("question") or r.get("workarea_text") or "").strip()
            if q:
                title[r["slug"]] = r.get("title") or ""
                sample[r["slug"]] = (q[:400], str(r.get("answer") or "")[:80])
    return title, sample


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    draft = pd.read_csv(DRAFT)
    templates = pd.read_csv(KHAN / "khan_templates.csv").set_index("slug")
    title, sample = samples()
    counts = {ds: pd.read_csv(TABLES / f"a1_skill_counts_{ds}.csv").set_index("source_skill")["rows"]
              for ds in ("assist2009", "assist2012")}

    # ---- A: draft pairs
    a = draft[["assistments_skill", "khan_slug", "template_status", "usable_problems", "junyi_accuracy"]].copy()
    a["rows_2009"] = a["assistments_skill"].map(counts["assist2009"]).fillna(0).astype(int)
    a["rows_2012"] = a["assistments_skill"].map(counts["assist2012"]).fillna(0).astype(int)
    a["template_title"] = a["khan_slug"].map(title)
    a["sample_question"] = a["khan_slug"].map(lambda s: sample.get(s, ("", ""))[0])
    a["sample_answer"] = a["khan_slug"].map(lambda s: sample.get(s, ("", ""))[1])
    a["priority"] = [1 if any(k == sk and str(sl).startswith(p) for sk, p in DOUBTFUL) else 2
                     for k, sl in zip(a["assistments_skill"], a["khan_slug"])]
    a = a.sort_values(["priority", "rows_2009", "rows_2012", "assistments_skill"], ascending=[True, False, False, True])
    a.insert(0, "row", range(1, len(a) + 1))
    a = a[["row", "priority", "assistments_skill", "rows_2009", "rows_2012", "khan_slug", "template_title",
           "template_status", "usable_problems", "junyi_accuracy", "sample_question", "sample_answer"]]
    a["decision"], a["khan_slugs"], a["note"] = "", "", ""

    # ---- B: 2012-only skills
    c12 = pd.read_csv(TABLES / "a1_skill_counts_assist2012.csv")
    known = set(draft["assistments_skill"])
    new = c12[(c12["rows"] >= MIN_ROWS_2012) & ~c12["source_skill"].isin(known)].copy()
    usable = templates.index[templates["status"] == "usable"]
    tw = {s: _words(s.replace("_", " ")) | _words(title.get(s, "")) for s in usable}
    cands = []
    for skill in new["source_skill"]:
        sw = _words(skill)
        scored = sorted(((len(sw & w) / len(sw | w), s) for s, w in tw.items() if sw & w), reverse=True)[:3]
        cands.append([f"{s} ({title.get(s, '')})" for _, s in scored] + [""] * (3 - len(scored)))
    b = pd.DataFrame({"row": range(1, len(new) + 1), "assistments_skill": new["source_skill"].to_numpy(),
                      "rows_2012": new["rows"].to_numpy(), "users_2012": new["users"].to_numpy(),
                      "accuracy_2012": new["accuracy"].to_numpy(), "current_tag": new["skill_tag"].to_numpy()})
    for i in range(3):
        b[f"candidate_{i + 1}"] = [c[i] for c in cands]
    b["decision"], b["khan_slugs"], b["note"] = "", "", ""

    # ---- C: templates with no skill
    tagged = set(draft["khan_slug"].dropna())
    topics = set(pd.read_csv(KHAN / "junyi_exercise_stats_survey.csv")["exercise"])
    untagged = [s for s in usable if s not in tagged and s not in topics]
    chunks_path = REPO / "lmsrag" / "artifacts" / "v2" / "corpus_chunks.jsonl"
    rule_tagged = set()
    if chunks_path.is_file():
        ch = pd.read_json(chunks_path, lines=True)
        rule_tagged = set(ch.loc[~ch["skill_tag"].str.startswith("khan:") & ch["khan_slug"].notna(), "khan_slug"])
    c = pd.DataFrame({"khan_slug": untagged})
    c["tag_now"] = ["rule" if s in rule_tagged else f"khan:{s}" for s in c["khan_slug"]]
    c["template_title"] = c["khan_slug"].map(title)
    c["usable_problems"] = c["khan_slug"].map(templates["usable_problems"]).astype(int)
    c["sample_question"] = c["khan_slug"].map(lambda s: sample.get(s, ("", ""))[0])
    c["needs_tag"] = c["tag_now"].str.startswith("khan:")
    c = c.sort_values(["needs_tag", "khan_slug"], ascending=[False, True]).drop(columns="needs_tag")
    c.insert(0, "row", range(1, len(c) + 1))
    c["decision"], c["assistments_skill"], c["note"] = "", "", ""

    for name, frame in (("A_draft_pairs", a), ("B_new_2012_skills", b), ("C_untagged_templates", c)):
        frame.to_csv(OUT / f"a4_review_{name}.csv", index=False, encoding="utf-8-sig")
        print(f"a4_review_{name}.csv: {len(frame)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
