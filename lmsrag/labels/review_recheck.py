"""Sheet D of the P1-05 review: a second look at the draft pairs dropped on sheet A that are easy to misjudge.

Two kinds of dropped pairs are listed, with one sample of *every* problem type of the template (sheet A showed only
the first type):

* the template is not usable (needs a figure, has no hints, not auto-gradable). Sheet A asked to judge the skill
  only, since such templates never enter the corpus, but they still tag Junyi responses;
* the template is usable and has more than one problem type, so the one sample may not have been representative.

The reviewer fills ``decision`` (keep / drop) and ``note``; ``build_skill_map`` then uses it instead of sheet A.

    python -m lmsrag.labels.review_recheck
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "lmsrag" / "data" / "labels"
KHAN = REPO / "lmsrag" / "data" / "khan_v2"
OUT = LABELS / "a4_review_D_recheck_drops.csv"


def type_samples() -> dict[str, list[str]]:
    """slug -> one 'question => answer' line per problem type, in type order."""
    seen: dict[str, dict[str, str]] = {}
    # corpus problems first, then every successful render (figure templates have no question text, so show a hint)
    for path in (KHAN / "khan_problems.jsonl", KHAN / "khan_rendered.jsonl"):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if "fail" in r:
                    continue
                types = seen.setdefault(r["slug"], {})
                key = str(r["problem_type"])
                if key in types:
                    continue
                text = (r.get("question") or r.get("workarea_text") or "").strip()
                if not text and r.get("hints"):
                    text = f"(đề là hình) gợi ý 1: {r['hints'][0]}"
                text = " ".join(text.split())[:300]
                types[key] = f"[dạng {len(types) + 1}] {text} => {r.get('answer', '')}"
    return {slug: list(t.values()) for slug, t in seen.items()}


def main() -> int:
    if OUT.is_file() and pd.read_csv(OUT, encoding="utf-8-sig", dtype=str)["decision"].notna().any():
        print(f"{OUT.name}: already has decisions, not overwritten")
        return 0
    a = pd.read_csv(LABELS / "a4_review_A_draft_pairs.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    a = a[(a["decision"].str.strip() == "drop") & (a["khan_slug"] != "")]
    samples = type_samples()
    junyi = pd.read_csv(KHAN / "junyi_exercise_stats_survey.csv").set_index("exercise")["responses"]
    n_types = a["khan_slug"].map(lambda s: len(samples.get(s, [])))
    a = a[(a["template_status"] != "usable") | (n_types > 1)].copy()
    a["why_listed"] = ["template không dùng được: chỉ xét kỹ năng" if s != "usable" else "template có nhiều dạng bài"
                       for s in a["template_status"]]
    d = pd.DataFrame({
        "row": range(1, len(a) + 1), "a_row": a["row"].to_numpy(), "assistments_skill": a["assistments_skill"].to_numpy(),
        "khan_slug": a["khan_slug"].to_numpy(), "template_title": a["template_title"].to_numpy(),
        "template_status": a["template_status"].to_numpy(), "why_listed": a["why_listed"].to_numpy(),
        "junyi_responses": a["khan_slug"].map(junyi).fillna(0).astype(int).to_numpy(),
        "all_types": a["khan_slug"].map(lambda s: "\n".join(samples.get(s, []))).to_numpy(),
        "decision_A": "drop", "decision": "", "note": a["note"].to_numpy(),
    })
    d.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"{OUT.name}: {len(d)} rows ({(d['template_status'] != 'usable').sum()} not usable, "
          f"{(d['template_status'] == 'usable').sum()} usable with several types)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
