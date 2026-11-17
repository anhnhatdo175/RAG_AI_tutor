"""Turn rendered Khan instances (see ``render_khan``) into the corpus file: keep items that are self-contained in
text, have a worked solution and an answer that can be checked automatically; record why every other item was dropped.

Usage:
    python -m lmsrag.data_tools.build_khan_corpus --rendered lmsrag/data/khan_v2/khan_rendered.jsonl \
        --repo <khan-exercises checkout> --out-dir lmsrag/data/khan_v2

Outputs in --out-dir:
    khan_problems.jsonl    one usable problem per line (doc_id, slug, problem_type, question, hints, solution, answer ...)
    khan_templates.csv     one row per template: status, number of usable problems and problem types

The filter is rule-based. Its precision must be reported in the data card from a manual sample (plan item A3).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# wording that means the statement cannot be answered from its text alone
VISUAL = re.compile(
    r"\b(shown|number line|graph(?:ed|s)?|table|figure|diagram|pictograph|picture|plot(?:ted)?|chart|protractor|grid|"
    r"the shape|below are|drag|move the|orange dot|empty cell|"
    r"(?:blue|orange|green|pink|red|purple) (?:dot|angle|line|point|row|column|segment))\b", re.I)
IMPERATIVE_GRAPH = re.compile(r"^\s*(graph|plot|draw|construct)\b", re.I)
CODE_ANSWER = re.compile(r"\b(guess|return|function|getContent|setContent)\b|graph\.|\bvar ")
HAS_NUMBER = re.compile(r"\d")
QUESTION_BLOCK = re.compile(r'class="(?:question|problem)"[\s\S]*?</div>')
# statements that read as text but rely on an unlabeled drawing (found by manual spot check, 2026-11-17)
MANUAL_FIGURE = {"angles_1", "angles_2", "scaling_vectors", "parallel_lines_2", "divisibility_intuition", "triangle_angles_1",
                 "quadrilateral_angles", "congruent_triangles_1", "congruent_triangles_2"}


def source_has_figure(repo: Path, slug: str) -> bool:
    """True when the template's own question block embeds a drawing canvas."""
    path = repo / "exercises" / f"{slug}.html"
    if not path.is_file():
        return False
    html = path.read_text(encoding="utf-8", errors="ignore")
    return "graphie" in " ".join(QUESTION_BLOCK.findall(html))


def classify(record: dict, src_figure: bool) -> str:
    if "fail" in record:
        return "render_fail"
    question = (record.get("workarea_text") or record.get("question") or "").strip()
    answer = (record.get("answer") or "").strip()
    if not record.get("hints"):
        return "no_hints"
    if record.get("answer_type") in ("custom", None) or CODE_ANSWER.search(answer) or not answer or len(answer) > 80:
        return "not_autogradable"
    canvas = bool(src_figure or record.get("has_figure"))
    if record["slug"] in MANUAL_FIGURE or VISUAL.search(question) or IMPERATIVE_GRAPH.search(question):
        return "needs_figure"
    if len(question) < 12:
        return "needs_figure" if canvas else "too_short"
    if canvas and not HAS_NUMBER.search(question):
        return "needs_figure"
    return "usable"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rendered", required=True, type=Path, nargs="+", help="one or more files written by render_khan")
    parser.add_argument("--repo", required=True, type=Path, help="khan-exercises checkout used for rendering")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    records = [json.loads(line) for f in args.rendered for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    figure_cache: dict[str, bool] = {}
    per_template: dict[str, Counter] = defaultdict(Counter)
    types: dict[str, set] = defaultdict(set)
    seen: set[tuple[str, str]] = set()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    with (args.out_dir / "khan_problems.jsonl").open("w", encoding="utf-8") as sink:
        for r in records:
            slug = r["slug"]
            if slug not in figure_cache:
                figure_cache[slug] = source_has_figure(args.repo, slug)
            status = classify(r, figure_cache[slug])
            per_template[slug][status] += 1
            if status != "usable":
                continue
            question = (r.get("workarea_text") or r["question"]).strip()
            if (slug, question) in seen:  # different seeds can produce the same instance
                per_template[slug]["duplicate"] += 1
                per_template[slug]["usable"] -= 1
                continue
            seen.add((slug, question))
            ptype = str(r.get("problem_type") or r["problem_index"])
            types[slug].add(ptype)
            sink.write(json.dumps({
                "doc_id": f"khan:{slug}:{ptype}:{r['seed']}",
                "slug": slug, "problem_type": ptype, "seed": r["seed"], "title": r.get("title") or "",
                "question": question, "hints": r["hints"], "solution": "\n".join(r["hints"]),
                "answer": r["answer"].strip(), "answer_type": r.get("answer_type"), "choices": r.get("choices") or [],
                "has_canvas": bool(figure_cache[slug] or r.get("has_figure")),
                "source": "Khan/khan-exercises@001b27a5^", "licence": "CC BY-NC-SA 3.0",
            }, ensure_ascii=False) + "\n")
            kept += 1

    with (args.out_dir / "khan_templates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slug", "status", "usable_problems", "problem_types", "rendered", "drop_reasons"])
        for slug in sorted(per_template):
            c = per_template[slug]
            usable = c.get("usable", 0)
            reasons = {k: v for k, v in c.items() if k != "usable" and v}
            status = "usable" if usable > 0 else (max(reasons, key=reasons.get) if reasons else "empty")
            writer.writerow([slug, status, usable, len(types[slug]), sum(v for k, v in c.items() if k != "duplicate"),
                             json.dumps(reasons, sort_keys=True)])

    statuses = Counter("usable" if c.get("usable", 0) > 0 else "dropped" for c in per_template.values())
    print(f"templates: {len(per_template)} ({dict(statuses)}) | usable problems written: {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
