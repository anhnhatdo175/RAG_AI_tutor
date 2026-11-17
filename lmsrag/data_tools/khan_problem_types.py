"""List the problem types of every khan-exercises template: (slug, problem_index, problem_id).

A template's ``<div class="problems">`` holds one child ``<div>`` per problem type. The renderer (``render_khan``)
selects a type by its position (``?problem=<index>``) and records that index, while the Junyi log records the
child's ``id`` attribute when it has one and the index otherwise. This table joins the two.

Usage:
    python -m lmsrag.data_tools.khan_problem_types --repo <khan-exercises checkout at 001b27a5^> \
        --out lmsrag/data/khan_v2/khan_problem_types.csv
"""
from __future__ import annotations

import argparse
import csv
from html.parser import HTMLParser
from pathlib import Path


class _ProblemTypes(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.problems_depth: int | None = None
        self.types: list[str | None] = []

    def handle_starttag(self, tag, attrs):
        if tag != "div":
            return
        self.depth += 1
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if self.problems_depth is None and "problems" in classes:
            self.problems_depth = self.depth
        elif self.problems_depth is not None and self.depth == self.problems_depth + 1:
            self.types.append(a.get("id"))

    def handle_endtag(self, tag):
        if tag != "div":
            return
        if self.problems_depth is not None and self.depth == self.problems_depth:
            self.problems_depth = -1  # closed; ignore any later block
        self.depth -= 1


def problem_types(html: str) -> list[str]:
    """Problem-type keys in document order: the child's id, or its index when it has none."""
    p = _ProblemTypes()
    p.feed(html)
    return [t if t else str(i) for i, t in enumerate(p.types)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = 0
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slug", "problem_index", "problem_id"])
        for path in sorted((args.repo / "exercises").glob("*.html")):
            if path.stem.startswith("khan-"):
                continue
            for i, key in enumerate(problem_types(path.read_text(encoding="utf-8", errors="ignore"))):
                w.writerow([path.stem, i, key])
                rows += 1
    print(f"wrote {rows} problem types to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
