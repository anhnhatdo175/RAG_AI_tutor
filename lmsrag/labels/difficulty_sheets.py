"""Sheets for the difficulty audit (card P1-06, steps 1-2).

100 ``worked_example`` problems of corpus v2, drawn with ``data_seed``: 20 per ``difficulty_level`` and about 50 Khan,
25 GSM8K, 25 MATH. GSM8K has no problem at level 4 and MATH only at levels 1, 2 and 4 (ties in the per-source ranks,
see ``corpus_v2.md``), so the per-level split is fixed in ``ALLOCATION``: every level keeps 20 problems, every source
keeps its share, and each level has at least two sources to compare. A Khan template gives at most one problem.

Each sheet row shows the problem, its worked solution and the stored answer (TeX made readable, original kept in the
``*_tex`` columns) and asks for ``rating`` 1-5 (difficulty for a lower-secondary student). Khan rows also ask
``figure_ok`` (the problem can be done from the text, without a figure) and ``answer_ok`` (the stored answer is right):
together they audit the Khan filter. Neither the automatic difficulty nor the source is shown; ``a3_difficulty_key.csv``
holds them and is not sent to the annotators.

    python -m lmsrag.labels.difficulty_sheets
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..corpus.build_corpus import khan_documents, text_documents
from ..datasets.build import REPO, load_config
from ..datasets.junyi2015 import DEFAULT_MAPPING
from ..skill_mapping import SkillMapper
from .make_sheets import plain_math

LABELS = REPO / "lmsrag" / "data" / "labels"
ANNOTATORS = ("nguoi2", "nguoi3")
# difficulty_level -> problems per source; rows sum to 20, columns to 50 / 25 / 25
ALLOCATION = {
    1: {"khan": 8, "gsm8k": 6, "math": 6},
    2: {"khan": 8, "gsm8k": 6, "math": 6},
    3: {"khan": 13, "gsm8k": 7},
    4: {"khan": 7, "math": 13},
    5: {"khan": 14, "gsm8k": 6},
}
NOT_ASKED = "–"  # the two Khan checks on GSM8K / MATH rows
SHEET_COLUMNS = ["item", "problem", "solution", "answer", "rating", "figure_ok", "answer_ok", "note",
                 "problem_tex", "solution_tex", "answer_tex"]
KEY_COLUMNS = ["item", "doc_id", "chunk_id", "source", "khan_slug", "problem_type", "skill_tag", "difficulty",
               "difficulty_level", "difficulty_source", "difficulty_n"]


def display_answer(answer: str, answer_type: str) -> str:
    """The stored answer as a reader should see it. Khan's ``radical`` / ``cuberoot`` answers keep only the number
    under one root sign (7√3 is stored as 147, meaning √147), so the sign is added back; round 1 of P1-06 showed the
    bare number and one rater read it as a wrong answer (d066)."""
    if answer_type == "radical":
        return f"√({answer})"
    if answer_type == "cuberoot":
        return f"∛({answer})"
    return answer


def corpus_documents(cfg: dict) -> pd.DataFrame:
    """One row per worked_example document of the built corpus: its metadata from the chunks, and the full problem,
    solution and answer from the same document builders (a long document is split over several chunks)."""
    chunks = pd.read_json(REPO / cfg["output_dir"] / "corpus_chunks.jsonl", lines=True)
    first = chunks[(chunks["content_type"] == "worked_example") & (chunks["chunk_idx"] == 0)]
    mapper = SkillMapper()
    mapping = REPO / cfg["datasets"]["junyi2015"].get("skill_map", DEFAULT_MAPPING)
    text = pd.concat([khan_documents(mapping, mapper), text_documents(REPO / "lmsrag" / "data" / "raw", mapper)[0]])
    text = text.drop_duplicates("doc_id").set_index("doc_id")[["question", "solution", "answer"]]
    docs = first.rename(columns={"answer": "answer_meta"}).join(text, on="doc_id")
    missing = docs["question"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} corpus documents not found by the builders; rebuild the corpus first")
    return docs.reset_index(drop=True)


def sample(docs: pd.DataFrame, rng: np.random.Generator, allocation: dict = ALLOCATION) -> pd.DataFrame:
    picked, seen_templates = [], set()
    for level, per_source in sorted(allocation.items()):
        for source, n in per_source.items():
            pool = docs[(docs["difficulty_level"] == level) & (docs["source"] == source)].sort_values("doc_id")
            if len(pool) < n:
                raise SystemExit(f"level {level} {source}: {len(pool)} documents, {n} needed")
            take = []
            for i in rng.permutation(len(pool)):
                row = pool.iloc[i]
                if source == "khan" and row["khan_slug"] in seen_templates:
                    continue
                take.append(row)
                if source == "khan":
                    seen_templates.add(row["khan_slug"])
                if len(take) == n:
                    break
            if len(take) < n:
                raise SystemExit(f"level {level} {source}: only {len(take)} distinct templates")
            picked += take
    out = pd.DataFrame(picked).reset_index(drop=True)
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


def make(cfg: dict, annotators: tuple[str, ...] = ANNOTATORS, out_dir: Path = LABELS) -> pd.DataFrame:
    for a in annotators:
        path = out_dir / f"a3_difficulty_{a}.csv"
        if path.is_file() and pd.read_csv(path, encoding="utf-8-sig", dtype=str)["rating"].notna().any():
            raise SystemExit(f"{path.name} already has ratings; not overwritten")
    rng = np.random.default_rng(int(cfg["data_seed"]))
    items = sample(corpus_documents(cfg), rng)
    items.insert(0, "item", [f"d{i:03d}" for i in range(1, len(items) + 1)])
    for col in ("question", "solution", "answer"):
        items[col] = items[col].astype(str)
    khan = items["source"] == "khan"
    sheet = pd.DataFrame({
        "item": items["item"],
        "problem": items["question"].map(plain_math), "solution": items["solution"].map(plain_math),
        "answer": [display_answer(plain_math(a), t) for a, t in zip(items["answer"], items["answer_type"])],
        "rating": "", "figure_ok": np.where(khan, "", NOT_ASKED), "answer_ok": np.where(khan, "", NOT_ASKED),
        "note": "",
        "problem_tex": items["question"], "solution_tex": items["solution"], "answer_tex": items["answer"],
    })[SHEET_COLUMNS]
    for a in annotators:
        sheet.to_csv(out_dir / f"a3_difficulty_{a}.csv", index=False, encoding="utf-8-sig")
    items[KEY_COLUMNS].to_csv(out_dir / "a3_difficulty_key.csv", index=False, encoding="utf-8-sig")
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    items = make(load_config(REPO / args.config))
    print(f"{len(items)} problems; by source {items['source'].value_counts().to_dict()}")
    print(pd.crosstab(items["difficulty_level"], items["source"], margins=True).to_string())
    print("  Khan templates:", items.loc[items["source"] == "khan", "khan_slug"].nunique(),
          "| difficulty_source:", json.dumps(items["difficulty_source"].value_counts().to_dict()))
    print("  sheets:", ", ".join(f"a3_difficulty_{a}.csv" for a in ANNOTATORS), "| key: a3_difficulty_key.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
