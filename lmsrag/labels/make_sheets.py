"""Labelling sheets for the audit of the skill dictionary (card P1-05, step 5).

150 (log skill, content) pairs drawn with ``data_seed``:

* 100 pairs (ASSISTments skill, Khan template) from ``skill_map_v1.csv``, with probability proportional to the skill's
  number of responses in ASSISTments 2009 + 2012, split evenly over the skill's templates. Only templates that enter
  the corpus (``usable``) are drawn, since the others have no readable problem text to show;
* 50 pairs (ASSISTments skill, GSM8K/MATH chunk tagged by the text rule of ``SkillMapper``), the skill drawn the same
  way among skills that have such chunks, then one of its chunks uniformly.

Each pair shows the skill name, the template title or the source, one problem drawn at random and its answer. The
annotators fill ``label`` (1 = the problem practises that skill, 0 = it does not) and ``note``. The order is shuffled
and the sheets do not say how a pair was made; that goes to ``a4_mapping_key.csv``, which annotators must not open.

    python -m lmsrag.labels.make_sheets --config configs/data.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.build import REPO, load_config
from ..datasets.skill_map import skill_key

LABELS = REPO / "lmsrag" / "data" / "labels"
MAPPINGS = REPO / "lmsrag" / "data" / "mappings"
KHAN = REPO / "lmsrag" / "data" / "khan_v2"
TABLES = REPO / "reports" / "tables"
N_KHAN, N_RULE = 100, 50
ANNOTATORS = ("nguoi2", "nguoi3")
SHEET_COLUMNS = ["item", "skill", "skill_meaning", "source", "title", "problem", "answer", "label", "note",
                 "problem_tex"]
MAX_CHARS = 700
GLOSSARY = LABELS / "a4_skill_glossary.csv"

# TeX -> plain text, so annotators who do not read TeX can follow the problem in Excel; the original stays in
# problem_tex. Order matters: macros with arguments first, then symbols, then leftover braces and dollars.
_TEX_ARG = [
    (r"\\(?:color)\{[^{}]*\}\{([^{}]*)\}", r"\1"),
    (r"\\(?:phantom|hphantom|vphantom)\{[^{}]*\}", ""),
    # Khan colour macros, standard and custom (\leadingColor, \exponentColor, ...)
    (r"\\(?:blue|green|pink|red|purple|orange|gray|grey|BLUE|GREEN|PINK|RED|[a-z]+Color)\{([^{}]*)\}", r"\1"),
    (r"\\(?:text|textbf|mathrm|mathbf|textit|mbox|boxed|cancel|bf|it|operatorname)\{([^{}]*)\}", r"\1"),
    (r"\\cancelto\{[^{}]*\}\{([^{}]*)\}", r"\1"),
    (r"\\binom\{([^{}]*)\}\{([^{}]*)\}", r"C(\1, \2)"),
    (r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)"),  # Khan splits some over two lines
    (r"\\sqrt\[3\]\{([^{}]*)\}", r"∛(\1)"),
    (r"\\sqrt\{([^{}]*)\}", r"√(\1)"),
    # \overline: repeating digits of a decimal, or a segment in geometry
    (r"\\overline\{(\d+)\}", r"\1 (lặp lại)"),
    (r"\\overline\{([^{}]*)\}", r"đoạn \1"),
    (r"\\(?:begin|end)\{[^{}]*\}", ""),
    (r"\^\{([^{}]*)\}", r"^\1"),
]
_TEX_SYM = [
    # a colour whose argument still holds braces, e.g. \pink{{?}}: drop the command, the braces go at the end
    (r"\\(?:blue|green|pink|red|purple|orange|gray|grey|BLUE|GREEN|PINK|RED|[a-z]+Color)(?![a-zA-Z])", ""),
    (r"\\\\", " "),  # a line break first, or "\ " below would leave one of its backslashes
    (r"&=&", "="), (r"&", ""), (r"\\\[|\\\]", ""), (r"\\lvert|\\rvert", "|"), (r"\\lceil", "⌈"), (r"\\rceil", "⌉"),
    (r"\\lfloor", "⌊"), (r"\\rfloor", "⌋"), (r"\\ldots|\\dots|\\cdots", "…"), (r"\\angle\s*", "∠"), (r"\\lcm", "lcm"),
    (r"\\gcd", "gcd"), (r"\\ ", " "),
    (r"\\left|\\right", ""), (r"\\times", "×"), (r"\\div", "÷"), (r"\\cdot", "·"), (r"\\pm", "±"),
    (r"\\leq?(?![a-z])", "≤"), (r"\\geq?(?![a-z])", "≥"), (r"\\neq", "≠"), (r"\\approx", "≈"), (r"\\pi(?![a-zA-Z])", "π"),
    (r"\^\\circ|\\circ", "°"), (r"\\%", "%"), (r"\\(?:Huge|huge|LARGE|Large|large|small|displaystyle)", ""),
    (r"\\(?:qquad|quad|space|,|;|!|:)", " "),
    (r"\\(?:Bigg|bigg|Big|big)[lr]?", ""), (r"\\mid", "|"), (r"\\in(?![a-zA-Z])", "∈"), (r"\\RR|\\mathbb R", "ℝ"),
    (r"\\star", "⋆"), (r"\\Rightarrow|\\implies", "⇒"), (r"\\rightarrow|\\to(?![a-zA-Z])", "→"),
    (r"\\triangle", "△"), (r"\\theta", "θ"), (r"\\alpha", "α"), (r"\\beta", "β"),
    (r"\\(sin|cos|tan|log|ln)(?![a-zA-Z])", r"\1"), (r"\\infty", "∞"), (r"\\equiv", "≡"),
]


def plain_math(text: str) -> str:
    import re

    # an escaped dollar is money (bare dollars only delimit maths); escaped braces are set braces
    out = str(text).replace("\\$", "\x00").replace("\\{", "\x01").replace("\\}", "\x02")
    for _ in range(8):  # nested arguments, e.g. a fraction inside a colour
        before = out
        for pat, rep in _TEX_ARG:
            out = re.sub(pat, rep, out)
        if out == before:
            break
    for pat, rep in _TEX_SYM:
        out = re.sub(pat, rep, out)
    out = out.replace("$", "").replace("{", "").replace("}", "")
    out = out.replace("\x00", "$").replace("\x01", "{").replace("\x02", "}")
    return re.sub(r"[ \t]+", " ", out).strip()


def skill_rows() -> dict[str, int]:
    """skill key -> responses in ASSISTments 2009 + 2012 after cleaning, before the skill filter."""
    out: dict[str, int] = {}
    for ds in ("assist2009", "assist2012"):
        c = pd.read_csv(TABLES / f"a1_skill_counts_{ds}.csv")
        for s, n in zip(c["source_skill"], c["rows"]):
            out[skill_key(s)] = out.get(skill_key(s), 0) + int(n)
    return out


def weighted_sample(weights: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of k items drawn without replacement with probability proportional to ``weights``."""
    p = weights / weights.sum()
    return rng.choice(len(weights), size=min(k, int((p > 0).sum())), replace=False, p=p)


def khan_pairs(v1: pd.DataFrame, rows: dict[str, int], rng: np.random.Generator, n: int) -> pd.DataFrame:
    pairs = v1[(v1["source_skill"] != "") & (v1["khan_slug"] != "") & (v1["template_status"] == "usable")].copy()
    pairs["skill_rows"] = pairs["source_skill"].map(lambda s: rows.get(skill_key(s), 0))
    pairs = pairs[pairs["skill_rows"] > 0]
    per_skill = pairs.groupby("source_skill")["khan_slug"].transform("size")
    pairs["weight"] = pairs["skill_rows"] / per_skill
    pick = pairs.iloc[weighted_sample(pairs["weight"].to_numpy(float), n, rng)].copy()

    problems = pd.read_json(KHAN / "khan_problems.jsonl", lines=True, dtype={"problem_type": str})
    by_slug = {slug: g for slug, g in problems.groupby("slug")}
    shown = [by_slug[s].iloc[int(rng.integers(len(by_slug[s])))] for s in pick["khan_slug"]]
    return pd.DataFrame({
        "pair_type": "khan_template", "method": pick["method"].to_numpy(), "skill": pick["source_skill"].to_numpy(),
        "canonical_tag": pick["canonical_tag"].to_numpy(), "khan_slug": pick["khan_slug"].to_numpy(),
        "chunk_id": "", "doc_id": [p["doc_id"] for p in shown], "weight": pick["weight"].round(2).to_numpy(),
        "source": "Khan Academy", "title": [p["title"] for p in shown],
        "problem": [str(p["question"]).strip() for p in shown], "answer": [str(p["answer"]) for p in shown],
    })


def split_text(text: str) -> tuple[str, str]:
    """'Problem: … Solution: … Answer: …' -> (problem, answer)."""
    problem = text.split("\nSolution:")[0].removeprefix("Problem:").strip()
    answer = text.rsplit("\nAnswer:", 1)[1].strip() if "\nAnswer:" in text else ""
    return problem, answer


def rule_pairs(v1: pd.DataFrame, rows: dict[str, int], rng: np.random.Generator, n: int) -> pd.DataFrame:
    chunks = []
    with open(REPO / "lmsrag" / "artifacts" / "v2" / "corpus_chunks.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            # one chunk per document (the first), so the problem statement is shown whole
            if r["source"] in ("gsm8k", "math") and r["content_type"] == "worked_example" and r["chunk_idx"] == 0:
                chunks.append(r)
    ch = pd.DataFrame(chunks)
    # tag -> the log skill it stands for: the dictionary skill with that tag and the most responses
    named = v1[(v1["source_skill"] != "") & (v1["canonical_tag"] != "")].drop_duplicates("source_skill").copy()
    named["skill_rows"] = named["source_skill"].map(lambda s: rows.get(skill_key(s), 0))
    tag_rows = named.groupby("canonical_tag")["skill_rows"].sum()
    tag_skill = named.sort_values("skill_rows", ascending=False).drop_duplicates("canonical_tag")
    tag_skill = tag_skill.set_index("canonical_tag")["source_skill"]
    ch = ch[ch["skill_tag"].isin(tag_rows[tag_rows > 0].index)]

    tags = sorted(ch["skill_tag"].unique())
    weights = np.array([float(tag_rows[t]) for t in tags])
    picked, seen = [], set()
    # draw a tag by responses, then a chunk of it uniformly; a tag may be drawn again, a chunk may not
    while len(picked) < n:
        t = tags[int(rng.choice(len(tags), p=weights / weights.sum()))]
        pool = ch[(ch["skill_tag"] == t) & ~ch["chunk_id"].isin(seen)]
        if pool.empty:
            continue
        r = pool.iloc[int(rng.integers(len(pool)))]
        seen.add(r["chunk_id"])
        picked.append(r)
    pick = pd.DataFrame(picked)
    texts = [split_text(t) for t in pick["text"]]
    return pd.DataFrame({
        "pair_type": "rule_chunk", "method": "rule_text", "skill": pick["skill_tag"].map(tag_skill).to_numpy(),
        "canonical_tag": pick["skill_tag"].to_numpy(), "khan_slug": "", "chunk_id": pick["chunk_id"].to_numpy(),
        "doc_id": pick["doc_id"].to_numpy(), "weight": pick["skill_tag"].map(tag_rows).to_numpy(),
        "source": pick["source"].map({"gsm8k": "GSM8K", "math": "MATH"}).to_numpy(), "title": "",
        "problem": [p for p, _ in texts], "answer": [a for _, a in texts],
    })


def make(cfg: dict, annotators: tuple[str, ...] = ANNOTATORS, out_dir: Path = LABELS) -> pd.DataFrame:
    for a in annotators:
        path = out_dir / f"a4_mapping_{a}.csv"
        if path.is_file() and pd.read_csv(path, encoding="utf-8-sig", dtype=str)["label"].notna().any():
            raise SystemExit(f"{path.name} already has labels; not overwritten")
    rng = np.random.default_rng(int(cfg["data_seed"]))
    v1 = pd.read_csv(MAPPINGS / "skill_map_v1.csv", dtype=str, keep_default_na=False)
    rows = skill_rows()
    items = pd.concat([khan_pairs(v1, rows, rng, N_KHAN), rule_pairs(v1, rows, rng, N_RULE)], ignore_index=True)
    items = items.iloc[rng.permutation(len(items))].reset_index(drop=True)
    items.insert(0, "item", [f"m{i:03d}" for i in range(1, len(items) + 1)])
    for col in ("problem", "answer"):
        items[col] = items[col].map(lambda s: s if len(s) <= MAX_CHARS else s[:MAX_CHARS] + " …")
    items["problem_tex"] = items["problem"]
    items["problem"], items["answer"] = items["problem"].map(plain_math), items["answer"].map(plain_math)
    glossary = pd.read_csv(GLOSSARY, encoding="utf-8").set_index("skill")["meaning_vi"]
    missing = sorted(set(items["skill"]) - set(glossary.index))
    if missing:
        raise SystemExit(f"{GLOSSARY.name} has no meaning for: {missing}")
    items["skill_meaning"] = items["skill"].map(glossary)

    sheet = items.assign(label="", note="")[SHEET_COLUMNS]
    for a in annotators:
        sheet.to_csv(out_dir / f"a4_mapping_{a}.csv", index=False, encoding="utf-8-sig")
    key = items[["item", "pair_type", "method", "skill", "canonical_tag", "khan_slug", "chunk_id", "doc_id", "weight"]]
    key.to_csv(out_dir / "a4_mapping_key.csv", index=False, encoding="utf-8-sig")
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    items = make(load_config(REPO / args.config))
    print(f"{len(items)} pairs: {items['pair_type'].value_counts().to_dict()}")
    print("  by method:", items["method"].value_counts().to_dict())
    print("  distinct skills:", items["skill"].nunique(), "| templates:", items.loc[items['khan_slug'] != '', 'khan_slug'].nunique())
    print("  sheets:", ", ".join(f"a4_mapping_{a}.csv" for a in ANNOTATORS), "| key: a4_mapping_key.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
