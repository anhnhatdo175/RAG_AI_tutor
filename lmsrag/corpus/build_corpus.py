"""Build corpus v2: Khan worked examples and hint steps, GSM8K, MATH (plan card P1-04, teacher's item A3).

    python -m lmsrag.corpus.build_corpus --config configs/data.yaml

Inputs: ``lmsrag/data/khan_v2/`` (problems, templates, problem types), ``lmsrag/data/raw/gsm8k_*.jsonl`` and
``math_test.jsonl``, ``lmsrag/artifacts/v2/interactions_junyi2015.parquet`` (history split only), the skill dictionary.
Outputs: ``<output_dir>/corpus_chunks.jsonl``, ``<output_dir>/exercise_bank_v2.jsonl``,
``reports/tables/a3_corpus_summary.csv``, ``reports/figures/a3_difficulty_by_source.png``.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..datasets.build import REPO, load_config
from ..datasets.junyi2015 import DEFAULT_MAPPING, khan_tag_lists
from ..datasets.skill_map import skill_key
from ..layer2_embedding import chunk as chunker
from ..layer2_embedding.clean import ASY_RE, clean_records
from ..layer2_embedding.parse import parse_gsm8k, parse_math
from ..skill_mapping import SkillMapper
from . import difficulty as D

MATH_SUBJECTS = {"Prealgebra", "Algebra", "Number Theory", "Counting & Probability"}
MATH_LEVELS = (1, 3)
MAX_TOKENS = 512
TOKENIZER = "BAAI/bge-small-en-v1.5"
LICENCE = {"khan": "CC BY-NC-SA 3.0", "gsm8k": "MIT", "math": "MIT"}
KHAN_DIR = REPO / "lmsrag" / "data" / "khan_v2"


@lru_cache(maxsize=1)
def bge_counter():
    """Token counter of the embedding model (special tokens included), so the 512 budget is the model's own."""
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    except OSError:
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tok.model_max_length = 10**9  # we only count; silences the "longer than 512" warning
    return lambda text: len(tok(text, add_special_tokens=True, truncation=False)["input_ids"])


# ---------------------------------------------------------------- documents (one row per problem)
def khan_documents(mapping_csv: Path, mapper: SkillMapper) -> pd.DataFrame:
    probs = pd.read_json(KHAN_DIR / "khan_problems.jsonl", lines=True, dtype={"problem_type": str})
    types = pd.read_csv(KHAN_DIR / "khan_problem_types.csv", dtype=str)
    key = dict(zip(zip(types["slug"], types["problem_index"]), types["problem_id"]))
    probs["type_key"] = [key.get((s, p), p) for s, p in zip(probs["slug"], probs["problem_type"].astype(str))]
    topics = pd.read_csv(KHAN_DIR / "junyi_exercise_stats_survey.csv", usecols=["exercise", "topic"])
    topic = dict(zip(topics["exercise"], topics["topic"].astype(str)))
    tag_lists = khan_tag_lists(mapping_csv, mapper)

    def tags_for(slug: str, text: str) -> tuple[list[str], str]:
        if slug in tag_lists:  # v1 also lists templates reviewed as having no ASSISTments skill, as khan:<slug>
            tags = tag_lists[slug]
            return tags, "template_only" if all(t.startswith("khan:") for t in tags) else "dictionary"
        if slug in topic:  # same rule as the Junyi adapter, so chunks and Junyi responses share the tag
            return [f"junyi:{topic[slug]}"], "junyi_topic"
        tag, _, _ = mapper.tag_text(text.lower(), f"khan:{slug}", None, False)
        # last resort: the template itself is the skill; listed for the P1-05 review so it gets a real tag
        return ([tag], "rule") if tag else ([f"khan:{slug}"], "template_only")

    tagged = [tags_for(s, q + "\n" + sol) for s, q, sol in zip(probs["slug"], probs["question"], probs["solution"])]
    return pd.DataFrame({
        "doc_id": probs["doc_id"], "source": "khan", "khan_slug": probs["slug"], "problem_type": probs["type_key"],
        "type_key": probs["type_key"], "title": probs["title"], "question": probs["question"],
        "solution": probs["solution"], "answer": probs["answer"].astype(str), "answer_type": probs["answer_type"],
        "hints": probs["hints"], "skill_tags": [t for t, _ in tagged], "tag_method": [m for _, m in tagged],
        "subject": None, "level": None, "raw_steps": probs["hints"].map(len),
    })


def text_documents(raw_dir: Path, mapper: SkillMapper) -> tuple[pd.DataFrame, dict]:
    """GSM8K (train + test) and MATH (4 subjects, Level 1-3, no [asy]) through the existing parse/clean stages."""
    records, report = [], {}
    for f in sorted(raw_dir.glob("gsm8k_*.jsonl")):
        records.extend(parse_gsm8k(f))
    math_all = [r for f in sorted(raw_dir.glob("math_*.jsonl")) for r in parse_math(f)]
    report["math_input"] = len(math_all)
    math = [r for r in math_all if r.subject in MATH_SUBJECTS and MATH_LEVELS[0] <= (r.level or 0) <= MATH_LEVELS[1]]
    report["math_after_subject_level"] = len(math)
    math = [r for r in math if not (ASY_RE.search(r.question) or ASY_RE.search(r.solution))]
    report["math_after_asy"] = len(math)
    report["gsm8k_input"] = len(records)
    kept, clean_report = clean_records(records + math)
    report.update({f"clean_{k}": v for k, v in clean_report.items()})

    rows = []
    for r in kept:
        if r.is_gsm8k:
            split, idx = r.doc_id.split("-")[1:]
            doc_id, source, raw = f"gsm8k:{split}:{idx}", "gsm8k", float(r.extra["n_steps"])
        else:
            # unique_id "test/prealgebra/126.json" is stored by the parser as "test_prealgebra_126"
            doc_id, source, raw = f"math:{r.doc_id.removeprefix('math-')}", "math", float(r.level)
        tag, _, _ = mapper.tag_text(r.question.lower(), r.subject, r.level, r.is_gsm8k)
        rows.append({"doc_id": doc_id, "source": source, "khan_slug": None, "problem_type": None, "type_key": None,
                     "title": r.subject, "question": r.question, "solution": r.solution, "answer": r.answer,
                     "answer_type": "number" if r.is_gsm8k else "math_expression", "hints": [],
                     "skill_tags": [tag] if tag else [], "tag_method": "rule" if tag else "none",
                     "subject": r.subject, "level": r.level, "raw_steps": raw})
    return pd.DataFrame(rows), report


# ---------------------------------------------------------------- difficulty
def add_difficulty(docs: pd.DataFrame, junyi: pd.DataFrame | None, cfg: dict) -> tuple[pd.DataFrame, dict]:
    m = float(cfg["difficulty"]["shrinkage_m"])
    min_n = int(cfg["difficulty"]["min_responses_unit"])
    docs = docs.copy()
    docs["p_correct_hist"], docs["difficulty_n"], docs["difficulty_unit"] = np.nan, 0, None
    khan = docs["source"] == "khan"
    info: dict = {}
    if junyi is not None:
        unit, template = D.junyi_history_counts(junyi)
        hist = D.khan_history_difficulty(docs[khan].rename(columns={"khan_slug": "slug"}), unit, template, m, min_n)
        docs.loc[khan, ["p_correct_hist", "difficulty_n", "difficulty_unit"]] = hist.to_numpy()
        info["junyi_units_ge_min_n"] = int((unit["n"] >= min_n).sum())
    docs["p_correct_hist"] = docs["p_correct_hist"].astype(float)
    docs["difficulty_n"] = docs["difficulty_n"].astype(int)
    has_hist = khan & docs["p_correct_hist"].notna()
    raw = pd.Series(np.nan, index=docs.index)
    raw[has_hist] = 1.0 - docs.loc[has_hist, "p_correct_hist"]
    no_hist = khan & ~has_hist
    if has_hist.any() and no_hist.any():
        model = D.fit_steps_model(docs.loc[has_hist, "raw_steps"], raw[has_hist])
        raw[no_hist] = model.predict(docs.loc[no_hist, "raw_steps"].to_numpy(float))
        info["isotonic_steps_to_difficulty"] = {int(s): round(float(model.predict([s])[0]), 4)
                                                for s in sorted(docs.loc[khan, "raw_steps"].unique())}
    other = ~khan
    raw[other] = docs.loc[other, "raw_steps"]
    docs["difficulty_raw"] = raw
    docs["difficulty_source"] = np.select([has_hist, no_hist, docs["source"] == "gsm8k"],
                                          ["junyi_history", "khan_steps", "gsm8k_steps"], "math_level")
    docs["difficulty"] = D.percentile_within(docs["difficulty_raw"], docs["source"]).round(6)
    docs["difficulty_level"] = D.quintile_levels(docs["difficulty"])
    return docs, info


# ---------------------------------------------------------------- chunks
def _py(v):
    """numpy scalars -> Python, NaN -> None, so every record is plain JSON."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def worked_text(d) -> str:
    return f"Problem: {d.question}\nSolution: {d.solution}\nAnswer: {d.answer}".strip()


def make_chunks(docs: pd.DataFrame, count) -> tuple[list[dict], dict]:
    chunks, truncated, hints_split = [], 0, 0
    common = ["doc_id", "source", "khan_slug", "problem_type", "difficulty", "difficulty_level", "difficulty_source",
              "difficulty_n", "answer", "answer_type"]
    for d in docs.itertuples(index=False):
        base = {c: _py(getattr(d, c)) for c in common}
        base["p_correct_hist"] = None if pd.isna(d.p_correct_hist) else round(float(d.p_correct_hist), 6)
        base.update(skill_tag=d.skill_tags[0], skill_tags=list(d.skill_tags), licence=LICENCE[d.source])
        text = worked_text(d)
        header = f"[Problem: {' '.join(str(d.question).split()[:25])} ...]"
        pieces = [text] if count(text) <= MAX_TOKENS else chunker.chunk_text(text, header=header,
                                                                              max_tokens=MAX_TOKENS, count=count)
        truncated += len(pieces) > 1
        for j, p in enumerate(pieces):
            chunks.append({"chunk_id": f"{d.doc_id}#{j}", **base, "content_type": "worked_example", "chunk_idx": j,
                           "n_chunks": len(pieces), "text": p, "token_count": count(p)})
        for k, hint in enumerate(d.hints, 1):
            t = f"Problem: {d.question}\nHint {k} of {len(d.hints)}: {hint}"
            parts = [t] if count(t) <= MAX_TOKENS else chunker.chunk_text(t, header=header, max_tokens=MAX_TOKENS,
                                                                           count=count)
            hints_split += len(parts) > 1
            for j, p in enumerate(parts):
                suffix = f"h{k}" if len(parts) == 1 else f"h{k}.{j}"
                chunks.append({"chunk_id": f"{d.doc_id}#{suffix}", **base, "content_type": "hint_step", "chunk_idx": k,
                               "n_chunks": len(d.hints), "text": p, "token_count": count(p)})
    return chunks, {"worked_docs_split_over_512_tokens": truncated, "hint_steps_split_over_512_tokens": hints_split}


def exercise_bank(docs: pd.DataFrame) -> list[dict]:
    """Auto-gradable problems (Khan and GSM8K); ``text`` is the statement only."""
    out = []
    for d in docs[docs["source"].isin(["khan", "gsm8k"])].itertuples(index=False):
        out.append({"chunk_id": f"{d.doc_id}#q", "doc_id": d.doc_id, "source": d.source, "skill_tag": d.skill_tags[0],
                    "skill_tags": list(d.skill_tags), "khan_slug": _py(d.khan_slug), "problem_type": _py(d.problem_type),
                    "content_type": "exercise", "difficulty": float(d.difficulty), "difficulty_level": int(d.difficulty_level),
                    "difficulty_source": d.difficulty_source, "difficulty_n": int(d.difficulty_n),
                    "p_correct_hist": None if pd.isna(d.p_correct_hist) else round(float(d.p_correct_hist), 6),
                    "text": d.question, "answer": d.answer, "answer_type": d.answer_type, "licence": LICENCE[d.source]})
    return out


# ---------------------------------------------------------------- report
def coverage_rows(docs: pd.DataFrame, tables_dir: Path, mapping_csv: Path) -> list[dict]:
    """Share of ASSISTments responses (before the skill filter) whose skill has (a) a usable Khan template in the
    dictionary, (b) at least one chunk of any source through its canonical tag."""
    rows = []
    mapping = pd.read_csv(mapping_csv)
    skill_col = "source_skill" if "source_skill" in mapping.columns else "assistments_skill"
    templates = pd.read_csv(KHAN_DIR / "khan_templates.csv")
    usable = set(templates.loc[templates["status"] == "usable", "slug"])
    # compare on the normalised key: the 2009 log has labels with trailing spaces ("Circumference ")
    with_khan = {skill_key(s) for s in mapping.loc[mapping["khan_slug"].isin(usable), skill_col].dropna()}
    chunk_tags = {t for tags in docs["skill_tags"] for t in tags}
    for ds in ("assist2009", "assist2012"):
        path = tables_dir / f"a1_skill_counts_{ds}.csv"
        if not path.is_file():
            continue
        c = pd.read_csv(path)
        total = c["rows"].sum()
        rows.append({"section": "coverage", "source": ds, "key": "share_rows_skill_has_usable_khan_template",
                     "value": round(float(c.loc[c["source_skill"].map(skill_key).isin(with_khan), "rows"].sum()
                                          / total), 4)})
        rows.append({"section": "coverage", "source": ds, "key": "share_rows_skill_tag_has_any_chunk",
                     "value": round(float(c.loc[c["skill_tag"].isin(chunk_tags), "rows"].sum() / total), 4)})
    return rows


def summary(docs: pd.DataFrame, chunks: list[dict], info: dict, tables_dir: Path, mapping_csv: Path) -> pd.DataFrame:
    rows = [{"section": "build", "source": "all", "key": k, "value": json.dumps(v) if isinstance(v, dict) else v}
            for k, v in info.items()]
    ch = pd.DataFrame(chunks)
    for (src, ctype), n in ch.groupby(["source", "content_type"]).size().items():
        rows.append({"section": "chunks", "source": src, "key": ctype, "value": int(n)})
    for (src, lvl), n in docs.groupby(["source", "difficulty_level"]).size().items():
        rows.append({"section": "docs_by_level", "source": src, "key": f"level_{lvl}", "value": int(n)})
    for src, g in docs.groupby("source"):
        for method, n in g["tag_method"].value_counts().items():
            rows.append({"section": "tag_method", "source": src, "key": method, "value": int(n)})
        rows.append({"section": "difficulty_source", "source": src, "key": "counts",
                     "value": json.dumps(g["difficulty_source"].value_counts().to_dict())})
    long = docs.explode("skill_tags")
    levels = long.groupby("skill_tags")["difficulty_level"].nunique()
    canonical = levels[~levels.index.str.contains(r"^(?:junyi|khan):", regex=True)]
    rows += [{"section": "skills", "source": "all", "key": "skills_with_chunks", "value": int(len(levels))},
             {"section": "skills", "source": "all", "key": "skills_with_chunks_at_ge3_levels",
              "value": int((levels >= 3).sum())},
             {"section": "skills", "source": "all", "key": "canonical_skills_with_chunks", "value": int(len(canonical))},
             {"section": "skills", "source": "all", "key": "canonical_skills_with_chunks_at_ge3_levels",
              "value": int((canonical >= 3).sum())}]
    rows += coverage_rows(docs, tables_dir, mapping_csv)
    return pd.DataFrame(rows)


def plot_difficulty(docs: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["khan", "gsm8k", "math"]
    shares = (docs.groupby("source")["difficulty_level"].value_counts(normalize=True).unstack(fill_value=0)
              .reindex(order).reindex(columns=range(1, 6), fill_value=0))
    counts = docs["source"].value_counts().reindex(order)
    blues = ["#cfe0f3", "#9ec2e6", "#6aa0d6", "#3b78bf", "#1f4e8c"]
    fig, ax = plt.subplots(figsize=(7.5, 2.8), dpi=150)
    left = np.zeros(len(order))
    for lvl, color in zip(range(1, 6), blues):
        ax.barh(range(len(order)), shares[lvl], left=left, color=color, edgecolor="white", linewidth=1,
                label=f"Level {lvl}")
        left += shares[lvl].to_numpy()
    ax.set_yticks(range(len(order)), [f"{s} (n={counts[s]:,})" for s in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of problems")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(ncols=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8)
    ax.set_title("Difficulty level (corpus-wide quintiles) by source", fontsize=10, pad=24, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "data.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    out_dir = REPO / cfg["output_dir"]
    tables_dir = REPO / cfg.get("tables_dir", "reports/tables")
    mapping_csv = REPO / cfg["datasets"]["junyi2015"].get("skill_map", DEFAULT_MAPPING)
    mapper = SkillMapper()

    khan = khan_documents(mapping_csv, mapper)
    text_docs, info = text_documents(REPO / "lmsrag" / "data" / "raw", mapper)
    docs = pd.concat([khan, text_docs], ignore_index=True)
    info["untagged_docs_dropped"] = json.dumps(docs.loc[docs["skill_tags"].map(len) == 0, "source"].value_counts().to_dict())
    docs = docs[docs["skill_tags"].map(len) > 0].reset_index(drop=True)
    junyi_path = out_dir / "interactions_junyi2015.parquet"
    junyi = pd.read_parquet(junyi_path, columns=["item_id", "khan_slug", "correct", "split"]) if junyi_path.is_file() else None
    docs, diff_info = add_difficulty(docs, junyi, cfg)
    info.update(diff_info)

    chunks, chunk_info = make_chunks(docs, bge_counter())
    info.update(chunk_info)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "corpus_chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    bank = exercise_bank(docs)
    with (out_dir / "exercise_bank_v2.jsonl").open("w", encoding="utf-8") as f:
        for b in bank:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    table = summary(docs, chunks, info, tables_dir, mapping_csv)
    table.to_csv(tables_dir / "a3_corpus_summary.csv", index=False)
    plot_difficulty(docs, REPO / "reports" / "figures" / "a3_difficulty_by_source.png")
    print(f"docs {len(docs):,} | chunks {len(chunks):,} | exercise bank {len(bank):,}")
    print(table.to_string(index=False, max_colwidth=90))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
