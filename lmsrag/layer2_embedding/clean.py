"""
LAYER 2 / stage 2 - CLEAN.  Normalise text, drop unusable records, remove (near-)duplicates. Every stage is counted.

 1. unicode NFC + invisible characters (U+2028, zero-width, NBSP) + whitespace
 2. GSM8K calculator annotations  <<48/2=24>>  removed
 3. Asymptote diagrams [asy]...[/asy] cannot be rendered in text-only RAG -> record flagged `has_figure`
    and (default) dropped when the statement depends on the figure
 4. quality filters: empty / too short / too long
 5. exact duplicates (hash of normalised question) and near duplicates (MinHash-LSH on 5-word shingles, Jaccard >= 0.85)
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import replace

from .parse import ParsedRecord

CALC_RE = re.compile(r"<<[^>]*>>")
ASY_RE = re.compile(r"\[asy\].*?\[/asy\]", re.S)
INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u2028\u2029\u0085\x0b\x0c]")
FIG_REF_RE = re.compile(r"\b(shown|figure|diagram|picture|graph below|as follows|in the (?:figure|diagram))\b", re.I)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = INVISIBLE_RE.sub(" ", text)
    text = CALC_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _shingles(text: str, k: int = 5) -> set[str]:
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def clean_records(records, min_chars=12, max_words=3000, drop_figure_dependent=True, near_dup_threshold=0.85):
    report = Counter()
    kept, seen_exact = [], set()
    try:
        from datasketch import MinHash, MinHashLSH
        lsh = MinHashLSH(threshold=near_dup_threshold, num_perm=64)
    except ImportError:                                   # exact dedup only
        lsh = None
    for r in records:
        report["input"] += 1
        has_fig = bool(ASY_RE.search(r.question) or ASY_RE.search(r.solution))
        q = normalize(ASY_RE.sub(" [figure] ", r.question))
        s = normalize(ASY_RE.sub(" [figure] ", r.solution))
        a = normalize(r.answer)
        if has_fig:
            report["flagged_has_figure"] += 1
            if drop_figure_dependent and (ASY_RE.search(r.question) or FIG_REF_RE.search(q)):
                report["dropped_figure_dependent"] += 1
                continue
        nw = len(q.split())
        if len(q) < min_chars:
            report["dropped_too_short"] += 1
            continue
        if nw + len(s.split()) > max_words:
            report["dropped_too_long"] += 1
            continue
        h = hashlib.sha1(re.sub(r"\W+", "", q.lower()).encode()).hexdigest()
        if h in seen_exact:
            report["dropped_exact_duplicate"] += 1
            continue
        seen_exact.add(h)
        if lsh is not None:
            m = MinHash(num_perm=64)
            for sh in _shingles(q):
                m.update(sh.encode())
            if lsh.query(m):
                report["dropped_near_duplicate"] += 1
                continue
            lsh.insert(r.doc_id, m)
        kept.append(replace(r, question=q, solution=s, answer=a, extra={**r.extra, "has_figure": has_fig}))
        report["kept"] += 1
    return kept, dict(report)
