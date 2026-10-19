"""
LAYER 2 / stage 3 - RECURSIVE CHUNKING (docx 2.3.1: <=512 tokens, overlap 50 words).

The splitter tries the coarsest separator first (paragraph -> line -> sentence -> word) and only falls through to a finer one when a
piece is still too long, so chunks end on natural boundaries.  Greedy merge fills each chunk up to the budget; the last
`overlap_words` words of the previous chunk are prepended to the next one; continuation chunks also get a short
"[Problem: ...]" header so they stay self-explanatory.  Token count uses tiktoken when its vocabulary is available,
otherwise a regex proxy (words + punctuation + numbers) that tracks BPE counts on math text closely enough for a budget.
"""
from __future__ import annotations

import re
from functools import lru_cache

MAX_TOKENS = 512
OVERLAP_WORDS = 50
_TOK = re.compile(r"\\[A-Za-z]+|\w+|[^\w\s]")
_SEPS = [r"\n\s*\n", r"\n", r"(?<=[.!?])\s+", r"\s+"]


@lru_cache(maxsize=1)
def _tiktoken():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def n_tokens(text: str) -> int:
    enc = _tiktoken()
    return len(enc.encode(text)) if enc else len(_TOK.findall(text))


def _split_recursive(text: str, budget: int, level: int = 0) -> list[str]:
    if n_tokens(text) <= budget:
        return [text]
    if level >= len(_SEPS):                                 # a single over-long "word" (e.g. huge LaTeX blob): hard cut
        return [text[i:i + budget * 3] for i in range(0, len(text), budget * 3)]
    parts = [p for p in re.split(_SEPS[level], text) if p and p.strip()]
    if len(parts) == 1:
        return _split_recursive(text, budget, level + 1)
    out, cur = [], ""
    joiner = "\n\n" if level == 0 else "\n" if level == 1 else " "
    for p in parts:
        cand = f"{cur}{joiner}{p}" if cur else p
        if n_tokens(cand) <= budget:
            cur = cand
        else:
            if cur:
                out.append(cur)
            if n_tokens(p) > budget:
                out.extend(_split_recursive(p, budget, level + 1))
                cur = ""
            else:
                cur = p
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, header: str = "", max_tokens: int = MAX_TOKENS, overlap_words: int = OVERLAP_WORDS) -> list[str]:
    """Return chunks each with n_tokens <= max_tokens (overlap and header included)."""
    if n_tokens(text) <= max_tokens:
        return [text]
    head_budget = n_tokens(header) + 4 if header else 0
    pieces = _split_recursive(text, max(64, max_tokens - 2 * overlap_words - head_budget))
    out = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        words = prev.split()[-overlap_words:]
        while True:                                          # shrink the overlap until the chunk fits the budget
            body = f"{' '.join(words)} {cur}".strip()
            cand = f"{header}\n{body}" if header else body
            if n_tokens(cand) <= max_tokens or not words:
                break
            words = words[max(1, len(words) // 10):] if len(words) > 1 else []
        if n_tokens(cand) > max_tokens:                      # extreme case: header + piece alone too big
            cand = body
        out.append(cand)
    return out


def make_chunks(record, **kw) -> list[dict]:
    """ParsedRecord -> list of chunk dicts (metadata is attached later by the tagging stage)."""
    body = f"Problem: {record.question}\nSolution: {record.solution}".strip() if record.solution else record.question
    header = f"[Problem: {' '.join(record.question.split()[:25])} ...]"
    pieces = chunk_text(body, header=header, **kw)
    return [dict(chunk_id=f"{record.doc_id}#{j}", doc_id=record.doc_id, chunk_idx=j, n_chunks=len(pieces), text=p,
                 token_count=n_tokens(p), content_type=record.content_type if j == 0 else record.content_type + "_cont")
            for j, p in enumerate(pieces)]
