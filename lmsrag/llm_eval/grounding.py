"""Claim-level NLI grounding and LLM-judge prompt helpers."""
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from typing import Any


def split_claims(problem: str, solution: str = "") -> list[str]:
    """Split generated problem and solution text into non-empty sentence claims."""
    claims: list[str] = []
    for text in (problem, solution):
        if not isinstance(text, str):
            raise ValueError("Problem and solution must be strings")
        claims.extend(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
            if sentence.strip()
        )
    return claims


def _chunk_text(chunk: str | dict[str, Any]) -> str:
    text = chunk if isinstance(chunk, str) else chunk.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Every retrieved chunk must contain non-empty text")
    return text.strip()


def evaluate_nli_grounding(
    claims: Sequence[str],
    chunks: Sequence[str | dict[str, Any]],
    score_entailment: Callable[[Sequence[tuple[str, str]]], Sequence[float]],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Score each claim against all retrieved chunks using an injected NLI model."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    normalized_claims = [claim.strip() for claim in claims]
    if any(not claim for claim in normalized_claims):
        raise ValueError("Claims must be non-empty strings")
    normalized_chunks = [_chunk_text(chunk) for chunk in chunks]
    if not normalized_chunks:
        return {
            "applicable": False,
            "reason": "no_retrieved_chunks",
            "threshold": float(threshold),
            "claims": [
                {"claim": claim, "max_entailment": None, "supported": None}
                for claim in normalized_claims
            ],
            "supported_claims": None,
        }
    if not normalized_claims:
        raise ValueError("At least one claim is required when retrieved chunks are present")

    pairs = [(chunk, claim) for claim in normalized_claims for chunk in normalized_chunks]
    scores = list(score_entailment(pairs))
    if len(scores) != len(pairs):
        raise ValueError(f"NLI scorer returned {len(scores)} scores for {len(pairs)} claim/chunk pairs")
    if any(isinstance(score, bool) or not isinstance(score, (int, float))
           or not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("NLI entailment scores must be finite probabilities in [0, 1]")

    results = []
    cursor = 0
    for claim in normalized_claims:
        claim_scores = [float(score) for score in scores[cursor:cursor + len(normalized_chunks)]]
        cursor += len(normalized_chunks)
        maximum = max(claim_scores)
        results.append({
            "claim": claim,
            "max_entailment": maximum,
            "supported": maximum >= threshold,
        })
    supported_count = sum(result["supported"] for result in results)
    return {
        "applicable": True,
        "threshold": float(threshold),
        "claims": results,
        "supported_claims": supported_count,
        "support_rate": supported_count / len(results),
    }


def build_llm_judge_prompt(claim: str, chunks: Sequence[str | dict[str, Any]]) -> str:
    """Create the fixed support-only rubric prompt; the caller supplies the judge model."""
    if not claim.strip():
        raise ValueError("claim must be non-empty")
    sources = [_chunk_text(chunk) for chunk in chunks]
    if not sources:
        raise ValueError("LLM grounding judgment is not applicable without retrieved chunks")
    return (
        "Decide whether the claim is supported by the supplied source text. "
        "Mark supported only when the sources entail the claim; do not use outside knowledge. "
        'Return only JSON: {"supported": true} or {"supported": false}.\n'
        f"Claim:\n{claim.strip()}\n"
        "Sources:\n"
        + "\n\n".join(f"[{index}] {source}" for index, source in enumerate(sources, start=1))
    )


def parse_judge_verdict(response: str) -> bool:
    """Parse only an explicit JSON boolean or an unambiguous supported/unsupported label."""
    if not isinstance(response, str) or not response.strip():
        raise ValueError("LLM judge response must be non-empty text")
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        label = response.strip().casefold()
        if label == "supported":
            return True
        if label == "unsupported":
            return False
        raise ValueError("LLM judge response must contain an explicit supported verdict")
    if (not isinstance(parsed, dict) or set(parsed) != {"supported"}
            or not isinstance(parsed.get("supported"), bool)):
        raise ValueError('LLM judge JSON must have a boolean "supported" field')
    return parsed["supported"]
