"""Copying, duplication, skill coverage, usage, and Socratic leakage metrics."""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from typing import Any

from lmsrag.llm_eval.answer_check import answers_equivalent


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("Text metrics require strings")
    return re.findall(r"\w+|[^\w\s]", text.casefold(), flags=re.UNICODE)


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Length of the longest common subsequence using O(len(right)) memory."""
    if len(left) < len(right):
        left, right = right, left
    row = [0] * (len(right) + 1)
    for left_token in left:
        previous = 0
        for index, right_token in enumerate(right, start=1):
            old = row[index]
            row[index] = previous + 1 if left_token == right_token else max(row[index], row[index - 1])
            previous = old
    return row[-1]


def longest_common_run(left: Sequence[str], right: Sequence[str]) -> int:
    """Length of the longest contiguous shared token sequence."""
    if len(left) < len(right):
        left, right = right, left
    row = [0] * (len(right) + 1)
    longest = 0
    for left_token in left:
        previous = 0
        for index, right_token in enumerate(right, start=1):
            old = row[index]
            row[index] = previous + 1 if left_token == right_token else 0
            previous = old
            longest = max(longest, row[index])
    return longest


def rouge_l_f1(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    common = lcs_length(left, right)
    precision = common / len(left)
    recall = common / len(right)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def copying_metrics(
    generated_problem: str,
    chunks: Sequence[str | dict[str, Any]],
    *,
    min_common_tokens: int = 12,
    rouge_l_threshold: float = 0.8,
) -> dict[str, Any]:
    if min_common_tokens < 1 or not 0 <= rouge_l_threshold <= 1:
        raise ValueError("Copy thresholds must be positive and within [0, 1]")
    generated_tokens = tokenize(generated_problem)
    best: dict[str, Any] = {"chunk_id": None, "common_tokens": 0, "rouge_l": 0.0}
    for chunk in chunks:
        text = chunk if isinstance(chunk, str) else chunk.get("text")
        if not isinstance(text, str):
            raise ValueError("Every chunk must be a string or contain string text")
        source_tokens = tokenize(text)
        common = longest_common_run(generated_tokens, source_tokens)
        rouge = rouge_l_f1(generated_tokens, source_tokens)
        if (common, rouge) > (best["common_tokens"], best["rouge_l"]):
            best = {
                "chunk_id": None if isinstance(chunk, str) else chunk.get("doc_id", chunk.get("chunk_id")),
                "common_tokens": common,
                "rouge_l": rouge,
            }
    return {
        **best,
        "copied": best["common_tokens"] >= min_common_tokens or best["rouge_l"] >= rouge_l_threshold,
    }


def duplicate_pairs(
    items: Sequence[dict[str, Any]],
    embeddings: Sequence[Sequence[float]],
    *,
    threshold: float = 0.9,
) -> list[dict[str, Any]]:
    """Find within-skill BGE cosine duplicates; equality to the threshold is not a match."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if len(items) != len(embeddings):
        raise ValueError("Each item must have exactly one embedding")
    vectors: list[list[float]] = []
    for vector in embeddings:
        values = [float(value) for value in vector]
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("Embeddings must be non-empty finite vectors")
        vectors.append(values)
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) > 1:
        raise ValueError("All embeddings must have the same dimension")

    norms = [math.sqrt(sum(value * value for value in vector)) for vector in vectors]
    if any(norm == 0 for norm in norms):
        raise ValueError("Zero-length embeddings cannot be compared")
    pairs = []
    for left_index, left in enumerate(items):
        for right_index in range(left_index + 1, len(items)):
            right = items[right_index]
            if left.get("skill_tag") != right.get("skill_tag"):
                continue
            cosine = sum(a * b for a, b in zip(vectors[left_index], vectors[right_index])) / (
                norms[left_index] * norms[right_index]
            )
            if cosine > threshold:
                pairs.append({
                    "left_id": left.get("request_id"),
                    "right_id": right.get("request_id"),
                    "skill_tag": left.get("skill_tag"),
                    "cosine": cosine,
                })
    return pairs


def generation_validation_metrics(result_or_error: Any) -> dict[str, Any]:
    """Summarize schema validation and retries reported by LLMGenerator."""
    if isinstance(result_or_error, dict):
        attempts = result_or_error.get("generation_attempts")
        normalized_safety = result_or_error.get("normalized_safety")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ValueError("Validated generation must include a positive generation_attempts count")
        if not isinstance(normalized_safety, bool):
            raise ValueError("Validated generation must include boolean normalized_safety")
        return {
            "json_valid": True,
            "retry_count": attempts - 1,
            "normalized_safety": normalized_safety,
            "error": None,
        }
    attempts = getattr(result_or_error, "attempts", None)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("Generation error must expose a positive attempts count")
    return {
        "json_valid": False,
        "retry_count": attempts - 1,
        "normalized_safety": None,
        "error": str(result_or_error),
    }


def evaluate_skill_coverage(
    problem: str,
    target_skill: str,
    mapper: Any,
    judge: Callable[[str, str], bool],
    *,
    subject: str = "Mathematics",
    level: int | None = None,
    is_gsm8k: bool = False,
) -> dict[str, Any]:
    rule_tag, topic_label, confidence = mapper.tag_text(problem, subject, level, is_gsm8k)
    rule_match = rule_tag == target_skill
    judge_match = judge(problem, target_skill)
    if not isinstance(judge_match, bool):
        raise ValueError("Skill judge must return a boolean")
    return {
        "rule_tag": rule_tag,
        "topic_label": topic_label,
        "rule_confidence": confidence,
        "rule_match": rule_match,
        "judge_match": judge_match,
        "agreement": rule_match == judge_match,
        "covered": rule_match if rule_match == judge_match else None,
    }


def ollama_usage_metrics(response: dict[str, Any], retrieval_latency_ms: float) -> dict[str, int | float]:
    if not isinstance(response, dict):
        raise ValueError("Ollama response must be an object")
    if (isinstance(retrieval_latency_ms, bool) or not isinstance(retrieval_latency_ms, (int, float))
            or not math.isfinite(retrieval_latency_ms) or retrieval_latency_ms < 0):
        raise ValueError("retrieval_latency_ms must be finite and non-negative")
    values = {}
    for key in ("total_duration", "prompt_eval_count", "eval_count"):
        value = response.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Ollama response {key} must be a non-negative integer")
        values[key] = value
    return {**values, "retrieval_latency_ms": float(retrieval_latency_ms)}


_ANSWER_TOKEN = re.compile(
    r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?(?!\w|\.\d)"
)
_ROOT_TOKEN = re.compile(
    r"(?<!\w)[-+]?\d*\s*(?:\\sqrt|sqrt|√|∛|cbrt)\s*(?:\([^()]*\)|\{[^{}]*\}|\d+)"
)


def _contains_equivalent_answer(text: str, answer_key: str | int | float, answer_type: str | None) -> bool:
    answer_text = re.sub(r"\s+", "", str(answer_key)).casefold()
    if answer_text and re.search(r"[A-Za-z=<>+*/^()]", answer_text) and answer_type not in {
        "radical", "cuberoot",
    }:
        spaced_pattern = r"(?<!\w)" + r"\s*".join(re.escape(char) for char in answer_text) + r"(?!\w)"
        if re.search(spaced_pattern, text, flags=re.IGNORECASE):
            return True
    candidates = [match.group(0) for match in _ANSWER_TOKEN.finditer(text)]
    candidates.extend(match.group(0) for match in _ROOT_TOKEN.finditer(text))
    candidates.extend((text.strip(),))
    return any(answers_equivalent(candidate, answer_key, answer_type) for candidate in candidates)


def socratic_answer_leakage(
    events: Sequence[dict[str, Any]],
    answer_key: str | int | float,
    *,
    answer_type: str | None = None,
) -> dict[str, Any]:
    """Flag an assistant answer shown before any student turn has supplied it."""
    student_has_answer = False
    for index, event in enumerate(events):
        role = event.get("role")
        content = event.get("content")
        if role not in {"assistant", "student"} or not isinstance(content, str):
            raise ValueError("Each event requires role assistant/student and string content")
        contains_answer = _contains_equivalent_answer(content, answer_key, answer_type)
        if role == "assistant" and contains_answer and not student_has_answer:
            return {"leaked": True, "event_index": index, "student_had_answer": False}
        if role == "student" and contains_answer:
            student_has_answer = True
    return {"leaked": False, "event_index": None, "student_had_answer": student_has_answer}
