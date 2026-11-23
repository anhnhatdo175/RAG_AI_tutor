"""Lightweight fixed rubric helpers shared by labels and LLM evaluation."""
from __future__ import annotations

import math


def target_level(mastery: float) -> int:
    """Map mastery to the fixed 1–5 target level using half-up rounding."""
    if isinstance(mastery, bool) or not isinstance(mastery, (int, float)) or not math.isfinite(mastery):
        raise ValueError("mastery must be a finite number")
    return int(min(5, max(1, math.floor(1 + 5 * mastery + 0.5))))
