"""Conservative query cleanup and configurable spelling/abbreviation expansion."""
from __future__ import annotations

import re


DEFAULT_SPELLING_CORRECTIONS = {
    "fracion": "fraction",
    "substract": "subtract",
    "substraction": "subtraction",
    "probablity": "probability",
    "perimiter": "perimeter",
}

DEFAULT_ABBREVIATIONS = {
    "gcf": "greatest common factor",
    "gcd": "greatest common divisor",
    "lcm": "least common multiple",
    "pemdas": "order of operations",
}

_WORD = re.compile(r"(?<![A-Za-z0-9_\\])([A-Za-z]{2,})(?![A-Za-z0-9_])")


def _preserve_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement if " " in replacement else replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


class QueryRewriter:
    """Normalize prose while leaving numbers, LaTeX commands, and operators untouched."""

    def __init__(self, spelling_corrections: dict[str, str] | None = None,
                 abbreviations: dict[str, str] | None = None):
        self.spelling_corrections = {
            key.casefold(): value for key, value in
            (DEFAULT_SPELLING_CORRECTIONS if spelling_corrections is None else spelling_corrections).items()
        }
        self.abbreviations = {
            key.casefold(): value for key, value in
            (DEFAULT_ABBREVIATIONS if abbreviations is None else abbreviations).items()
        }

    def rewrite(self, query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        text = " ".join(query.strip().split())

        def replace(match: re.Match[str]) -> str:
            word = match.group(1)
            key = word.casefold()
            replacement = self.spelling_corrections.get(key)
            if replacement is None:
                replacement = self.abbreviations.get(key)
            return _preserve_case(word, replacement) if replacement else word

        return _WORD.sub(replace, text)
