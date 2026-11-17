"""Common interaction table shared by all logs (plan tab "Quy ước chung", section 3).

Every model reads ``lmsrag/artifacts/v2/interactions_<dataset>.parquet``; nothing downstream reads the raw CSVs.
"""
from __future__ import annotations

import pandas as pd

# column -> pandas dtype after ``coerce``. "string" columns may hold nulls only where NULLABLE says so.
COLUMNS: dict[str, str] = {
    "dataset": "string",
    "user_id": "string",
    "item_id": "string",
    "source_skill": "string",
    "skill_tag": "string",
    "khan_slug": "string",
    "ts": "int64",
    "correct": "int8",
    "rt_ms": "float32",
    "attempts": "int16",
    "hints": "int16",
    "section_type": "string",
    "split": "string",
}
NULLABLE = {"skill_tag", "khan_slug", "rt_ms"}
DATASETS = {"assist2009", "assist2012", "junyi2015"}
SPLITS = ("history", "validation", "test")


class SchemaError(ValueError):
    """Raised when a frame does not satisfy the common interaction schema."""


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Select the schema columns in canonical order and cast them to the schema dtypes."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    out = df[list(COLUMNS)].copy()
    for col, dtype in COLUMNS.items():
        out[col] = out[col].astype(dtype)
    return out


def validate(df: pd.DataFrame, *, require_split: bool = True) -> None:
    """Check column names, dtypes, null rules and value domains; raise ``SchemaError`` on the first violation."""
    if list(df.columns) != list(COLUMNS):
        raise SchemaError(f"columns must be exactly {list(COLUMNS)} in this order, got {list(df.columns)}")
    for col, dtype in COLUMNS.items():
        if str(df[col].dtype) != dtype:
            raise SchemaError(f"column {col!r} has dtype {df[col].dtype}, expected {dtype}")
        if col not in NULLABLE and not (col == "split" and not require_split) and df[col].isna().any():
            raise SchemaError(f"column {col!r} contains nulls")
    bad = set(df["dataset"].unique()) - DATASETS
    if bad:
        raise SchemaError(f"unknown dataset values: {sorted(bad)}")
    if not df["correct"].isin([0, 1]).all():
        raise SchemaError("correct must be 0 or 1")
    if require_split and not df["split"].isin(SPLITS).all():
        raise SchemaError(f"split must be one of {SPLITS}")
    rt = df["rt_ms"].dropna()
    if len(rt) and (rt <= 0).any():
        raise SchemaError("rt_ms must be positive or null")
    if (df["attempts"] < 0).any() or (df["hints"] < 0).any():
        raise SchemaError("attempts and hints must be non-negative")


def clean_rt(values: pd.Series, clip: tuple[float, float]) -> pd.Series:
    """Response time in ms: non-positive values become missing, the rest are clipped to ``clip``."""
    rt = pd.to_numeric(values, errors="coerce").astype("float64")
    rt = rt.where(rt > 0)
    return rt.clip(*clip).astype("float32")


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in COLUMNS.items()})


