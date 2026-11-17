"""Build the common interaction tables and the cleaning reports.

    python -m lmsrag.datasets.build --config configs/data.yaml [--only assist2009,assist2012]

Writes ``<output_dir>/interactions_<dataset>.parquet`` and one CSV per report table in ``<tables_dir>``.
"""
from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
ADAPTERS = {"assist2009": "lmsrag.datasets.assist2009"}


def load_config(path: Path) -> dict:
    """Read the data config; uses ``lmsrag.settings`` once it exists (plan card P3-01)."""
    try:
        from lmsrag import settings  # type: ignore[attr-defined]
        return settings.load(path)
    except (ImportError, AttributeError):
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "data.yaml")
    parser.add_argument("--only", help="comma-separated dataset names (default: every registered adapter)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    names = args.only.split(",") if args.only else list(ADAPTERS)
    unknown = [n for n in names if n not in ADAPTERS]
    if unknown:
        parser.error(f"no adapter for {unknown}; known: {sorted(ADAPTERS)}")
    out_dir, tables_dir = REPO / cfg["output_dir"], REPO / cfg.get("tables_dir", "reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        started = time.time()
        result = importlib.import_module(ADAPTERS[name]).build(cfg, REPO)
        target = out_dir / f"interactions_{name}.parquet"
        result.interactions.to_parquet(target, index=False)
        for table, frame in result.tables.items():
            frame.to_csv(tables_dir / f"{table}.csv", index=False)
        df = result.interactions
        shares = df["split"].value_counts(normalize=True).round(4).to_dict()
        print(f"{name}: {len(df):,} rows, {df.user_id.nunique():,} learners, accuracy {df.correct.mean():.4f}, "
              f"splits {shares} -> {target.relative_to(REPO)} ({time.time() - started:.1f}s)")
        for k, v in result.facts.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
