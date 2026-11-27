"""Zip what the Kaggle GPU runs need, and unpack their results (LightGCN of P2-06, DKT/SAKT of P2-01).

    python -m lmsrag.kaggle_bundle pack D:\\kaggle_bundle.zip        # upload this zip as a Kaggle Dataset
    python -m lmsrag.kaggle_bundle unpack D:\\results_lightgcn.zip   # results downloaded from the notebook

``pack`` takes the Python package (no artifacts), ``configs/``, the P2-03 choice files the models read
(``reports/tables/b1_variants_choices_*.json``) and the three ``interactions_<dataset>.parquet`` of ``output_dir``
(about 80 MB). ``unpack`` only accepts files under ``reports/tables/`` and ``<output_dir>/predictions/`` whose names
start with ``b4_lightgcn`` or ``b1_deep_``, and refuses anything else.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from .datasets.build import REPO, load_config

ALLOWED_PREFIX = ("b4_lightgcn", "b1_deep_")


def pack(dest: Path, config: str = "configs/data.yaml") -> list[str]:
    cfg = load_config(REPO / config)
    out_root = Path(cfg["output_dir"])
    files = [p for p in (REPO / "lmsrag").rglob("*.py") if "artifacts" not in p.parts and "__pycache__" not in p.parts]
    files += [p for p in (REPO / "configs").rglob("*") if p.is_file()]
    files += sorted((REPO / cfg.get("tables_dir", "reports/tables")).glob("b1_variants_choices_*.json"))
    files += sorted((REPO / out_root).glob("interactions_*.parquet"))
    names = []
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            name = p.relative_to(REPO).as_posix()
            z.write(p, name)
            names.append(name)
    return names


def unpack(src: Path, config: str = "configs/data.yaml") -> list[str]:
    cfg = load_config(REPO / config)
    rel = lambda p: (REPO / p).resolve().relative_to(REPO.resolve()).as_posix()  # noqa: E731
    dirs = {rel(cfg.get("tables_dir", "reports/tables")), rel(Path(cfg["output_dir"]) / "predictions")}
    out = []
    with zipfile.ZipFile(src) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parent, _, name = info.filename.rpartition("/")
            if parent not in dirs or not name.startswith(ALLOWED_PREFIX) or ".." in info.filename:
                raise SystemExit(f"unexpected file in {src.name}: {info.filename}")
            target = REPO / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))
            out.append(info.filename)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["pack", "unpack"])
    ap.add_argument("path", type=Path)
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args(argv)
    names = pack(args.path, args.config) if args.action == "pack" else unpack(args.path, args.config)
    print("\n".join(names))
    print(f"{len(names)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
