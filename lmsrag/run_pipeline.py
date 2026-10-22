"""Build missing Layer 1-2 artifacts, then run the interactive Layer 1-6 loop.

Usage: python -m lmsrag.run_pipeline --user-id 123 --skill Addition-and-Subtraction-Fractions
"""
from __future__ import annotations

import argparse
import sys

from lmsrag import config


def _artifacts_ready() -> bool:
    required = (
        config.ART / "index_manifest.json",
        config.CHUNKS_JSONL,
        config.EMB_NPY,
        config.SKILL_CATALOG_JSON,
        config.ART / "exercise_bank.jsonl",
        config.ART / "learner_state.sqlite",
    )
    return all(path.is_file() for path in required)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--skill", required=True, help="Canonical skill tag")
    parser.add_argument("--query", help="Learning topic; defaults to --skill")
    parser.add_argument("--subject", default="Mathematics")
    parser.add_argument("--grade", default="lower_secondary")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild Layer 1 and Layer 2 artifacts")
    args = parser.parse_args(argv)

    if args.rebuild or not _artifacts_ready():
        from lmsrag.download_data import download_missing
        from lmsrag.layer1_data.build import build as build_layer1
        from lmsrag.layer2_embedding.build import build as build_layer2

        try:
            print("[lmsrag] Preparing source data")
            download_missing()
            print("[lmsrag] Building Layer 1")
            build_layer1()
            print("[lmsrag] Building Layer 2 (LSA)")
            build_layer2(embedder="lsa")
        except Exception as exc:
            print(f"[lmsrag] Build failed: {exc}", file=sys.stderr)
            return 2
    else:
        print("[lmsrag] Reusing existing Layer 1-2 artifacts")

    from lmsrag.app import main as run_learning_loop

    app_args = ["--user-id", str(args.user_id), "--skill", args.skill,
                "--subject", args.subject, "--grade", args.grade]
    if args.query:
        app_args.extend(["--query", args.query])
    return run_learning_loop(app_args)


if __name__ == "__main__":
    raise SystemExit(main())
