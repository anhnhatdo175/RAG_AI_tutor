"""Canonical one-command experiment entrypoint."""
from __future__ import annotations

import argparse
import json

from .canonical_pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical temporal vector-profile RAG experiment.")
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
