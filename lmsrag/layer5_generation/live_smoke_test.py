"""Run a real local Ollama inference through the Layer 5 generator."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    os.environ["LLM_BACKEND"] = "ollama"
    os.environ["GENERATION_MODEL"] = args.model
    os.environ["OLLAMA_BASE_URL"] = args.base_url
    os.environ.setdefault("OLLAMA_CONTEXT_LENGTH", "2048")
    os.environ.setdefault("OLLAMA_MAX_NEW_TOKENS", "512")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from lmsrag.layer1_data.learner_state import LearnerState
    from lmsrag.layer5_generation import GenerationContext, LLMGenerator

    skill = "Addition-and-Subtraction-Fractions"
    state = LearnerState(user_id=1)
    state.canonical_M[skill] = 0.48
    context = GenerationContext.from_sources(
        retrieved_docs=[{
            "doc_id": "demo-fractions-01",
            "skill_tag": skill,
            "subject": "Mathematics",
            "grade": "lower_secondary",
            "text": "Để cộng hai phân số khác mẫu số, hãy tìm mẫu số chung rồi quy đồng tử số trước khi cộng.",
        }],
        candidate_tasks=[{
            "item_id": "demo-task-01",
            "doc_id": "demo-fractions-01",
            "skill_tag": skill,
            "mapping_status": "mapped",
            "subject": "Mathematics",
            "grade": "lower_secondary",
            "difficulty": 0.6,
            "statement": "Tính 1/3 + 1/4.",
            "response_type": "numeric",
        }],
        learner_state=state,
        constraints={
            "mode": "socratic",
            "target_skill": skill,
            "target_subject": "Mathematics",
            "target_grade": "lower_secondary",
            "language": "vi",
            "max_steps": 3,
            "allow_new_task_generation": False,
        },
        top_k=1,
        top_n=1,
    )

    started = time.perf_counter()
    result = LLMGenerator().generate(context)
    evidence = {
        "backend": "ollama-local",
        "model": args.model,
        "endpoint": args.base_url,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "latency_seconds": round(time.perf_counter() - started, 2),
        "project_schema_validation": "passed",
        "context_source": "demo data; no learner personal data",
        "result": result,
    }
    rendered = json.dumps(evidence, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())