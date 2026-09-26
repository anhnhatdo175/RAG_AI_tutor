"""Single canonical temporal vector-profile RAG experiment."""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
ROOT = Path(__file__).resolve().parent
MATERIALS = ROOT / "materials"
MANIFEST = ROOT / "corpus_manifest.json"
DATASET = ROOT / "skill_builder_data.csv"
FINAL = ROOT / "artifacts" / "final"
SELECTED_USERS = {"78523", "78561", "78571", "78544", "78557"}


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    skill_id: str
    skill_name: str
    difficulty: str
    content_type: str
    source_file: str


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="latin-1", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            try:
                row["order_id"] = int(row["order_id"])
                row["correct"] = int(row["correct"])
                row["attempt_count"] = int(row["attempt_count"])
                row["ms_first_response"] = float(row["ms_first_response"])
            except (KeyError, TypeError, ValueError):
                continue
            if (row.get("user_id") in SELECTED_USERS and row.get("skill_id")
                    and row.get("problem_id")):
                rows.append(row)
    return sorted(rows, key=lambda r: (str(r["user_id"]), r["order_id"]))


def load_chunks() -> list[Chunk]:
    chunks = []
    for doc in json.loads(MANIFEST.read_text(encoding="utf-8-sig")):
        raw = (ROOT / doc["file"]).read_text(encoding="utf-8-sig")
        body = raw.split("---", 2)[-1].strip()
        chunks.append(Chunk(
            chunk_id=doc["doc_id"], doc_id=doc["doc_id"], text=body,
            skill_id=str(doc["skill_id"]), skill_name=doc["skill_name"],
            difficulty=doc["difficulty"], content_type=doc["content_type"],
            source_file=doc["file"],
        ))
    return chunks


def difficulty_from_population(rows: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[str(row["problem_id"])].append(row["correct"])
    result = {}
    for item, values in grouped.items():
        p = sum(values) / len(values)
        result[item] = "beginner" if p >= 0.78 else "intermediate" if p >= 0.55 else "advanced"
    return result


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_user[str(row["user_id"])].append(row)
    history, validation, test = [], [], []
    for user_rows in by_user.values():
        n = len(user_rows)
        a, b = max(1, int(n * 0.70)), max(2, int(n * 0.85))
        history.extend(user_rows[:a])
        validation.extend(user_rows[a:b])
        test.extend(user_rows[b:])
    return history, validation, test


def encode_profile(rows: list[dict[str, Any]], model: SentenceTransformer) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["user_id"]), str(row["skill_id"]))].append(row)
    profiles: dict[str, Any] = {}
    for (user, skill), attempts in grouped.items():
        newest = max(r["order_id"] for r in attempts)
        grouped_outcomes = {0: 0.0, 1: 0.0}
        for row in attempts:
            age = max(0, newest - row["order_id"])
            recency = 0.5 ** (age / max(1, newest - min(r["order_id"] for r in attempts) + 1))
            speed = 1.0 / (1.0 + max(0.0, row["ms_first_response"]) / 60000.0)
            grouped_outcomes[row["correct"]] += (1.0 + 0.5 * row["correct"]) * recency * (0.75 + 0.25 * speed)
        texts = [
            f"Skill {skill}, learner answered incorrectly and needs targeted support",
            f"Skill {skill}, learner answered correctly and demonstrates mastery",
        ]
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                               batch_size=2, show_progress_bar=False).astype("float32")
        vector = np.average(vectors, axis=0, weights=np.asarray([grouped_outcomes[0], grouped_outcomes[1]])).astype("float32")
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        profiles.setdefault(user, {"student_id": user, "skills": {}})["skills"][skill] = {
            "attempts": len(attempts),
            "accuracy": round(sum(r["correct"] for r in attempts) / len(attempts), 4),
            "profile_vector": vector.tolist(),
        }
    return profiles


def build_index(chunks: list[Chunk], model: SentenceTransformer):
    vectors = model.encode([c.text for c in chunks], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, vectors


def _profile_vector(profile: dict[str, Any], student: str, skill: str | None) -> np.ndarray:
    data = profile.get(student, {}).get("skills", {}).get(str(skill), {}) if skill else {}
    return np.asarray(data.get("profile_vector", []), dtype="float32")


def _candidate_hits(scores, ids, chunks, vectors, profile_vector, predicted_skill,
                    alpha, beta, top_k, known_skill=None, profile_fallback=False):
    hits = []
    for semantic, idx in zip(scores, ids):
        chunk = chunks[int(idx)]
        if known_skill is not None and chunk.skill_id != str(known_skill):
            continue
        profile_score = float(np.dot(profile_vector, vectors[int(idx)])) if profile_vector.size else 0.0
        skill_score = 1.0 if predicted_skill and chunk.skill_id == str(predicted_skill) else 0.0
        hits.append({"doc_id": chunk.doc_id, "skill_id": chunk.skill_id,
                     "difficulty": chunk.difficulty, "content_type": chunk.content_type,
                     "semantic_score": float(semantic), "skill_score": skill_score,
                     "profile_score": profile_score, "difficulty_score": profile_score,
                     "profile_fallback": profile_fallback,
                     "score": alpha * float(semantic) + beta * profile_score})
    return sorted(hits, key=lambda h: h["score"], reverse=True)[:top_k]


def _semantic_candidates(query, index, model):
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, ids = index.search(q, index.ntotal)
    return scores[0], ids[0]


def retrieve_known_skill(query, student_id, known_skill_id, profiles, index, chunks, model,
                         vectors, alpha, beta, top_k=5):
    scores, ids = _semantic_candidates(query, index, model)
    return _candidate_hits(scores, ids, chunks, vectors,
                           _profile_vector(profiles, student_id, known_skill_id),
                           known_skill_id, alpha, beta, top_k,
                           known_skill=known_skill_id,
                           profile_fallback=not bool(_profile_vector(profiles, student_id, known_skill_id).size))


def evaluate(rows, profile, chunks, index, vectors, model, item_difficulty, alpha, beta,
             split_name):
    out = []
    for row in rows:
        skill = str(row["skill_id"])
        query = f"Solve a learning problem for {row.get('skill_name', skill)} (item {row['problem_id']})."
        hits = retrieve_known_skill(query, str(row["user_id"]), skill, profile, index, chunks, model, vectors, alpha, beta)
        gold_diff = item_difficulty.get(str(row["problem_id"]), "intermediate")
        ranks = [i + 1 for i, h in enumerate(hits) if h["skill_id"] == skill]
        diff_ranks = [i + 1 for i, h in enumerate(hits) if h["skill_id"] == skill and h["difficulty"] == gold_diff]
        out.append({"split": split_name, "track": "known_skill", "student_id": str(row["user_id"]), "skill_id": skill,
                    "problem_id": str(row["problem_id"]), "gold_difficulty": gold_diff,
                    "skill_hit": int(bool(ranks)), "skill_rank": ranks[0] if ranks else None,
                    "difficulty_hit": int(bool(diff_ranks)), "difficulty_rank": diff_ranks[0] if diff_ranks else None,
                    "difficulty_mrr_value": 1.0 / diff_ranks[0] if diff_ranks else 0.0,
                    "predicted_skill": hits[0]["skill_id"] if hits else None,
                    "profile_fallback": bool(hits[0]["profile_fallback"]) if hits else True,
                    "top_doc_ids": [h["doc_id"] for h in hits],
                    "top_scores": hits})
    return out


def metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = max(1, len(rows))
    return {"n": len(rows), "skill_recall_at_5": sum(r["skill_hit"] for r in rows) / n,
            "skill_mrr": sum(1 / r["skill_rank"] if r["skill_rank"] else 0 for r in rows) / n,
            "difficulty_hit_at_5": sum(r["difficulty_hit"] for r in rows) / n,
            "difficulty_mrr": sum(1 / r["difficulty_rank"] if r["difficulty_rank"] else 0 for r in rows) / n}


def bootstrap_ci(rows: list[dict[str, Any]], field: str, seed: int = 7) -> list[float]:
    if not rows:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    values = np.asarray([float(r[field]) for r in rows], dtype="float32")
    samples = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(500)]
    return [round(float(np.percentile(samples, 2.5)), 4), round(float(np.percentile(samples, 97.5)), 4)]


def write_report_files(report: dict[str, Any], baseline: list[dict[str, Any]], adaptive: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    comparison = []
    for name, rows in (("Traditional RAG", baseline), ("Adaptive Profile-Based RAG", adaptive)):
        values = metrics(rows)
        values["system"] = name
        values["difficulty_mrr_ci95"] = bootstrap_ci(rows, "difficulty_mrr_value")
        comparison.append(values)
    with (FINAL / f"{prefix}_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["system", "skill_recall_at_5", "skill_mrr", "difficulty_hit_at_5", "difficulty_mrr", "difficulty_mrr_ci95"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in comparison)

    students = sorted({r["student_id"] for r in baseline})
    per_student = []
    for student in students:
        b = [r for r in baseline if r["student_id"] == student]
        a = [r for r in adaptive if r["student_id"] == student]
        per_student.append({"student_id": student, "traditional_difficulty_mrr": metrics(b)["difficulty_mrr"],
                            "adaptive_difficulty_mrr": metrics(a)["difficulty_mrr"], "test_rows": len(b)})
    with (FINAL / f"{prefix}_per_student.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["student_id", "traditional_difficulty_mrr", "adaptive_difficulty_mrr", "test_rows"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_student)

    labels = ["Skill Recall@5", "Skill MRR", "Difficulty Hit@5", "Difficulty MRR"]
    keys = ["skill_recall_at_5", "skill_mrr", "difficulty_hit_at_5", "difficulty_mrr"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, [comparison[0][k] for k in keys], width, label="Traditional RAG")
    ax.bar(x + width / 2, [comparison[1][k] for k in keys], width, label="Adaptive Profile-Based RAG")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_title("Traditional RAG vs Adaptive Profile-Based RAG")
    ax.legend(); ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(FINAL / f"{prefix}_comparison.png", dpi=180); plt.close(fig)

    x = np.arange(len(per_student))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, [r["traditional_difficulty_mrr"] for r in per_student], width, label="Traditional RAG")
    ax.bar(x + width / 2, [r["adaptive_difficulty_mrr"] for r in per_student], width, label="Adaptive Profile-Based RAG")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Difficulty MRR"); ax.set_xticks(x, [r["student_id"] for r in per_student])
    ax.set_title("Difficulty MRR by learner"); ax.legend(); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(FINAL / f"{prefix}_per_student.png", dpi=180); plt.close(fig)

    report["comparison"] = comparison
    report["per_student"] = per_student
    write_json(FINAL / f"{prefix}_report.json", report)
    md = ["# Temporal Vector-Profile RAG Report", "", "## Experimental setup", "",
          f"- Dataset interactions: {sum(report['split'].values())}",
          f"- Temporal split: history={report['split']['history']}, validation={report['split']['validation']}, test={report['split']['test']}",
          f"- Learner profiles: {report['profiles']}; corpus chunks: {report['chunks']}",
          f"- Selected configuration: alpha={report['selected_config']['alpha']}, beta={report['selected_config']['beta']}", "",
          "## Main results", "", "| System | Skill Recall@5 | Skill MRR | Difficulty Hit@5 | Difficulty MRR |", "|---|---:|---:|---:|---:|"]
    for row in comparison:
        md.append(f"| {row['system']} | {row['skill_recall_at_5']:.4f} | {row['skill_mrr']:.4f} | {row['difficulty_hit_at_5']:.4f} | {row['difficulty_mrr']:.4f} |")
    md += ["", "## Figures", "", f"- `{prefix}_comparison.png`: overall metric comparison.", f"- `{prefix}_per_student.png`: difficulty MRR by learner.", "", "## Limitations", "",
           "- ASSISTments export does not contain complete problem statements; queries use skill name and problem ID.",
           "- The instructional corpus is synthetic and requires subject-matter review.",
           "- Results measure retrieval ranking, not final generated-answer quality."]
    (FINAL / f"{prefix}_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"comparison": comparison, "per_student": per_student}


def run() -> dict[str, Any]:
    started = time.perf_counter()
    rows = read_rows(DATASET)
    print(f"[data] loaded {len(rows)} selected-user interactions", flush=True)
    history, validation, test = split_rows(rows)
    print(f"[split] history={len(history)} validation={len(validation)} test={len(test)}", flush=True)
    model = SentenceTransformer(MODEL_NAME)
    print("[model] loaded", flush=True)
    chunks = load_chunks()
    index, vectors = build_index(chunks, model)
    print(f"[index] embedded {len(chunks)} chunks", flush=True)
    profile = encode_profile(history, model)
    print(f"[profile] built {len(profile)} learner profiles", flush=True)
    item_diff = difficulty_from_population(history)
    configs = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75)]
    validation_results = {}
    for alpha_value, beta_value in configs:
        rows_for_config = evaluate(validation, profile, chunks, index, vectors, model, item_diff,
                                   alpha_value, beta_value, "validation")
        validation_results[f"alpha={alpha_value},beta={beta_value}"] = metrics(rows_for_config)
    best_key = max(validation_results, key=lambda k: (validation_results[k]["difficulty_mrr"], validation_results[k]["skill_mrr"]))
    known_alpha, known_beta = [float(x.split("=")[1]) for x in best_key.split(",")]
    known_baseline = evaluate(test, profile, chunks, index, vectors, model, item_diff, 1.0, 0.0, "test")
    known_adaptive = evaluate(test, profile, chunks, index, vectors, model, item_diff, known_alpha, known_beta, "test")
    print(f"[evaluation] known_skill alpha={known_alpha} beta={known_beta}", flush=True)
    base_report = {"method": "temporal_vector_profile_rag", "model": MODEL_NAME,
              "split": {"history": len(history), "validation": len(validation), "test": len(test)},
              "profiles": len(profile), "chunks": len(chunks),
              "limitations": ["ASSISTments export has no full problem statement; queries use skill_name and problem_id.",
                              "Corpus is a synthetic instructional scaffold and requires subject-matter review."]}
    FINAL.mkdir(parents=True, exist_ok=True)
    write_json(FINAL / "profiles.json", profile)
    known_report = {**base_report, "track": "known_skill", "primary": True,
                    "selected_config": {"alpha": known_alpha, "beta": known_beta},
                    "validation": validation_results, "traditional_rag": metrics(known_baseline),
                    "adaptive_rag": metrics(known_adaptive)}
    report = {**known_report, "elapsed_seconds": round(time.perf_counter() - started, 3)}
    write_report_files(report, known_baseline, known_adaptive, "known_skill")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
