"""Colab-friendly Traditional RAG and Adaptive Profile-aware RAG pipeline.

This module uses the generated ASSISTments corpus, learner profiles, and
manifest in this directory. It deliberately separates:

1. document chunking and metadata persistence;
2. embedding and FAISS indexing;
3. metadata filtering;
4. semantic retrieval;
5. profile-aware reranking;
6. alpha/beta experiments and error analysis.

The answer generator is optional. Retrieval and ranking experiments can run
without a generative model, which makes debugging and evaluation reproducible.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "artifacts"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DIFFICULTIES = ("beginner", "intermediate", "advanced")
CONTENT_TYPES = (
    "concept",
    "worked_example",
    "misconception",
    "strategy",
    "transfer_practice",
)


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
    target_student_ids: list[str]


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    semantic_score: float
    profile_score: float
    skill_id: str
    difficulty: str
    content_type: str
    source_file: str


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---"):
        raise ValueError("Material is missing YAML frontmatter.")
    end = text.find("\n---", 3)
    if end < 0:
        raise ValueError("Material frontmatter is not closed.")
    result: dict[str, Any] = {}
    for line in text[4:end].splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        raw = raw.strip()
        if raw.startswith("["):
            result[key.strip()] = json.loads(raw)
        else:
            result[key.strip()] = json.loads(raw)
    return result


def split_sections(body: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", body))
    if not matches:
        return [("document", body.strip())]
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        content = body[match.end() : end].strip()
        if content:
            sections.append((match.group(1).strip(), content))
    return sections


def chunk_materials(
    materials_dir: Path,
    manifest_path: Path,
    max_chars: int = 1400,
) -> list[Chunk]:
    manifest = read_json(manifest_path)
    chunks: list[Chunk] = []
    for doc in manifest:
        material_path = materials_dir.parent / doc["file"]
        raw = material_path.read_text(encoding="utf-8")
        frontmatter = parse_frontmatter(raw)
        body = raw[raw.find("\n---", 3) + 4 :].strip()
        sections = split_sections(body)
        for section_name, section_text in sections:
            text = f"{section_name}\n{section_text}".strip()
            if len(text) <= max_chars:
                pieces = [text]
            else:
                words = text.split()
                pieces, current = [], []
                length = 0
                for word in words:
                    if current and length + len(word) + 1 > max_chars:
                        pieces.append(" ".join(current))
                        current, length = [], 0
                    current.append(word)
                    length += len(word) + 1
                if current:
                    pieces.append(" ".join(current))
            for piece_index, piece in enumerate(pieces):
                chunk_id = f"{doc['doc_id']}__{section_name.lower().replace(' ', '-')}-{piece_index}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc["doc_id"],
                        text=piece,
                        skill_id=str(frontmatter["skill_id"]),
                        skill_name=frontmatter["skill_name"],
                        difficulty=frontmatter["difficulty"],
                        content_type=frontmatter["content_type"],
                        source_file=doc["file"],
                        target_student_ids=frontmatter.get("target_student_ids", []),
                    )
                )
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as handle:
        return [Chunk(**json.loads(line)) for line in handle if line.strip()]


def build_index(
    chunks: list[Chunk],
    model_name: str,
    index_path: Path,
    embeddings_path: Path,
    model: SentenceTransformer | None = None,
) -> None:
    model = model or SentenceTransformer(model_name)
    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    np.save(embeddings_path, embeddings)


def load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    return read_json(path)


def profile_lookup(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["skill_id"]): item for item in profile["skills"]}


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def infer_skill_id(query: str, chunks: list[Chunk]) -> str | None:
    query_lower = query.lower()
    candidates = [
        chunk
        for chunk in chunks
        if chunk.skill_name.lower() in query_lower
        or f"skill {chunk.skill_id}" in query_lower
    ]
    return candidates[0].skill_id if candidates else None


def metadata_matches(
    chunk: Chunk,
    skill_id: str | None = None,
    difficulty: str | None = None,
    content_type: str | None = None,
) -> bool:
    return (
        (skill_id is None or chunk.skill_id == str(skill_id))
        and (difficulty is None or chunk.difficulty == difficulty)
        and (content_type is None or chunk.content_type == content_type)
    )


def profile_match_score(
    chunk: Chunk,
    profile: dict[str, Any] | None,
    query_skill_id: str | None,
) -> float:
    if profile is None:
        return 0.0
    skills = profile_lookup(profile)
    target = skills.get(chunk.skill_id)
    if target is None:
        return 0.0
    score = 0.0
    if query_skill_id and chunk.skill_id == query_skill_id:
        score += 0.35
    recommended = target["recommended_material_difficulty"]
    distance = abs(DIFFICULTIES.index(chunk.difficulty) - DIFFICULTIES.index(recommended))
    score += 0.55 * (1.0 - distance / 2.0)
    content_bonus = {
        "beginner": {"misconception": 0.10, "worked_example": 0.08},
        "intermediate": {"strategy": 0.10, "worked_example": 0.06},
        "advanced": {"transfer_practice": 0.10, "strategy": 0.06},
    }
    score += content_bonus.get(target["mastery_level"], {}).get(chunk.content_type, 0.0)
    return min(score, 1.0)


def retrieve(
    query: str,
    index: faiss.Index,
    chunks: list[Chunk],
    model: SentenceTransformer,
    top_k: int = 10,
    profile: dict[str, Any] | None = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    skill_id: str | None = None,
    difficulty: str | None = None,
    content_type: str | None = None,
) -> list[Hit]:
    query_vector = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    candidate_k = min(index.ntotal, max(top_k * 10, top_k))
    semantic_scores, ids = index.search(query_vector, candidate_k)
    query_skill_id = skill_id or infer_skill_id(query, chunks)
    hits: list[Hit] = []
    for semantic_score, row_id in zip(semantic_scores[0], ids[0]):
        if row_id < 0:
            continue
        chunk = chunks[int(row_id)]
        if not metadata_matches(chunk, skill_id, difficulty, content_type):
            continue
        profile_score = profile_match_score(chunk, profile, query_skill_id)
        score = alpha * float(semantic_score) + beta * profile_score
        hits.append(
            Hit(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=score,
                semantic_score=float(semantic_score),
                profile_score=profile_score,
                skill_id=chunk.skill_id,
                difficulty=chunk.difficulty,
                content_type=chunk.content_type,
                source_file=chunk.source_file,
            )
        )
    hits.sort(key=lambda hit: hit.score, reverse=True)
    return hits[:top_k]


def traditional_rag(
    query: str,
    index: faiss.Index,
    chunks: list[Chunk],
    model: SentenceTransformer,
    top_k: int = 5,
    **filters: str | None,
) -> list[Hit]:
    """Traditional RAG: semantic retrieval without learner-profile scoring."""
    return retrieve(
        query,
        index,
        chunks,
        model,
        top_k=top_k,
        alpha=1.0,
        beta=0.0,
        **filters,
    )


def adaptive_rag(
    query: str,
    profile: dict[str, Any],
    index: faiss.Index,
    chunks: list[Chunk],
    model: SentenceTransformer,
    top_k: int = 5,
    alpha: float = 0.5,
    beta: float = 0.5,
    **filters: str | None,
) -> list[Hit]:
    """Adaptive RAG: semantic retrieval reranked with the learner profile."""
    return retrieve(
        query,
        index,
        chunks,
        model,
        top_k=top_k,
        profile=profile,
        alpha=alpha,
        beta=beta,
        **filters,
    )


def context_from_hits(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{hit.doc_id} | skill={hit.skill_id} | difficulty={hit.difficulty}]\n{hit.text}"
        for hit in hits
    )


def exact_answer_match(answer: str, gold: str) -> bool:
    return normalize(answer) == normalize(gold)


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_alpha_beta_experiment(
    questions: list[dict[str, Any]],
    index: faiss.Index,
    chunks: list[Chunk],
    profiles: dict[str, dict[str, Any]],
    model: SentenceTransformer,
    top_k: int,
    alphas: Iterable[float],
    betas: Iterable[float],
) -> list[dict[str, Any]]:
    """Compare ranking weights on one shared, unfiltered candidate pool.

    ``skill_id`` may be supplied as a gold/evaluation label, but
    ``expected_difficulty`` is never passed as a retrieval filter. Otherwise
    the experiment would reveal the answer to both systems before reranking.
    """
    rows = []
    for alpha in alphas:
        for beta in betas:
            for question in questions:
                profile = profiles.get(str(question["student_id"]))
                expected_difficulty = question.get("difficulty")
                if expected_difficulty is None and profile:
                    skill_profile = profile_lookup(profile).get(
                        str(question.get("skill_id", ""))
                    )
                    if skill_profile:
                        expected_difficulty = skill_profile[
                            "recommended_material_difficulty"
                        ]
                hits = retrieve(
                    question["question"],
                    index,
                    chunks,
                    model,
                    top_k=top_k,
                    profile=profile,
                    alpha=alpha,
                    beta=beta,
                    skill_id=question.get("skill_id"),
                    difficulty=None,
                )
                gold_skill = str(question.get("skill_id", ""))
                gold_hit = any(hit.skill_id == gold_skill for hit in hits)
                gold_difficulty = expected_difficulty
                difficulty_hit = (
                    gold_difficulty is None
                    or any(
                        hit.skill_id == gold_skill
                        and hit.difficulty == gold_difficulty
                        for hit in hits
                    )
                )
                rows.append(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "student_id": question["student_id"],
                        "question": question["question"],
                        "gold_skill_id": gold_skill,
                        "expected_difficulty": expected_difficulty,
                        "top_doc_ids": [hit.doc_id for hit in hits],
                        "top_difficulties": [hit.difficulty for hit in hits],
                        "top_content_types": [hit.content_type for hit in hits],
                        "retrieval_hit": int(gold_hit),
                        "difficulty_hit": int(difficulty_hit),
                        "skill_rank": next(
                            (
                                index + 1
                                for index, hit in enumerate(hits)
                                if hit.skill_id == gold_skill
                            ),
                            None,
                        ),
                        "difficulty_rank": next(
                            (
                                index + 1
                                for index, hit in enumerate(hits)
                                if hit.skill_id == gold_skill
                                and hit.difficulty == gold_difficulty
                            ),
                            None,
                        ),
                    }
                )
    return rows


def error_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row for row in rows if not row["retrieval_hit"]]
    by_difficulty = Counter(
        row.get("expected_difficulty", "unknown") for row in errors
    )
    error_types = Counter()
    for row in rows:
        if not row["retrieval_hit"]:
            error_types["wrong_skill"] += 1
        elif not row["difficulty_hit"]:
            error_types["right_skill_wrong_difficulty"] += 1
        else:
            error_types["success"] += 1
    by_setting: dict[str, dict[str, float]] = {}
    for alpha in sorted({row["alpha"] for row in rows}, reverse=True):
        for beta in sorted({row["beta"] for row in rows}, reverse=True):
            setting_rows = [
                row
                for row in rows
                if row["alpha"] == alpha and row["beta"] == beta
            ]
            key = f"alpha={alpha},beta={beta}"
            by_setting[key] = {
                "skill_recall_at_k": sum(
                    row["retrieval_hit"] for row in setting_rows
                )
                / len(setting_rows),
                "difficulty_match_at_k": sum(
                    row["difficulty_hit"] for row in setting_rows
                )
                / len(setting_rows),
            }
    return {
        "total_cases": len(rows),
        "errors": len(errors),
        "error_rate": len(errors) / len(rows) if rows else 0.0,
        "skill_recall_at_k": (
            sum(row["retrieval_hit"] for row in rows) / len(rows) if rows else 0.0
        ),
        "difficulty_match_at_k": (
            sum(row["difficulty_hit"] for row in rows) / len(rows) if rows else 0.0
        ),
        "errors_by_expected_difficulty": dict(by_difficulty),
        "error_types": dict(error_types),
        "by_alpha_beta": by_setting,
        "examples": errors[:50],
    }


def cmd_build(args: argparse.Namespace) -> None:
    chunks = chunk_materials(
        args.materials,
        args.manifest,
        max_chars=args.max_chars,
    )
    save_chunks(chunks, args.chunks)
    model = SentenceTransformer(args.model)
    build_index(chunks, args.model, args.index, args.embeddings, model)
    print(json.dumps({"chunks": len(chunks), "index_size": len(chunks)}))


def cmd_retrieve(args: argparse.Namespace) -> None:
    chunks = load_chunks(args.chunks)
    profiles = load_profiles(args.profiles)
    model = SentenceTransformer(args.model)
    index = faiss.read_index(str(args.index))
    profile = profiles.get(str(args.student_id)) if args.student_id else None
    hits = retrieve(
        args.query,
        index,
        chunks,
        model,
        top_k=args.top_k,
        profile=profile,
        alpha=args.alpha,
        beta=args.beta,
        skill_id=args.skill_id,
        difficulty=args.difficulty,
        content_type=args.content_type,
    )
    print(json.dumps([asdict(hit) for hit in hits], ensure_ascii=False, indent=2))


def cmd_experiment(args: argparse.Namespace) -> None:
    chunks = load_chunks(args.chunks)
    profiles = load_profiles(args.profiles)
    questions = load_questions(args.questions)
    model = SentenceTransformer(args.model)
    index = faiss.read_index(str(args.index))
    rows = run_alpha_beta_experiment(
        questions,
        index,
        chunks,
        profiles,
        model,
        args.top_k,
        [float(value) for value in args.alphas.split(",")],
        [float(value) for value in args.betas.split(",")],
    )
    write_json(args.results, rows)
    write_json(args.error_report, error_analysis(rows))
    print(json.dumps(error_analysis(rows), ensure_ascii=False, indent=2))


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--materials", type=Path, default=ROOT / "materials")
    build.add_argument("--manifest", type=Path, default=ROOT / "corpus_manifest.json")
    build.add_argument("--model", default=DEFAULT_MODEL)
    build.add_argument("--max-chars", type=int, default=1400)
    build.add_argument("--chunks", type=Path, default=DEFAULT_OUTPUT / "chunks.jsonl")
    build.add_argument("--index", type=Path, default=DEFAULT_OUTPUT / "faiss.index")
    build.add_argument("--embeddings", type=Path, default=DEFAULT_OUTPUT / "embeddings.npy")
    build.set_defaults(func=cmd_build)

    retrieve_parser = subparsers.add_parser("retrieve")
    retrieve_parser.add_argument("--query", required=True)
    retrieve_parser.add_argument("--student-id")
    retrieve_parser.add_argument("--skill-id")
    retrieve_parser.add_argument("--difficulty", choices=DIFFICULTIES)
    retrieve_parser.add_argument("--content-type", choices=CONTENT_TYPES)
    retrieve_parser.add_argument("--top-k", type=int, default=5)
    retrieve_parser.add_argument("--alpha", type=float, default=1.0)
    retrieve_parser.add_argument("--beta", type=float, default=0.0)
    retrieve_parser.add_argument("--model", default=DEFAULT_MODEL)
    retrieve_parser.add_argument("--chunks", type=Path, default=DEFAULT_OUTPUT / "chunks.jsonl")
    retrieve_parser.add_argument("--index", type=Path, default=DEFAULT_OUTPUT / "faiss.index")
    retrieve_parser.add_argument("--profiles", type=Path, default=ROOT / "learner_profiles.json")
    retrieve_parser.set_defaults(func=cmd_retrieve)

    experiment = subparsers.add_parser("experiment")
    experiment.add_argument("--questions", type=Path, required=True)
    experiment.add_argument("--alphas", default="1.0,0.75,0.5,0.25")
    experiment.add_argument("--betas", default="0.0,0.25,0.5,0.75,1.0")
    experiment.add_argument("--top-k", type=int, default=5)
    experiment.add_argument("--model", default=DEFAULT_MODEL)
    experiment.add_argument("--chunks", type=Path, default=DEFAULT_OUTPUT / "chunks.jsonl")
    experiment.add_argument("--index", type=Path, default=DEFAULT_OUTPUT / "faiss.index")
    experiment.add_argument("--profiles", type=Path, default=ROOT / "learner_profiles.json")
    experiment.add_argument("--results", type=Path, default=DEFAULT_OUTPUT / "alpha_beta_results.json")
    experiment.add_argument("--error-report", type=Path, default=DEFAULT_OUTPUT / "error_analysis.json")
    experiment.set_defaults(func=cmd_experiment)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    namespace = parser.parse_args()
    namespace.func(namespace)
