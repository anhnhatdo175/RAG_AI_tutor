"""Top-k runs of four retrieval systems over the gold set queries (card P2-02), for the P1-07 pools.

Output: one file per system in ``<output_dir>/pool_runs/<system>.csv`` with columns ``query_id, chunk_id, rank, score``
(rank 1 = best), ``depth`` rows per query (``gold.fill_depth``, so ``gold_pool.py`` can fill short pools), and
``pool_runs/manifest.json`` (models, parameters, corpus hash). The format is the one ``lmsrag.labels.gold_pool`` reads.

Shared protocol (every system):

* candidates of a query = ``worked_example`` chunks whose ``skill_tags`` contain the query's ``skill_tag`` (the same
  rule ``gold_pool.py`` uses for its random checks), so systems differ only in how they order the skill's chunks;
* query text = the ``query`` column of ``queries_v1.csv`` after ``QueryRewriter`` (the same sentence for all systems);
* ties are broken by ``chunk_id`` so a run is the same byte for byte on a second call.

Systems (names as in ``configs/data.yaml`` key ``gold.systems``):

``bm25``          Okapi BM25 (k1 = 1.5, b = 0.75), statistics fitted on all worked examples; tokens are lower-case words
                  and numbers, English stop words removed (otherwise "student", "needs", "level" add noise only).
``lsa``           the existing ``LsaEmbedder`` (TF-IDF 1-2 gram + SVD 256, ``random_state = data_seed``) fitted on the
                  text of every chunk of the corpus; cosine similarity.
``bge_ce``        Traditional RAG: ``BAAI/bge-small-en-v1.5`` dense top ``candidate_k`` (query prefix of BGE), then the
                  cross-encoder ``cross-encoder/ms-marco-MiniLM-L-6-v2`` reranks them; score = CE logit.
``adaptive_old``  the Adaptive RAG formula of Chương 2 as implemented in ``layer3_retrieval.AdaptiveRetriever``:
                  ``(1 - beta) * minmax(CE) + beta * profile_zpd`` with beta = ``config.RAG_PROFILE_WEIGHT`` (0.65),
                  ``profile_zpd = (cos(p, v_chunk) + 1) / 2 * zpd_score(difficulty - mastery)`` and the old piecewise
                  ``zpd_score``. No learner exists for a gold query, so the profile vector p is the BGE centroid of the
                  skill's worked examples (a learner who has practised only this skill) and mastery is the query's
                  mastery; no CF term. It is the old method on the new continuous difficulty, kept as the baseline that
                  P2-05 must beat; nothing in it was tuned on the gold set.

``bge_ce`` and ``adaptive_old`` need the HuggingFace hub (models are downloaded once, about 120 MB). Systems that
cannot load their model fail with a clear message; the others still run (``--systems`` selects which).

    python -m lmsrag.retrieval.pool_runs --config configs/data.yaml
    python -m lmsrag.retrieval.pool_runs --systems bm25 lsa          # offline, about 20 seconds
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..datasets.build import REPO, load_config
from ..embedder import LsaEmbedder
from ..layer3_retrieval.query import QueryRewriter
from ..layer4_recommender.recommender import zpd_score

SYSTEMS = ("bm25", "lsa", "bge_ce", "adaptive_old")
RUN_COLUMNS = ["query_id", "chunk_id", "rank", "score"]
BGE_MODEL = "BAAI/bge-small-en-v1.5"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)?")


def tokenize(text: str, stop: frozenset[str] = frozenset()) -> list[str]:
    return [t for t in _TOKEN.findall(str(text).lower()) if t not in stop]


def english_stop_words() -> frozenset[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return frozenset(ENGLISH_STOP_WORDS)


class BM25:
    """Okapi BM25 with the Lucene idf ``log(1 + (N - df + 0.5) / (df + 0.5))`` (never negative)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype="float64")
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str], idx: np.ndarray) -> np.ndarray:
        out = np.zeros(len(idx))
        for j, i in enumerate(idx):
            tf, norm = self.tf[i], self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            out[j] = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in query if t in tf)
        return out


def candidates(queries: pd.DataFrame, worked: pd.DataFrame) -> dict[str, np.ndarray]:
    """Row positions in ``worked`` of the chunks of each query's skill."""
    tags = worked["skill_tags"].map(set)
    return {q.query_id: np.flatnonzero(tags.map(lambda s, t=q.skill_tag: t in s).to_numpy())
            for q in queries.itertuples()}


def top(query_id: str, chunk_ids: np.ndarray, scores: np.ndarray, depth: int) -> pd.DataFrame:
    """The ``depth`` best chunks, score descending then chunk_id ascending (deterministic ties)."""
    order = sorted(range(len(chunk_ids)), key=lambda j: (-float(scores[j]), str(chunk_ids[j])))[:depth]
    return pd.DataFrame({"query_id": query_id, "chunk_id": chunk_ids[order], "rank": np.arange(1, len(order) + 1),
                         "score": np.round(np.asarray(scores, dtype="float64")[order], 6)})


def run_bm25(queries, worked, cand, texts, depth) -> pd.DataFrame:
    stop = english_stop_words()
    bm = BM25([tokenize(t, stop) for t in worked["text"]])
    ids = worked["chunk_id"].to_numpy()
    return pd.concat([top(q.query_id, ids[cand[q.query_id]], bm.scores(tokenize(texts[q.query_id], stop),
                                                                         cand[q.query_id]), depth)
                      for q in queries.itertuples()], ignore_index=True)


def run_lsa(queries, worked, cand, texts, depth, all_texts: list[str], seed: int) -> pd.DataFrame:
    lsa = LsaEmbedder(dim=256, seed=seed).fit(all_texts)
    v = lsa.encode(worked["text"].tolist())
    qv = lsa.encode_queries([texts[q] for q in queries["query_id"]])
    ids = worked["chunk_id"].to_numpy()
    return pd.concat([top(q.query_id, ids[cand[q.query_id]], v[cand[q.query_id]] @ qv[k], depth)
                      for k, q in enumerate(queries.itertuples())], ignore_index=True)


def load_models():
    """BGE bi-encoder and the cross-encoder; a clear error when the hub cannot be reached."""
    import os
    os.environ.setdefault("USE_TF", "0")
    try:
        from ..embedder import BgeEmbedder
        from ..layer3_retrieval.retriever import CrossEncoderReranker
        return BgeEmbedder(BGE_MODEL), CrossEncoderReranker(CE_MODEL)
    except Exception as exc:  # noqa: BLE001 - network, missing package, proxy: all mean "cannot run here"
        raise SystemExit(f"bge_ce/adaptive_old need {BGE_MODEL} and {CE_MODEL} from the HuggingFace hub "
                         f"({type(exc).__name__}: {str(exc)[:200]}). Run them where the hub is reachable "
                         "(own machine or Kaggle), e.g. --systems bge_ce adaptive_old") from exc


def dense_rerank(queries, worked, cand, texts, encoder, reranker, candidate_k: int) -> dict[str, dict]:
    """Per query: positions of the dense top ``candidate_k`` and their CE scores (shared by bge_ce, adaptive_old)."""
    v = encoder.encode(worked["text"].tolist())
    qv = encoder.encode_queries([texts[q] for q in queries["query_id"]])
    out = {}
    for k, q in enumerate(queries.itertuples()):
        idx = cand[q.query_id]
        sim = v[idx] @ qv[k]
        keep = sorted(range(len(idx)), key=lambda j: (-float(sim[j]), str(worked["chunk_id"].iat[idx[j]])))
        keep = idx[keep[:candidate_k]]
        ce = np.asarray(reranker.score(texts[q.query_id], worked["text"].iloc[keep].tolist()), dtype="float64")
        out[q.query_id] = {"idx": keep, "ce": ce, "centroid": v[idx].mean(0)}
    return out, v


def adaptive_old_scores(ce: np.ndarray, cos_profile: np.ndarray, difficulty: np.ndarray, mastery: float,
                        beta: float = config.RAG_PROFILE_WEIGHT) -> np.ndarray:
    """Old Adaptive RAG score (``AdaptiveRetriever.retrieve`` with a profile vector, mastery, no CF, confidence 1)."""
    lo, hi = float(ce.min()), float(ce.max())
    sem = (ce - lo) / (hi - lo) if hi - lo > 1e-12 else np.full(len(ce), 0.5)
    pz = np.clip((cos_profile + 1.0) / 2.0, 0.0, 1.0) * np.array([zpd_score(float(d) - mastery) for d in difficulty])
    return (1.0 - beta) * sem + beta * pz


def run_dense(queries, worked, cand, texts, depth, candidate_k, systems) -> dict[str, pd.DataFrame]:
    encoder, reranker = load_models()
    dr, v = dense_rerank(queries, worked, cand, texts, encoder, reranker, candidate_k)
    ids, diff = worked["chunk_id"].to_numpy(), worked["difficulty"].to_numpy(dtype="float64")
    runs: dict[str, list] = {s: [] for s in systems}
    for q in queries.itertuples():
        r = dr[q.query_id]
        if "bge_ce" in systems:
            runs["bge_ce"].append(top(q.query_id, ids[r["idx"]], r["ce"], depth))
        if "adaptive_old" in systems:
            p = r["centroid"] / max(float(np.linalg.norm(r["centroid"])), 1e-12)
            s = adaptive_old_scores(r["ce"], v[r["idx"]] @ p, diff[r["idx"]], float(q.mastery))
            runs["adaptive_old"].append(top(q.query_id, ids[r["idx"]], s, depth))
    return {s: pd.concat(f, ignore_index=True) for s, f in runs.items()}


def check_run(run: pd.DataFrame, queries: pd.DataFrame, worked: pd.DataFrame, cand: dict, depth: int) -> None:
    """Fail on what ``gold_pool.load_runs`` would reject, and on chunks outside the query's skill."""
    assert list(run.columns) == RUN_COLUMNS
    assert not run.duplicated(["query_id", "chunk_id"]).any()
    allowed = {q: set(worked["chunk_id"].iloc[i]) for q, i in cand.items()}
    assert all(c in allowed[q] for q, c in zip(run["query_id"], run["chunk_id"])), "chunk outside the query skill"
    n = run.groupby("query_id").size()
    expect = pd.Series({q: min(depth, len(i)) for q, i in cand.items()})
    assert set(n.index) == set(queries["query_id"]) and (n == expect[n.index]).all()


def summary(runs: dict[str, pd.DataFrame], queries: pd.DataFrame, worked: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per system, top k: mean |level - target|, share at the target level, and overlap with each other system."""
    lv = worked.set_index("chunk_id")["difficulty_level"]
    tgt = queries.set_index("query_id")["target_level"]
    tops = {s: r[r["rank"] <= k] for s, r in runs.items()}
    rows = []
    for s, t in tops.items():
        gap = (t["chunk_id"].map(lv) - t["query_id"].map(tgt)).abs()
        row = {"system": s, "k": k, "mean_level_gap": round(gap.mean(), 4), "share_at_target": round((gap == 0).mean(), 4)}
        sets = t.groupby("query_id")["chunk_id"].agg(set)
        for o, u in tops.items():
            other = u.groupby("query_id")["chunk_id"].agg(set)
            row[f"overlap_{o}"] = round(float(np.mean([len(sets[q] & other[q]) / k for q in sets.index])), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--systems", nargs="+", choices=SYSTEMS, default=None, help="default: gold.systems of the config")
    ap.add_argument("--corpus", default=None, help="default: <output_dir>/corpus_chunks.jsonl")
    ap.add_argument("--candidate-k", type=int, default=50, help="dense candidates reranked by the cross-encoder")
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    g, seed = cfg["gold"], int(cfg["data_seed"])
    systems = args.systems or list(g["systems"])
    out_root = REPO / cfg["output_dir"]
    corpus = Path(args.corpus) if args.corpus else out_root / "corpus_chunks.jsonl"
    out_dir = out_root / "pool_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    depth = int(g["fill_depth"])

    queries = pd.read_csv(REPO / "lmsrag" / "data" / "gold" / "queries_v1.csv", dtype={"query_id": str})
    chunks = pd.read_json(corpus, lines=True, dtype={"chunk_id": str})
    worked = chunks[chunks["content_type"] == "worked_example"].sort_values("chunk_id").reset_index(drop=True)
    cand = candidates(queries, worked)
    empty = [q for q, i in cand.items() if not len(i)]
    if empty:
        raise SystemExit(f"{len(empty)} queries have no worked example of their skill: {empty[:5]}")
    rw = QueryRewriter()
    texts = {q.query_id: rw.rewrite(q.query) for q in queries.itertuples()}

    t0, runs = time.time(), {}
    if "bm25" in systems:
        runs["bm25"] = run_bm25(queries, worked, cand, texts, depth)
    if "lsa" in systems:
        runs["lsa"] = run_lsa(queries, worked, cand, texts, depth, chunks["text"].tolist(), seed)
    dense = [s for s in systems if s in ("bge_ce", "adaptive_old")]
    if dense:
        runs.update(run_dense(queries, worked, cand, texts, depth, args.candidate_k, dense))

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    sha = hashlib.sha256(corpus.read_bytes()).hexdigest()[:16]
    for s, r in runs.items():
        check_run(r, queries, worked, cand, depth)
        r.to_csv(out_dir / f"{s}.csv", index=False)
        manifest[s] = {"rows": len(r), "queries": int(r["query_id"].nunique()), "depth": depth, "corpus_sha256_16": sha,
                       "data_seed": seed, "candidate_k": args.candidate_k if s in dense else None,
                       "models": {"bge_ce": [BGE_MODEL, CE_MODEL], "adaptive_old": [BGE_MODEL, CE_MODEL]}.get(s),
                       "beta": config.RAG_PROFILE_WEIGHT if s == "adaptive_old" else None}
        print(f"{s}: {len(r)} rows, {r['query_id'].nunique()} queries -> {out_dir / (s + '.csv')}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    present = {s: pd.read_csv(out_dir / f"{s}.csv", dtype={"query_id": str, "chunk_id": str})
               for s in SYSTEMS if (out_dir / f"{s}.csv").is_file()}
    table = summary(present, queries, worked, int(g["top_k"]))
    tables = REPO / cfg.get("tables_dir", "reports/tables")
    table.to_csv(tables / "b3_pool_runs_summary.csv", index=False)
    print(table.to_string(index=False))
    print(f"done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
