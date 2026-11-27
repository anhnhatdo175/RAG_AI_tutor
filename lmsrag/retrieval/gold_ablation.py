"""Ablation of the adaptive retrieval score on the frozen gold set (card P2-07, review C3).

The system ablated is ``adaptive_old`` (Chương 2, ``pool_runs.adaptive_old_scores``), the best system on gold-v1
(``b3_gold_eval.csv``); the LTR of P2-05 is not ablated here because it did not transfer to the gold set. Each
variant re-ranks the **judged pool** of every gold query (all chunks labelled, so no pool bias), nothing is tuned:
beta = ``config.RAG_PROFILE_WEIGHT`` as in the code.

    score = (1 - beta) * minmax(semantic) + beta * profile * zpd

``full``              as above, ``zpd`` = old piecewise ``zpd_score(difficulty - mastery)``
``no_semantic``       beta = 1 (no cross-encoder / semantic term)
``no_zpd``            zpd = 1
``no_profile``        profile = 1 (cos to the skill centroid dropped)
``swap_encoder``      semantic term from another encoder (see ``--encoder``)
``swap_zpd_phat``     zpd = 1 - |p_hat - 0.7| / 0.7, p_hat = sigmoid(logit mastery - logit(1 - e)) as in P2-05
``swap_zpd_level``    zpd = 1 - |difficulty_level - target_level| / 4 (the gold query rule). Written after
                      ``b3_gold_eval.csv`` showed the level gap alone ranks well: report it as exploratory.
``semantic_only``     minmax(semantic) alone (reference)

``--encoder bge`` (needs the HuggingFace hub, run on your own machine): semantic = cross-encoder logit, profile = BGE
cosine to the skill centroid, ``swap_encoder`` = BGE bi-encoder cosine instead of the cross-encoder.
``--encoder lexical`` (offline): semantic = BM25, profile = LSA cosine, ``swap_encoder`` = LSA cosine instead of BM25.

CF is not ablated here: a gold query has no learner (CF is ablated in ``lmsrag.ranking.ablation``).
Paired bootstrap over the 180 queries (1,000 resamples), each variant against ``full``, Holm.

    python -m lmsrag.retrieval.gold_ablation --encoder lexical
    python -m lmsrag.retrieval.gold_ablation --encoder bge

Output ``reports/tables/c3_ablation_retrieval_<encoder>.csv``.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config
from ..datasets.build import REPO, load_config
from ..layer4_recommender.recommender import zpd_score
from . import gold_eval as G
from . import pool_runs as P

VARIANTS = ["full", "no_semantic", "no_zpd", "no_profile", "swap_encoder", "swap_zpd_phat", "swap_zpd_level",
            "semantic_only"]


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.full(len(x), 0.5)


def variant_scores(sem, sem_alt, cos, difficulty, level, e, mastery: float, target: int,
                   beta: float = config.RAG_PROFILE_WEIGHT) -> dict[str, np.ndarray]:
    prof = np.clip((cos + 1.0) / 2.0, 0.0, 1.0)
    z_old = np.array([zpd_score(float(d) - mastery) for d in difficulty])
    m = np.clip(mastery, 0.02, 0.98)
    p = 1 / (1 + np.exp(-(np.log(m / (1 - m)) - np.log((1 - e) / e))))
    z_p = np.clip(1 - np.abs(p - 0.7) / 0.7, 0, 1)
    z_l = 1 - np.abs(level - target) / 4
    s, s2 = minmax(sem), minmax(sem_alt)
    mix = lambda sm, pr, z: (1 - beta) * sm + beta * pr * z  # noqa: E731
    return {"full": mix(s, prof, z_old), "no_semantic": prof * z_old, "no_zpd": mix(s, prof, 1.0),
            "no_profile": mix(s, 1.0, z_old), "swap_encoder": mix(s2, prof, z_old), "swap_zpd_phat": mix(s, prof, z_p),
            "swap_zpd_level": mix(s, prof, z_l), "semantic_only": s}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--encoder", choices=["lexical", "bge"], default="lexical")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    cfg = load_config(REPO / args.config)
    seed = int(cfg["data_seed"])
    out_root = REPO / cfg["output_dir"]
    tables = REPO / cfg.get("tables_dir", "reports/tables")
    gold = G.load_gold()
    chunks = pd.read_json(out_root / "corpus_chunks.jsonl", lines=True, dtype={"chunk_id": str})
    worked = chunks[chunks["content_type"] == "worked_example"].sort_values("chunk_id").reset_index(drop=True)
    queries = pd.read_csv(REPO / "lmsrag" / "data" / "gold" / "queries_v1.csv", dtype={"query_id": str})
    cand = P.candidates(queries, worked)
    queries = queries.set_index("query_id")
    pos = pd.Series(np.arange(len(worked)), index=worked["chunk_id"])
    from ..embedder import LsaEmbedder
    from ..layer3_retrieval.query import QueryRewriter
    rw = QueryRewriter()
    stop = P.english_stop_words()
    e = pd.to_numeric(worked.get("p_correct_hist"), errors="coerce").to_numpy(dtype=float)
    e = np.clip(np.where(np.isnan(e), 1 - worked["difficulty"].to_numpy(dtype=float), e), 0.02, 0.98)
    diff = worked["difficulty"].to_numpy(dtype=float)
    level = worked["difficulty_level"].to_numpy(dtype=float)
    lsa = LsaEmbedder(dim=256, seed=seed).fit(chunks["text"].tolist())
    v_lsa = lsa.encode(worked["text"].tolist())
    if args.encoder == "bge":
        encoder, reranker = P.load_models()
        v_enc = encoder.encode(worked["text"].tolist())
    else:
        bm = P.BM25([P.tokenize(t, stop) for t in worked["text"]])
        v_enc = v_lsa
    rows = []
    for q, g in gold.groupby("query_id"):
        g = g.sort_values("chunk_id")
        qq = queries.loc[q]
        text = rw.rewrite(qq["query"])
        idx = pos.loc[g["chunk_id"]].to_numpy()
        centroid = v_enc[cand[q]].mean(0)
        cos = v_enc[idx] @ (centroid / max(float(np.linalg.norm(centroid)), 1e-12))
        if args.encoder == "bge":
            sem = np.asarray(reranker.score(text, worked["text"].iloc[idx].tolist()), dtype=float)
            sem_alt = v_enc[idx] @ encoder.encode_queries([text])[0]
        else:
            sem = bm.scores(P.tokenize(text, stop), idx)
            sem_alt = v_lsa[idx] @ lsa.encode_queries([text])[0]
        sc = variant_scores(sem, sem_alt, cos, diff[idx], level[idx], e[idx], float(qq["mastery"]),
                            int(qq["target_level"]))
        judged = dict(zip(g["chunk_id"], g["label_final"].astype(int)))
        ids = g["chunk_id"].to_numpy()
        for name, s in sc.items():
            m = G.query_metrics(ids[np.lexsort((ids, -s))].tolist(), judged)
            rows.append({"view": "pool_rerank", "system": name, "query_id": q,
                         **{k: m[k] for k in ("ndcg@5", "ndcg@10", "p@5")}})
    per_q = pd.DataFrame(rows)
    pairs = [(v, "full") for v in VARIANTS if v != "full"]
    res = G.paired(per_q, "pool_rerank", pairs, ["ndcg@5", "ndcg@10"], args.n_boot, seed)
    res = res.drop(columns="view").rename(columns={"system": "variant", "mean_a": "value", "mean_b": "full"})
    res.insert(0, "encoder", args.encoder)
    res.round(4).to_csv(tables / f"c3_ablation_retrieval_{args.encoder}.csv", index=False)
    with pd.option_context("display.width", 200):
        print(res.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
