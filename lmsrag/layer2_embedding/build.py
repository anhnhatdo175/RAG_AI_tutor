"""Run LAYER 2 end-to-end:  Parse -> Clean -> Tag -> Chunk -> Embed -> Index (+ diagnostics).   python -m lmsrag.layer2_embedding.build"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np

from .. import config
from ..skill_mapping import SkillMapper
from . import chunk, clean, embed, index, parse


def build(embedder: str = "lsa") -> dict:
    t0 = time.time()
    mapper = SkillMapper()
    raw = list(parse.iter_records())
    recs, clean_report = clean.clean_records(raw)

    chunks = []
    for r in recs:
        tag, label, conf = mapper.tag_text(r.question.lower(), r.subject, r.level, r.is_gsm8k)
        for c in chunk.make_chunks(r):
            c.update(skill_tag=tag or "unmapped", topic_label=label or "unmapped", tag_confidence=conf or "none",
                     difficulty=r.difficulty, source=r.source, subject=r.subject, level=r.level, grade=parse.infer_grade(r),
                     has_figure=r.extra.get("has_figure", False), split=r.extra.get("split"), schema_version=index.SCHEMA_VERSION)
            chunks.append(c)

    layer = embed.EmbeddingLayer(embedder)
    V = layer.fit_encode_docs([c["text"] for c in chunks])
    layer.save(config.ART / "embedder_lsa.pkl")
    manifest = index.save_index(chunks, V, {"kind": embedder, "model": getattr(layer.model, "name", embedder),
                                            "fingerprint": layer.fingerprint([c["text"] for c in chunks])})
    store, _ = index.load_and_verify()

    tags = np.array([c["skill_tag"] for c in chunks])
    tok = np.array([c["token_count"] for c in chunks])
    # unfiltered topic-query sanity: how often does plain semantic search land in the right skill?
    prec = {}
    for t in sorted(set(tags) - {"unmapped"}):
        if (tags == t).sum() < config.MIN_SKILL_DOCS:
            continue
        q = layer.encode_queries([mapper.topic_label(t)])[0]
        hits = store.search(q, k=5)
        prec[t] = float(np.mean([store.meta[i]["skill_tag"] == t for i, _ in hits]))
    report = {"embedder": embedder, "n_input_records": len(raw), "clean": clean_report, "n_records": len(recs), "n_chunks": len(chunks),
              "chunks_per_record": {str(k): v for k, v in sorted(Counter(c["n_chunks"] for c in chunks).items())},
              "token_count": {"mean": float(tok.mean()), "p95": float(np.percentile(tok, 95)), "max": int(tok.max()),
                              "over_512": int((tok > chunk.MAX_TOKENS).sum())},
              "tokenizer": "tiktoken" if chunk._tiktoken() else "regex-proxy",
              "unit_norm": layer.norm_report, "geometry": embed.geometry_report(V, tags),
              "index_self_retrieval_acc": index.self_retrieval_check(store),
              "topic_query_prec5_unfiltered": {"mean": float(np.mean(list(prec.values()))), "by_tag": prec},
              "manifest": manifest, "elapsed_s": round(time.time() - t0, 1)}
    (config.ART / "layer2_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--embedder", default="lsa", choices=["lsa", "bge"])
    r = build(ap.parse_args().embedder)
    print(json.dumps({k: v for k, v in r.items() if k not in ("manifest", "topic_query_prec5_unfiltered")}, indent=1))
    print("topic-query P@5 (unfiltered):", round(r["topic_query_prec5_unfiltered"]["mean"], 3))
