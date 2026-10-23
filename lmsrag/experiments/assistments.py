"""Prequential learner-score evaluation on the ASSISTments temporal split."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from scipy import sparse
from scipy.stats import ttest_ind, ttest_rel

from .. import config, data, profile
from ..layer4_recommender import SkillCollaborativeFilter
from ..layer3_retrieval.retriever import load_query_encoder, ranking_metrics
from ..skill_mapping import SkillMapper


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    prediction = np.clip(np.asarray(prediction, dtype="float64"), 0.0, 1.0)
    try:
        auc: float | None = float(roc_auc_score(y, prediction))
    except ValueError:
        auc = None
    return {
        "AUC": auc,
        "RMSE": float(np.sqrt(mean_squared_error(y, prediction))),
        "MAE": float(mean_absolute_error(y, prediction)),
    }


def evaluate_ranking_runs(
    rankings: dict[str, dict[str, list[str]]],
    relevant_by_query: dict[str, set[str]],
    *,
    k: int = 5,
) -> dict[str, dict[str, float | int]]:
    """Aggregate ranking metrics for named baselines on the same ASSISTments queries.

    `rankings` maps strategy -> query key -> ordered candidate IDs.  Relevance labels
    must be prepared from held-out interactions; callers should not include history or
    validation rows when creating them.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not rankings or not relevant_by_query:
        raise ValueError("rankings and relevance labels must not be empty")
    output: dict[str, dict[str, float | int]] = {}
    for strategy, query_rankings in rankings.items():
        query_keys = sorted(set(query_rankings) & set(relevant_by_query))
        if not query_keys:
            raise ValueError(f"strategy {strategy!r} has no queries with relevance labels")
        per_query = [
            ranking_metrics(query_rankings[key], relevant_by_query[key], k=k)
            for key in query_keys
        ]
        metric_names = per_query[0].keys()
        output[strategy] = {
            "n_queries": len(query_keys),
            **{
                name: float(np.mean([metrics[name] for metrics in per_query]))
                for name in metric_names
            },
        }
    return output


def sweep_hybrid_weights(
    recommender_factory: Callable[[float, float], object],
    queries: dict[str, dict],
    relevant_by_query: dict[str, set[str]],
    weight_pairs: list[tuple[float, float]],
    *,
    k: int = 5,
) -> list[dict]:
    """Run a reproducible alpha/beta sweep against held-out ranking relevance.

    The factory receives `(alpha, beta)` and must return a recommender exposing
    `recommend(**query)`. Each query value contains that method's keyword arguments.
    """
    if not weight_pairs:
        raise ValueError("weight_pairs must not be empty")
    reports = []
    for alpha, beta in weight_pairs:
        if (not np.isfinite(alpha) or not np.isfinite(beta) or alpha < 0 or beta < 0
                or alpha + beta > 1):
            raise ValueError("each weight pair must be non-negative and sum to at most 1")
        recommender = recommender_factory(float(alpha), float(beta))
        ranked = {}
        for query_key, query in queries.items():
            arguments = dict(query)
            arguments.update(strategy="hybrid", top_k=k)
            ranked[query_key] = [
                str(item["item_id"]) for item in recommender.recommend(**arguments)
            ]
        metrics = evaluate_ranking_runs(
            {"hybrid": ranked}, relevant_by_query, k=k
        )["hybrid"]
        reports.append({"alpha": float(alpha), "beta": float(beta), **metrics})
    return reports


def evaluate_recommender_weight_sweep(
    *,
    seed: int = 0,
    k: int = 5,
    min_bucket_interactions: int = 2,
) -> dict:
    """Tune recommender alpha/beta on validation and evaluate the winner on test.

    ASSISTments has no shared item IDs with the external exercise bank, so relevance
    is a transparent proxy: for each learner/canonical-skill query, the relevant
    candidate difficulty is the difficulty bucket whose held-out success rate is
    closest to TARGET_SUCCESS. Only buckets with enough held-out interactions are
    considered. This measures future difficulty alignment, not item preference.
    """
    from ..layer1_data.learner_state import LearnerState, build_states
    from ..layer1_data.sources import filter_exercise_bank
    from ..layer2_embedding.index import load_and_verify
    from . import profile
    from ..layer4_recommender import HybridRecommender

    if k < 1 or min_bucket_interactions < 1:
        raise ValueError("k and min_bucket_interactions must be positive")

    mapper = SkillMapper()
    all_frame = data.load_logs(mapper)
    frame = data.filter_mapped_logs(all_frame)
    mapping_scope = {
        "all_interactions": int(len(all_frame)),
        "mapped_interactions_used": int(len(frame)),
        "excluded_unmapped_interactions": int(len(all_frame) - len(frame)),
        "mapped_share": float(len(frame) / len(all_frame)) if len(all_frame) else None,
    }
    ema_alpha, _ = profile.tune_alpha(frame)
    priors, tau_base = profile.skill_priors(frame)
    validation_states = build_states(
        frame, ema_alpha, priors, as_of=("history",), tau_base=tau_base
    )
    test_states = build_states(
        frame, ema_alpha, priors, as_of=("history", "val"), tau_base=tau_base
    )

    bank_path = config.ART / "exercise_bank.jsonl"
    exercise_bank = [
        json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_skill_grade: dict[str, Counter] = defaultdict(Counter)
    for item in exercise_bank:
        if item.get("mapping_status") == "mapped":
            by_skill_grade[str(item.get("skill_tag"))][str(item.get("grade"))] += 1
    grade_for_skill = {
        skill: sorted(grades.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
        for skill, grades in by_skill_grade.items() if grades
    }


    pool_by_skill = {
        skill: filter_exercise_bank(
            exercise_bank, config.SUBJECT_DEFAULT, grade, skill
        )
        for skill, grade in grade_for_skill.items()
    }
    pool_by_skill = {
        skill: pool for skill, pool in pool_by_skill.items() if len(pool) >= k
    }
    if not pool_by_skill:
        raise ValueError("No mapped exercise pools are large enough for ranking evaluation")

    store, _ = load_and_verify()
    encoder = load_query_encoder()
    skill_texts = sorted(pool_by_skill)
    vectors = np.asarray(encoder.encode_queries(skill_texts), dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] != len(skill_texts):
        raise ValueError("query encoder did not return one vector per canonical skill")
    vector_by_skill = {skill: vectors[index] for index, skill in enumerate(skill_texts)}
    interactions = frame[["user_id", "tag", "correct", "split", "problem_id", "bucket"]]
    recommender = HybridRecommender(
        store, exercise_bank, json.loads(config.SKILL_CATALOG_JSON.read_text(encoding="utf-8")),
        interactions, alpha=config.RANK_SEMANTIC_WEIGHT, beta=config.RANK_PROFILE_WEIGHT,
    )

    def prepare_queries(split: str, states: dict[int, LearnerState]) -> tuple[list[dict], dict[str, set[str]], dict]:
        heldout = frame.loc[
            frame["split"].eq(split)
            & frame["tag"].isin(pool_by_skill)
        ]
        relevant: dict[str, set[str]] = {}
        query_specs: list[dict] = []
        skipped = Counter()
        grouped = heldout.groupby(["user_id", "tag"], sort=True)
        for (user_id, skill), rows in grouped:
            bucket_stats = rows.groupby("bucket")["correct"].agg(["size", "mean"])
            supported = bucket_stats[bucket_stats["size"] >= min_bucket_interactions]
            if len(supported) < 2:
                skipped["fewer_than_two_supported_buckets"] += 1
                continue
            target_bucket = min(
                (int(bucket) for bucket in supported.index),
                key=lambda bucket: (
                    abs(float(supported.loc[bucket, "mean"]) - config.TARGET_SUCCESS),
                    bucket,
                ),
            )
            skill_pool = pool_by_skill[str(skill)]
            relevant_ids = {
                str(item["item_id"]) for item in skill_pool
                if int(np.argmin(np.abs(
                    np.asarray(config.DIFFICULTY_LEVELS, dtype="float32")
                    - float(item["difficulty"])
                ))) == target_bucket
            }
            if not relevant_ids:
                skipped["no_candidate_in_relevant_bucket"] += 1
                continue
            query_key = f"{int(user_id)}::{skill}"
            state = states.get(int(user_id), LearnerState(user_id=int(user_id)))
            query_specs.append({
                "key": query_key,
                "user_id": int(user_id),
                "state": state,
                "skill": str(skill),
                "grade": grade_for_skill[str(skill)],
                "query_vector": vector_by_skill[str(skill)],
                "pool": skill_pool,
            })
            relevant[query_key] = relevant_ids
        return query_specs, relevant, {
            "rows": int(len(heldout)),
            "queries": len(query_specs),
            "skipped": dict(skipped),
        }

    validation_queries, validation_relevance, validation_counts = prepare_queries(
        "val", validation_states
    )
    test_queries, test_relevance, test_counts = prepare_queries("test", test_states)
    if not validation_queries or not test_queries:
        raise ValueError("Not enough supported validation/test ranking queries")

    def cache_features(queries: list[dict]) -> list[dict]:
        cached = []
        for query in queries:
            rows = recommender.recommend(
                user_id=query["user_id"], state=query["state"],
                query_vector=query["query_vector"],
                target_subject=config.SUBJECT_DEFAULT, target_grade=query["grade"],
                target_skill=query["skill"], top_k=len(query["pool"]),
                seen_item_ids=set(), strategy="hybrid",
            )
            if len(rows) < k:
                continue
            cached.append({**query, "rows": rows})
        return cached

    validation_features = cache_features(validation_queries)
    test_features = cache_features(test_queries)
    validation_relevance = {
        query["key"]: validation_relevance[query["key"]] for query in validation_features
    }
    test_relevance = {query["key"]: test_relevance[query["key"]] for query in test_features}

    def score_pair(queries: list[dict], relevance: dict[str, set[str]], alpha: float, beta: float) -> dict:
        rankings = {}
        for query in queries:
            ranked = sorted(
                query["rows"],
                key=lambda item: (
                    alpha * float(item["semantic_score"])
                    + beta * float(item["profile_score"])
                    + (1.0 - alpha - beta) * float(item["cf_score"]),
                    str(item["item_id"]),
                ),
                reverse=True,
            )
            rankings[query["key"]] = [str(item["item_id"]) for item in ranked[:k]]
        return evaluate_ranking_runs({"hybrid": rankings}, relevance, k=k)["hybrid"]

    def simplex_grid(step: float, center: tuple[float, float] | None = None, radius: float | None = None) -> list[tuple[float, float]]:
        if center is None:
            units = round(1.0 / step)
            return [
                (round(i * step, 6), round(j * step, 6))
                for i in range(units + 1) for j in range(units + 1 - i)
            ]
        alpha_center, beta_center = center
        values = set()
        lower_a, upper_a = max(0.0, alpha_center - radius), min(1.0, alpha_center + radius)
        lower_b, upper_b = max(0.0, beta_center - radius), min(1.0, beta_center + radius)
        i_start, i_end = int(np.floor(lower_a / step)), int(np.ceil(upper_a / step))
        j_start, j_end = int(np.floor(lower_b / step)), int(np.ceil(upper_b / step))
        for i in range(i_start, i_end + 1):
            for j in range(j_start, j_end + 1):
                alpha, beta = round(i * step, 6), round(j * step, 6)
                if abs(alpha - alpha_center) <= radius + 1e-9 and abs(beta - beta_center) <= radius + 1e-9 and alpha + beta <= 1.0 + 1e-9:
                    values.add((alpha, beta))
        values.add(center)
        return sorted(values)

    evaluated: dict[tuple[float, float], dict] = {}
    best_pair: tuple[float, float] | None = None
    sweep_stages = []
    stages = [
        (0.10, None), (0.02, 0.10), (0.005, 0.02), (0.001, 0.005),
        (0.0002, 0.001),
    ]
    for step, radius in stages:
        pairs = simplex_grid(step) if best_pair is None else simplex_grid(step, best_pair, radius)
        for alpha, beta in pairs:
            if (alpha, beta) not in evaluated:
                evaluated[(alpha, beta)] = score_pair(
                    validation_features, validation_relevance, alpha, beta
                )
        best_pair = max(
            evaluated,
            key=lambda pair: (
                evaluated[pair][f"ndcg@{k}"],
                evaluated[pair][f"recall@{k}"],
                evaluated[pair][f"precision@{k}"],
                -pair[0], -pair[1],
            ),
        )
        sweep_stages.append({
            "step": step, "evaluated_pairs": len(pairs),
            "best_alpha": best_pair[0], "best_beta": best_pair[1],
            **evaluated[best_pair],
        })

    assert best_pair is not None
    best_alpha, best_beta = best_pair
    test_best = score_pair(test_features, test_relevance, best_alpha, best_beta)
    test_default = score_pair(
        test_features, test_relevance,
        config.RANK_SEMANTIC_WEIGHT, config.RANK_PROFILE_WEIGHT,
    )

    def baseline_rankings(queries: list[dict]) -> dict[str, dict[str, list[str]]]:
        rankings: dict[str, dict[str, list[str]]] = {
            name: {} for name in ("Random", "MostPopular", "PureCB-ZPD", "PureSemantic", "PureCF")
        }
        for query in queries:
            rows = query["rows"]
            user_id, skill = query["user_id"], query["skill"]
            scored = {
                "Random": [(int.from_bytes(hashlib.sha256(
                    f"{seed}:{user_id}:{row['item_id']}".encode()
                ).digest()[:8], "big") / (2**64 - 1), row) for row in rows],
                "MostPopular": [(recommender.cf.popularity(
                    row, skill, float(row["difficulty"])
                ), row) for row in rows],
                "PureCB-ZPD": [(float(row["profile_score"]), row) for row in rows],
                "PureSemantic": [(float(row["semantic_score"]), row) for row in rows],
                "PureCF": [(float(row["cf_predicted_success"]), row) for row in rows],
            }
            for name, pairs in scored.items():
                pairs.sort(key=lambda pair: (pair[0], str(pair[1]["item_id"])), reverse=True)
                rankings[name][query["key"]] = [str(row["item_id"]) for _, row in pairs[:k]]
        return rankings

    test_rankings = baseline_rankings(test_features)

    def hybrid_rankings(alpha: float, beta: float) -> dict[str, list[str]]:
        output = {}
        for query in test_features:
            rows = sorted(
                query["rows"],
                key=lambda item: (
                    alpha * float(item["semantic_score"])
                    + beta * float(item["profile_score"])
                    + (1.0 - alpha - beta) * float(item["cf_score"]),
                    str(item["item_id"]),
                ),
                reverse=True,
            )
            output[query["key"]] = [str(item["item_id"]) for item in rows[:k]]
        return output

    test_rankings["TunedHybrid"] = hybrid_rankings(best_alpha, best_beta)
    test_rankings["DefaultHybrid"] = hybrid_rankings(
        config.RANK_SEMANTIC_WEIGHT, config.RANK_PROFILE_WEIGHT
    )
    test_metrics = evaluate_ranking_runs(test_rankings, test_relevance, k=k)
    metric_names = list(ranking_metrics([], set(), k=k))
    query_keys = sorted(test_relevance)
    per_query = {
        method: np.asarray([
            [ranking_metrics(test_rankings[method][key], test_relevance[key], k=k)[metric]
             for metric in metric_names]
            for key in query_keys
        ], dtype="float64")
        for method in test_rankings
    }
    rng = np.random.default_rng(seed + 104729)
    boot_metrics = {method: [] for method in test_rankings}
    boot_differences = {method: [] for method in test_rankings if method != "MostPopular"}
    for _ in range(1000):
        sampled = rng.integers(0, len(query_keys), size=len(query_keys))
        baseline_sample = per_query["MostPopular"][sampled].mean(axis=0)
        for method in test_rankings:
            sample_mean = per_query[method][sampled].mean(axis=0)
            boot_metrics[method].append(sample_mean)
            if method != "MostPopular":
                boot_differences[method].append(sample_mean - baseline_sample)

    def metric_intervals(samples: list[np.ndarray]) -> dict[str, list[float]]:
        bounds = np.percentile(np.asarray(samples), [2.5, 97.5], axis=0)
        return {
            metric: [float(bounds[0, i]), float(bounds[1, i])]
            for i, metric in enumerate(metric_names)
        }

    recommender_confidence_intervals = {
        "method": f"paired percentile bootstrap over learner-skill queries, 1000 resamples, seed={seed + 104729}",
        "metrics": {method: metric_intervals(samples) for method, samples in boot_metrics.items()},
        "paired_difference_vs_MostPopular": {
            method: metric_intervals(samples) for method, samples in boot_differences.items()
        },
    }
    return {
        "dataset": "ASSISTments Skill Builder + mapped external exercise bank",
        "protocol": "history-only recommender state; select alpha/beta on validation; one final evaluation on test",
        "relevance_definition": (
            "proxy: per learner and canonical skill, relevant candidates are those in the difficulty bucket "
            "whose held-out empirical correctness rate is closest to TARGET_SUCCESS; at least two observations "
            "in each of two buckets are required. ASSISTments has no shared exercise IDs with the external bank."
        ),
        "target_success": config.TARGET_SUCCESS,
        "k": k,
        "min_bucket_interactions": min_bucket_interactions,
        "difficulty_levels": list(config.DIFFICULTY_LEVELS),
        "ema_alpha_for_states": float(ema_alpha),
        "counts": {"validation": validation_counts, "test": test_counts,
                   "validation_scored_queries": len(validation_features),
                   "test_scored_queries": len(test_features)},
        "mapped_interaction_scope": mapping_scope,
        "selection_metric": f"ndcg@{k}, tie-break recall@{k}, precision@{k}",
        "sweep": {"stages": sweep_stages, "total_unique_alpha_beta_pairs": len(evaluated),
                  "best_alpha": best_alpha, "best_beta": best_beta,
                  "best_validation_metrics": evaluated[best_pair]},
        "test": {"best_validation_pair": test_best,
                 "default_alpha_beta": {"alpha": config.RANK_SEMANTIC_WEIGHT,
                                        "beta": config.RANK_PROFILE_WEIGHT,
                                        "cf_weight": 1.0 - config.RANK_SEMANTIC_WEIGHT - config.RANK_PROFILE_WEIGHT,
                                        **test_default},
                 "baselines": {name: test_metrics[name] for name in
                               ("Random", "MostPopular", "PureCB-ZPD", "PureSemantic", "PureCF")},
                 "confidence_intervals_95": recommender_confidence_intervals},
        "seed": seed,
    }


def evaluate_adaptive_rag(
    *, seed: int = 0, k: int = 5, n_bootstrap: int = 1000,
    min_bucket_interactions: int = 2,
) -> dict:
    """Compare semantic-only retrieval with profile/ZPD/CF-adaptive reranking.

    Relevance is a transparent ASSISTments-derived proxy: among supported held-out
    difficulty buckets, select the bucket whose observed success rate is closest to
    TARGET_SUCCESS, then treat mapped corpus chunks at that difficulty as relevant.
    History-only learner states are used for adaptive scores; validation/test rows
    are used only to define the proxy target, never to build the state.
    """
    from ..layer1_data.learner_state import build_states
    from ..layer2_embedding.index import load_and_verify
    from ..layer4_recommender import HybridRecommender, zpd_score
    from ..layer3_retrieval.retriever import CrossEncoderReranker

    if k < 1 or n_bootstrap < 1 or min_bucket_interactions < 1:
        raise ValueError("k, n_bootstrap, and min_bucket_interactions must be positive")

    mapper = SkillMapper()
    all_frame = data.load_logs(mapper)
    frame = data.filter_mapped_logs(all_frame)
    ema_alpha, _ = profile.tune_alpha(frame)
    priors, tau_base = profile.skill_priors(frame)
    states = build_states(frame, ema_alpha, priors, as_of=("history",), tau_base=tau_base)
    store, manifest = load_and_verify()
    encoder = load_query_encoder()

    bank_path = config.ART / "exercise_bank.jsonl"
    exercise_bank = [
        json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    catalog = json.loads(config.SKILL_CATALOG_JSON.read_text(encoding="utf-8"))
    recommender = HybridRecommender(
        store, exercise_bank, catalog,
        frame[["user_id", "tag", "correct", "split", "problem_id", "bucket"]],
        alpha=config.RANK_SEMANTIC_WEIGHT, beta=config.RANK_PROFILE_WEIGHT,
    )

    skill_tags = sorted(set(frame["tag"].dropna().astype(str)))
    query_vectors = np.asarray(encoder.encode_queries(skill_tags), dtype="float32")
    if query_vectors.ndim != 2 or query_vectors.shape[0] != len(skill_tags):
        raise ValueError("Query encoder did not return one vector per skill")
    query_by_skill = {tag: query_vectors[i] for i, tag in enumerate(skill_tags)}
    reranker = CrossEncoderReranker()
    candidate_cache: dict[str, tuple[list[int], np.ndarray, str, np.ndarray]] = {}
    for skill in skill_tags:
        skill_items = [
            item for item in exercise_bank
            if item.get("mapping_status") == "mapped" and item.get("skill_tag") == skill
            and item.get("subject") == config.SUBJECT_DEFAULT
        ]
        if not skill_items:
            continue
        grade_counts = Counter(str(item.get("grade")) for item in skill_items)
        grade = sorted(grade_counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
        candidate_ids = store.filter_ids(
            skill_tag=skill, target_subject=config.SUBJECT_DEFAULT,
            target_grade=grade, mapped_only=True,
        )
        if len(candidate_ids) < k:
            continue
        query_vector = np.asarray(query_by_skill[skill], dtype="float32")
        query_norm = float(np.linalg.norm(query_vector))
        if not np.isfinite(query_vector).all() or query_norm <= 1e-12:
            continue
        dense = store.search_ids(query_vector / query_norm, candidate_ids, 50)
        if len(dense) < k:
            continue
        chunk_ids = [int(index) for index, _ in dense]
        rerank_scores = np.asarray(reranker.score(
            skill, [store.meta[index]["text"] for index in chunk_ids]
        ), dtype="float64")
        if len(rerank_scores) != len(chunk_ids) or not np.isfinite(rerank_scores).all():
            raise ValueError(f"Cross-Encoder returned invalid scores for skill {skill!r}")
        low, high = float(rerank_scores.min()), float(rerank_scores.max())
        semantic = ((rerank_scores - low) / (high - low)
                    if high - low > 1e-12 else np.full(len(rerank_scores), 0.5))
        candidate_cache[skill] = (chunk_ids, semantic, grade, candidate_ids)
    skill_tags = sorted(candidate_cache)
    if not skill_tags:
        raise ValueError("No mapped skill has enough retrieval candidates")

    queries_by_split: dict[str, list[dict]] = {"val": [], "test": []}
    skipped_by_split: dict[str, Counter] = {"val": Counter(), "test": Counter()}
    profile_score_cache: dict[tuple[int, str], np.ndarray] = {}
    cf_score_cache: dict[tuple[int, str], np.ndarray] = {}
    for split in ("val", "test"):
        split_frame = frame[frame["split"].eq(split) & frame["tag"].isin(skill_tags)]
        for (user_id, skill), rows in split_frame.groupby(["user_id", "tag"], sort=True):
            cached = candidate_cache.get(str(skill))
            if cached is None:
                skipped_by_split[split]["too_few_candidate_chunks"] += 1
                continue
            chunks, semantic, _grade, all_candidate_ids = cached
            bucket_stats = rows.groupby("bucket")["correct"].agg(["size", "mean"])
            supported = bucket_stats[bucket_stats["size"] >= min_bucket_interactions]
            if len(supported) < 2:
                skipped_by_split[split]["fewer_than_two_supported_buckets"] += 1
                continue
            target_bucket = min(
                (int(bucket) for bucket in supported.index),
                key=lambda bucket: (
                    abs(float(supported.loc[bucket, "mean"]) - config.TARGET_SUCCESS), bucket
                ),
            )
            relevant = {
                str(store.meta[i]["chunk_id"])
                for i in all_candidate_ids
                if int(np.argmin(np.abs(
                    np.asarray(config.DIFFICULTY_LEVELS, dtype="float32")
                    - float(store.meta[i].get("difficulty", 0.6))
                ))) == target_bucket
            }
            if not relevant:
                skipped_by_split[split]["no_chunk_at_target_difficulty"] += 1
                continue
            state = states.get(int(user_id))
            if state is None:
                skipped_by_split[split]["no_history_state"] += 1
                continue
            cache_key = (int(user_id), str(skill))
            if cache_key not in profile_score_cache:
                profile_vector = recommender.profile_vector_for(state, str(skill))
                mastery = recommender.mastery_for_skill(state, str(skill))
                scores = []
                for index in chunks:
                    meta = store.meta[index]
                    raw_profile = (
                        0.5 if profile_vector is None else float(np.clip(
                            (np.dot(profile_vector, store.vectors[index]) + 1.0) / 2.0,
                            0.0, 1.0,
                        ))
                    )
                    scores.append(raw_profile * zpd_score(float(meta["difficulty"]) - mastery))
                profile_score_cache[cache_key] = np.asarray(scores, dtype="float64")
                items_by_doc = {
                    str(item["doc_id"]): item for item in exercise_bank
                    if item.get("mapping_status") == "mapped"
                    and item.get("skill_tag") == str(skill)
                    and item.get("subject") == config.SUBJECT_DEFAULT
                    and str(item.get("grade")) == str(_grade)
                }
                cf_scores = []
                for index in chunks:
                    item = items_by_doc.get(str(store.meta[index]["doc_id"]))
                    if item is None:
                        cf_scores.append(float(profile_score_cache[cache_key][len(cf_scores)]))
                        continue
                    expected_success, _granularity = recommender.cf.predict_candidate(
                        int(user_id), item, str(skill), float(item["difficulty"])
                    )
                    cf_scores.append(float(np.clip(
                        1.0 - abs(expected_success - config.TARGET_SUCCESS), 0.0, 1.0
                    )))
                cf_score_cache[cache_key] = np.asarray(cf_scores, dtype="float64")
            order_semantic = np.lexsort((np.asarray(chunks), -semantic))
            queries_by_split[split].append({
                "query_id": f"{int(user_id)}::{skill}",
                "relevant": relevant,
                "chunks": chunks,
                "semantic_scores": semantic,
                "profile_scores": profile_score_cache[cache_key],
                "cf_target_scores": cf_score_cache[cache_key],
                "skill_history_count": int(recommender.cf.user_skill_history_counts.get(
                    (int(user_id), str(skill)), 0
                )),
                "traditional": [
                    str(store.meta[chunks[i]]["chunk_id"]) for i in order_semantic[:k]
                ],
                "target_bucket": target_bucket,
            })

    validation_queries = queries_by_split["val"]
    test_queries = queries_by_split["test"]
    if not validation_queries or not test_queries:
        raise ValueError("Need eligible validation and test queries for adaptive-RAG tuning")

    def adaptive_ranking(query: dict, beta: float, profile_blend: float,
                         confidence_tau: float) -> list[str]:
        personalized = (
            profile_blend * query["profile_scores"]
            + (1.0 - profile_blend) * query["cf_target_scores"]
        )
        count = query["skill_history_count"]
        confidence = 1.0 if confidence_tau == 0 else count / (count + confidence_tau)
        effective_beta = beta * confidence
        scores = (1.0 - effective_beta) * query["semantic_scores"] + effective_beta * personalized
        order = np.lexsort((np.asarray(query["chunks"]), -scores))
        return [str(store.meta[query["chunks"][i]]["chunk_id"]) for i in order[:k]]

    beta_grid = [round(float(value), 2) for value in np.arange(0.05, 1.0, 0.05)]
    confidence_tau_grid = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
    validation_sweep = []
    for profile_blend in (0.25, 0.5, 0.75, 1.0):
        for beta in beta_grid:
            for confidence_tau in confidence_tau_grid:
                rows = [
                    ranking_metrics(
                        adaptive_ranking(query, beta, profile_blend, confidence_tau),
                        query["relevant"], k,
                    ) for query in validation_queries
                ]
                validation_sweep.append({
                    "alpha_semantic_max": round(1.0 - beta, 2), "beta_personalized_max": beta,
                    "profile_vector_blend": profile_blend,
                    "cf_target_success_blend": round(1.0 - profile_blend, 2),
                    "confidence_tau": confidence_tau,
                    "n_queries": len(rows),
                    **{metric: float(np.mean([row[metric] for row in rows])) for metric in rows[0]},
                })
    selected = max(validation_sweep, key=lambda row: (
        row[f"ndcg@{k}"], row[f"recall@{k}"], row[f"precision@{k}"],
        -row["beta_personalized_max"], row["profile_vector_blend"],
        row["confidence_tau"],
    ))
    tuned_alpha = float(selected["alpha_semantic_max"])
    tuned_beta = float(selected["beta_personalized_max"])
    tuned_profile_blend = float(selected["profile_vector_blend"])
    tuned_confidence_tau = float(selected["confidence_tau"])

    methods = {
        "TraditionalRAG": lambda query: query["traditional"],
        "AdaptiveRAGDefault": lambda query: adaptive_ranking(
            query, config.RAG_PROFILE_WEIGHT, config.RAG_PROFILE_VECTOR_BLEND,
            config.RAG_CONFIDENCE_TAU,
        ),
        "AdaptiveRAGTuned": lambda query: adaptive_ranking(
            query, tuned_beta, tuned_profile_blend, tuned_confidence_tau
        ),
    }
    per_query: dict[str, list[dict[str, float]]] = {name: [] for name in methods}
    test_rankings = {name: {} for name in methods}
    for query in test_queries:
        for method, rank in methods.items():
            ranked = rank(query)
            test_rankings[method][query["query_id"]] = ranked
            per_query[method].append(ranking_metrics(ranked, query["relevant"], k))
    metric_names = list(ranking_metrics([], set(), k))
    means = {
        method: {
            metric: float(np.mean([row[metric] for row in per_query[method]]))
            for metric in metric_names
        }
        for method in methods
    }
    rng = np.random.default_rng(seed + 7919)
    ci: dict[str, dict[str, list[float]]] = {}
    for method in ("AdaptiveRAGDefault", "AdaptiveRAGTuned"):
        ci[method] = {}
        for metric in metric_names:
            deltas = np.asarray([
                per_query[method][i][metric] - per_query["TraditionalRAG"][i][metric]
                for i in range(len(test_queries))
            ])
            boot = [float(deltas[rng.integers(0, len(test_queries), size=len(test_queries))].mean())
                    for _ in range(n_bootstrap)]
            ci[method][metric] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return {
        "dataset": "ASSISTments Skill Builder + original mapped Khan/MATH/GSM8K corpus",
        "protocol": "per-learner chronological split; select semantic/personalization/blend weights on validation only; all learner signals use history only; test metrics are diagnostic because test results were inspected during iterative development",
        "comparison": {
            "TraditionalRAG": "dense top-50 within mapped skill/subject/grade candidates, ranked by Cross-Encoder score",
            "AdaptiveRAGDefault": "Cross-Encoder scores combined with history-only profile/ZPD and CF expected-success alignment at configured weights",
            "AdaptiveRAGTuned": "semantic weight, personalized weight, and profile-vs-CF blend selected on validation",
            "configured_alpha_semantic": config.RAG_SEMANTIC_WEIGHT,
            "configured_beta_personalized": config.RAG_PROFILE_WEIGHT,
            "configured_profile_vector_blend": config.RAG_PROFILE_VECTOR_BLEND,
            "configured_confidence_tau": config.RAG_CONFIDENCE_TAU,
            "selected_alpha_semantic": tuned_alpha,
            "selected_beta_personalized": tuned_beta,
            "selected_profile_vector_blend": tuned_profile_blend,
            "selected_confidence_tau": tuned_confidence_tau,
            "candidate_k": 50,
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        },
        "relevance_definition": (
            "proxy: choose among held-out difficulty buckets with enough observations the bucket whose empirical "
            "correctness rate is closest to TARGET_SUCCESS; all candidate chunks at that difficulty are relevant. "
            "This evaluates difficulty alignment at skill level, not exact Khan item relevance."
        ),
        "target_success": config.TARGET_SUCCESS,
        "k": k,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "ema_alpha_for_history_profile": float(ema_alpha),
        "mapped_interactions": int(len(frame)),
        "index_chunks": int(manifest["n_chunks"]),
        "n_validation_queries": len(validation_queries),
        "n_test_queries": len(test_queries),
        "skipped": {split: dict(counts) for split, counts in skipped_by_split.items()},
        "alpha_beta_sweep": {
            "primary_selection_metric": f"ndcg@{k}",
            "constraint": "maximum personalized beta >= 0.05 and profile-vector share >= 0.25; effective beta is scaled by n/(n+tau), with tau=0 disabling confidence gating",
            "personalized_signal_grid": "profile-vector/ZPD blended with user-specific CF probability of success nearest TARGET_SUCCESS; profile-vector blend constrained to >= 0.25 for online mastery adaptation",
            "confidence_tau_grid": list(confidence_tau_grid),
            "validation": validation_sweep,
            "selected": selected,
        },
        "metrics": means,
        "paired_adaptive_minus_traditional_95ci": ci,
        "queries": [
            {"query_id": query["query_id"], "target_bucket": query["target_bucket"],
             "relevant_count": len(query["relevant"])}
            for query in test_queries
        ],
    }


def _native_id(value: object) -> str | None:
    """Normalize an ASSISTments identifier without merging its ID namespace."""
    if pd.isna(value):
        return None
    try:
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    value = str(value).strip()
    return value or None


def evaluate_assistments_next_item(
    *, k: int = 5, seed: int = 0, n_bootstrap: int = 1000,
    mapped_only: bool = True,
) -> dict:
    """Evaluate future-item retrieval against real held-out ASSISTments IDs.

    The history-only item-item CF model uses co-occurrence of native IDs. It is
    evaluated separately for problem, assistment, and template IDs because those
    identifiers represent different units. The external RAG bank has no join key
    to these IDs, so these results do not claim to evaluate ranking of external
    Khan/MATH/GSM8K exercises.
    """
    if k < 1 or n_bootstrap < 1:
        raise ValueError("k and n_bootstrap must be >= 1")
    mapper = SkillMapper()
    frame = data.load_logs(mapper)
    input_rows = len(frame)
    mapped_frame = data.filter_mapped_logs(frame)
    mapped_rows = int(len(mapped_frame))
    if mapped_only:
        frame = mapped_frame
    id_columns = ("problem_id", "assistment_id", "template_id")
    missing = set(id_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"ASSISTments loader did not preserve IDs: {sorted(missing)}")

    reports: dict[str, dict] = {}
    for target_column in id_columns:
        work = frame[["user_id", "order_id", "split", target_column]].copy()
        work["native_id"] = work[target_column].map(_native_id)
        work = work[work.native_id.notna()]
        history = work[work.split.eq("history")]
        # Collapse multi-skill rows and repeated events to user-ID observations.
        user_item = history[["user_id", "native_id"]].drop_duplicates()
        if user_item.empty:
            reports[target_column] = {"status": "no_history_ids"}
            continue
        users = sorted(user_item.user_id.astype(int).unique().tolist())
        items = sorted(user_item.native_id.unique().tolist())
        user_index = {user: index for index, user in enumerate(users)}
        item_index = {item: index for index, item in enumerate(items)}
        row = user_item.user_id.astype(int).map(user_index).to_numpy()
        col = user_item.native_id.map(item_index).to_numpy()
        user_item_matrix = sparse.csr_matrix(
            (np.ones(len(user_item), dtype="float32"), (row, col)),
            shape=(len(users), len(items)),
        )
        item_similarity = cosine_similarity(user_item_matrix.T, dense_output=False).tocsr()
        popularity = np.bincount(col, minlength=len(items)).astype("float64")
        seen_by_user = {
            int(user): set(group.native_id)
            for user, group in user_item.groupby("user_id", sort=False)
        }

        split_reports: dict[str, dict] = {}
        for split in ("val", "test"):
            heldout = work[work.split.eq(split)]
            relevant_by_user = heldout.groupby("user_id").native_id.agg(
                lambda values: set(values)
            )
            rankings: dict[str, dict[int, list[str]]] = {
                "Random": {}, "MostPopular": {}, "ItemCF": {}
            }
            relevant: dict[int, set[str]] = {}
            candidate_sizes: list[int] = []
            total_positives = catalog_positives = eligible_positives = 0
            for user_id_raw, positives in relevant_by_user.items():
                user_id = int(user_id_raw)
                total_positives += len(positives)
                if user_id not in user_index:
                    continue
                seen_ids = seen_by_user[user_id]
                catalog_relevant = set(positives) & set(items)
                catalog_positives += len(catalog_relevant)
                eligible_relevant = catalog_relevant - seen_ids
                eligible_positives += len(eligible_relevant)
                if not eligible_relevant:
                    continue
                seen_indices = np.fromiter(
                    (item_index[item] for item in seen_ids), dtype=np.int32
                )
                candidate_mask = np.ones(len(items), dtype=bool)
                candidate_mask[seen_indices] = False
                candidate_indices = np.flatnonzero(candidate_mask)
                if not len(candidate_indices):
                    continue
                candidates = np.asarray(items, dtype=object)[candidate_indices]
                if len(seen_indices):
                    cf_scores = np.asarray(
                        item_similarity[seen_indices][:, candidate_indices].sum(axis=0)
                    ).reshape(-1)
                    cf_scores /= max(1, len(seen_indices))
                else:
                    cf_scores = np.zeros(len(candidates), dtype="float64")
                popular_scores = popularity[candidate_indices]

                def top_ids(scores: np.ndarray) -> list[str]:
                    take = min(k, len(scores))
                    selected = np.argpartition(-scores, take - 1)[:take]
                    boundary = float(scores[selected].min())
                    above = np.flatnonzero(scores > boundary)
                    tied = np.flatnonzero(scores == boundary)
                    tied_order = np.argsort(candidates[tied], kind="stable")
                    selected = np.concatenate((above, tied[tied_order[:take - len(above)]]))
                    order = selected[np.lexsort((candidates[selected], -scores[selected]))]
                    return [str(candidates[index]) for index in order]

                query_seed = int.from_bytes(hashlib.sha256(
                    f"{seed}:{target_column}:{split}:{user_id}".encode()
                ).digest()[:8], "big")
                random_rng = np.random.default_rng(query_seed)
                random_count = min(k, len(candidates))
                random_indices = random_rng.choice(
                    len(candidates), size=random_count, replace=False
                )
                rankings["Random"][user_id] = [
                    str(candidates[index]) for index in random_indices
                ]
                rankings["MostPopular"][user_id] = top_ids(popular_scores)
                rankings["ItemCF"][user_id] = top_ids(cf_scores)
                relevant[user_id] = eligible_relevant
                candidate_sizes.append(len(candidates))

            if relevant:
                metrics = evaluate_ranking_runs(rankings, relevant, k=k)
                query_ids = sorted(relevant)
                metric_names = list(ranking_metrics([], set(), k=k))
                per_user = {
                    name: np.asarray([
                        [ranking_metrics(rankings[name][user], relevant[user], k=k)[metric]
                         for metric in metric_names]
                        for user in query_ids
                    ], dtype="float64")
                    for name in rankings
                }
                rng = np.random.default_rng(seed + len(reports) * 100 + (0 if split == "val" else 1))
                boot_means = {name: [] for name in rankings}
                boot_diffs = {name: [] for name in rankings if name != "MostPopular"}
                for _ in range(n_bootstrap):
                    sampled = rng.integers(0, len(query_ids), size=len(query_ids))
                    baseline_sample = per_user["MostPopular"][sampled].mean(axis=0)
                    for name in rankings:
                        sample_mean = per_user[name][sampled].mean(axis=0)
                        boot_means[name].append(sample_mean)
                        if name != "MostPopular":
                            boot_diffs[name].append(sample_mean - baseline_sample)

                def intervals(samples: list[np.ndarray]) -> dict[str, list[float]]:
                    values = np.asarray(samples)
                    bounds = np.percentile(values, [2.5, 97.5], axis=0)
                    return {
                        metric: [float(bounds[0, index]), float(bounds[1, index])]
                        for index, metric in enumerate(metric_names)
                    }

                confidence_intervals = {
                    "metrics": {name: intervals(samples) for name, samples in boot_means.items()},
                    "paired_difference_vs_MostPopular": {
                        name: intervals(samples) for name, samples in boot_diffs.items()
                    },
                    "method": f"paired percentile bootstrap over learners, {n_bootstrap} resamples, seed={seed}",
                }
            else:
                metrics = {}
                confidence_intervals = {}
            split_reports[split] = {
                "n_users": len(relevant),
                "heldout_unique_positive_ids": total_positives,
                "positive_ids_in_history_catalog": catalog_positives,
                "unseen_positive_ids_in_candidate_catalog": eligible_positives,
                "catalog_coverage": (
                    catalog_positives / total_positives if total_positives else None
                ),
                "candidate_relevance_coverage": (
                    eligible_positives / total_positives if total_positives else None
                ),
                "mean_candidate_count": (
                    float(np.mean(candidate_sizes)) if candidate_sizes else 0.0
                ),
                "metrics": metrics,
                "confidence_intervals_95": confidence_intervals,
            }
        reports[target_column] = {
            "unit": {
                "problem_id": "individual ASSISTments question",
                "assistment_id": "ASSISTments activity/scaffold group",
                "template_id": "similar-question template group",
            }[target_column],
            "history_users": len(users),
            "history_catalog_size": len(items),
            "candidate_rule": "native IDs observed in history, excluding IDs already seen by the learner",
            "model": "ItemCF: mean cosine co-occurrence with learner history; fitted on history only",
            "splits": split_reports,
        }

    return {
        "dataset": "ASSISTments Skill Builder",
        "protocol": (
            "per-learner chronological 70/15/15 split; history-only candidate catalog and CF; "
            "validation/test are scored separately; external exercise-bank IDs are not joined"
        ),
        "target_definition": (
            "unique held-out native IDs that are present in the history catalog and not already seen by that learner; "
            "this is temporal unseen-item retrieval, not a single immediate-next-item prediction"
        ),
        "metrics_at_k": k,
        "mapped_only": mapped_only,
        "mapping_coverage": {
            "input_interactions": input_rows,
            "mapped_interactions": mapped_rows,
            "evaluated_interactions": len(frame),
            "mapped_share": mapped_rows / input_rows if input_rows else None,
        },
        "seed": seed,
        "id_semantics": {
            "problem_id": "question-level key",
            "assistment_id": "activity/scaffold key",
            "template_id": "similar-question template key",
        },
        "results": reports,
    }
def _cluster_bootstrap_intervals(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    users: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, list[float]]]:
    """User-clustered percentile CIs for RMSE/MAE and paired differences vs EMA."""
    unique_users, inverse = np.unique(users, return_inverse=True)
    n_users = len(unique_users)
    row_count = np.bincount(inverse, minlength=n_users).astype("float64")
    grouped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, prediction in predictions.items():
        residual = y - np.clip(prediction, 0.0, 1.0)
        grouped[name] = (
            np.bincount(inverse, weights=residual**2, minlength=n_users),
            np.bincount(inverse, weights=np.abs(residual), minlength=n_users),
        )

    rng = np.random.default_rng(seed)
    values: dict[str, dict[str, list[float]]] = {
        name: {"AUC": [], "RMSE": [], "MAE": []} for name in predictions
    }
    diffs: dict[str, dict[str, list[float]]] = {
        name: {"AUC_vs_EMA": [], "RMSE_vs_EMA": [], "MAE_vs_EMA": []}
        for name in predictions if name != "EMA mastery"
    }
    for _ in range(n_bootstrap):
        multiplicity = np.bincount(
            rng.integers(0, n_users, size=n_users), minlength=n_users
        ).astype("float64")
        denom = float(np.dot(multiplicity, row_count))
        row_weights = multiplicity[inverse]
        metrics: dict[str, tuple[float, float]] = {}
        aucs: dict[str, float] = {}
        for name, (sq_error, abs_error) in grouped.items():
            mse = float(np.dot(multiplicity, sq_error) / denom)
            mae = float(np.dot(multiplicity, abs_error) / denom)
            rmse = float(np.sqrt(mse))
            metrics[name] = (rmse, mae)
            values[name]["RMSE"].append(rmse)
            values[name]["MAE"].append(mae)
            try:
                auc = float(roc_auc_score(
                    y, np.clip(predictions[name], 0.0, 1.0), sample_weight=row_weights
                ))
            except ValueError:
                continue
            aucs[name] = auc
            values[name]["AUC"].append(auc)
        ema_rmse, ema_mae = metrics["EMA mastery"]
        for name in diffs:
            if name in aucs and "EMA mastery" in aucs:
                diffs[name]["AUC_vs_EMA"].append(aucs[name] - aucs["EMA mastery"])
            diffs[name]["RMSE_vs_EMA"].append(metrics[name][0] - ema_rmse)
            diffs[name]["MAE_vs_EMA"].append(metrics[name][1] - ema_mae)

    def ci(samples: list[float]) -> list[float]:
        low, high = np.percentile(samples, [2.5, 97.5])
        return [float(low), float(high)]

    return {
        "metrics": {name: {metric: ci(samples) for metric, samples in metrics.items()}
                    for name, metrics in values.items()},
        "paired_difference_vs_ema": {
            name: {metric: ci(samples) for metric, samples in metrics.items()}
            for name, metrics in diffs.items()
        },
    }


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise correction, preserving the caller's order."""
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype="float64"))
    adjusted = np.empty(len(p_values), dtype="float64")
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted.tolist()


def _learner_group_analysis(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    test_mask: np.ndarray,
) -> dict:
    """Measure test errors by learner strata defined only from mapped history."""
    history = frame.loc[frame["split"].eq("history")]
    history_accuracy = history.groupby("user_id")["correct"].mean()
    history_volume = history.groupby("user_id").size()
    axes = {
        "prior_accuracy": history_accuracy.to_dict(),
        "history_interaction_volume": history_volume.to_dict(),
    }
    test_frame = frame.loc[test_mask]
    test_users = test_frame["user_id"].astype(int).to_numpy()
    y = test_frame["correct"].to_numpy(dtype="float64")
    results: dict[str, dict] = {}
    welch_rows: list[dict] = []
    group_losses: dict[str, dict[str, dict[int, dict[str, float]]]] = {}

    for axis, raw_values in axes.items():
        ordered = sorted(
            ((int(user), float(value)) for user, value in raw_values.items()),
            key=lambda pair: (pair[1], pair[0]),
        )
        labels = ("low", "middle", "high")
        partitions = np.array_split(np.asarray([user for user, _ in ordered], dtype="int64"), 3)
        user_group = {
            int(user): label
            for label, group in zip(labels, partitions)
            for user in group
        }
        axis_report: dict[str, dict] = {}
        group_losses[axis] = {label: {} for label in labels}
        for label in labels:
            group_users = {user for user, group in user_group.items() if group == label}
            row_mask = np.isin(test_users, list(group_users))
            group_y = y[row_mask]
            output_methods = {}
            for method, full_prediction in predictions.items():
                test_prediction = np.asarray(full_prediction, dtype="float64")[test_mask]
                selected_prediction = test_prediction[row_mask]
                output_methods[method] = _metrics(group_y, selected_prediction)
                selected_users = test_users[row_mask]
                for user in np.unique(selected_users):
                    user_rows = selected_users == user
                    errors = group_y[user_rows] - selected_prediction[user_rows]
                    group_losses[axis][label].setdefault(int(user), {})[method] = {
                        "MAE": float(np.mean(np.abs(errors))),
                        "RMSE": float(np.sqrt(np.mean(errors**2))),
                    }
            axis_report[label] = {
                "n_learners": len(group_users),
                "test_learners": int(np.unique(test_users[row_mask]).size),
                "test_interactions": int(row_mask.sum()),
                "metrics": output_methods,
                "history_feature_range": [
                    min(raw_values[user] for user in group_users),
                    max(raw_values[user] for user in group_users),
                ] if group_users else None,
            }
        results[axis] = axis_report

        for left, right in (("low", "middle"), ("middle", "high"), ("low", "high")):
            left_users = group_losses[axis][left]
            right_users = group_losses[axis][right]
            for metric in ("MAE", "RMSE"):
                left_values = np.asarray([
                    values["EMA mastery"][metric] for values in left_users.values()
                ], dtype="float64")
                right_values = np.asarray([
                    values["EMA mastery"][metric] for values in right_users.values()
                ], dtype="float64")
                if len(left_values) < 2 or len(right_values) < 2:
                    continue
                test_result = ttest_ind(left_values, right_values, equal_var=False)
                raw_p_value = float(test_result.pvalue)
                welch_rows.append({
                    "axis": axis,
                    "model": "EMA mastery",
                    "metric": metric,
                    "groups": [left, right],
                    "mean_difference_left_minus_right": float(left_values.mean() - right_values.mean()),
                    "t_statistic": float(test_result.statistic),
                    "degrees_of_freedom": float(test_result.df),
                    "p_value": max(raw_p_value, np.finfo("float64").tiny),
                    "p_value_underflow": raw_p_value == 0.0,
                    "n_left": int(len(left_values)),
                    "n_right": int(len(right_values)),
                })

    adjusted = _holm_adjust([row["p_value"] for row in welch_rows])
    for row, p_adjusted in zip(welch_rows, adjusted):
        row["p_value_holm"] = p_adjusted
    return {
        "grouping_protocol": (
            "Within each axis, learners are ranked by history-only mapped data and split into three nearly equal groups; "
            "test outcomes are not used to define groups. Group metrics are evaluated on mapped test interactions."
        ),
        "groups": results,
        "welch_tests_on_independent_learner_groups": {
            "model": "EMA mastery",
            "metrics": ["MAE", "RMSE"],
            "multiple_testing_correction": "Holm across all reported Welch tests",
            "tests": welch_rows,
        },
    }


def _paired_tests_vs_ema(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    test_mask: np.ndarray,
) -> dict:
    """Paired per-learner t-tests for methods evaluated on the same test users."""
    users = frame.loc[test_mask, "user_id"].astype(int).to_numpy()
    y = frame.loc[test_mask, "correct"].to_numpy(dtype="float64")
    test_predictions = {
        method: np.clip(np.asarray(values, dtype="float64")[test_mask], 0.0, 1.0)
        for method, values in predictions.items()
    }
    user_ids = np.unique(users)
    per_user: dict[str, dict[int, dict[str, float]]] = {}
    for method, values in test_predictions.items():
        per_user[method] = {}
        for user in user_ids:
            mask = users == user
            residual = y[mask] - values[mask]
            per_user[method][int(user)] = {
                "MAE": float(np.mean(np.abs(residual))),
                "RMSE": float(np.sqrt(np.mean(residual**2))),
            }
    rows = []
    for method in predictions:
        if method == "EMA mastery":
            continue
        for metric in ("MAE", "RMSE"):
            differences = np.asarray([
                per_user[method][int(user)][metric]
                - per_user["EMA mastery"][int(user)][metric]
                for user in user_ids
            ], dtype="float64")
            result = ttest_rel(differences, np.zeros_like(differences))
            raw_p_value = float(result.pvalue)
            rows.append({
                "comparison": f"{method} minus EMA mastery",
                "metric": metric,
                "mean_per_learner_difference": float(differences.mean()),
                "t_statistic": float(result.statistic),
                "degrees_of_freedom": float(len(differences) - 1),
                "p_value": max(raw_p_value, np.finfo("float64").tiny),
                "p_value_underflow": raw_p_value == 0.0,
                "n_learners": int(len(differences)),
            })
    adjusted = _holm_adjust([row["p_value"] for row in rows])
    for row, p_adjusted in zip(rows, adjusted):
        row["p_value_holm"] = p_adjusted
    return {
        "method": "paired two-sided t-test on per-learner test errors; Holm correction across comparisons",
        "tests": rows,
    }


def evaluate_assistments(*, n_bootstrap: int = 1000, seed: int = 0) -> dict:
    mapper = SkillMapper()
    all_frame = data.load_logs(mapper)
    frame = data.filter_mapped_logs(all_frame)
    alpha, validation_log_loss = profile.tune_alpha(frame)
    state_trace = profile.run_states(frame, alpha, with_profile=False)
    test_mask = frame["split"].eq("test").to_numpy()
    test = frame.loc[test_mask]
    y = test["correct"].to_numpy(dtype="float64")

    history = frame["split"].eq("history")
    history_mean = float(frame.loc[history, "correct"].mean())
    history_user_mean = frame.loc[history].groupby("user_id")["correct"].mean()
    history_skill_mean = frame.loc[history].groupby("skill_id")["correct"].mean()
    previous = frame.groupby(["user_id", "skill_id"])["correct"].shift(1)
    prior = frame.loc[history].groupby("skill_id")["correct"].mean()
    last_answer = previous.fillna(frame["skill_id"].map(prior)).fillna(history_mean)

    predictions: dict[str, np.ndarray] = {
        "Global mean": np.full(len(frame), history_mean, dtype="float64"),
        "Skill mean (history)": frame["skill_id"].map(history_skill_mean).fillna(history_mean).to_numpy(),
        "User mean (history)": frame["user_id"].map(history_user_mean).fillna(history_mean).to_numpy(),
        "Last answer on skill": last_answer.to_numpy(dtype="float64"),
        "EMA mastery": state_trace["M_before"].to_numpy(dtype="float64"),
    }

    interactions = frame[["user_id", "tag", "correct", "split"]]
    cf = SkillCollaborativeFilter().fit(interactions)
    predictions["Skill CF"] = np.asarray([
        cf.predict(int(user_id), str(tag))
        if isinstance(tag, str) and tag != "unmapped" else cf.global_mean
        for user_id, tag in zip(frame["user_id"], frame["tag"])
    ], dtype="float64")

    test_predictions = {name: values[test_mask] for name, values in predictions.items()}
    return {
        "dataset": "ASSISTments Skill Builder",
        "protocol": "per-learner chronological 70/15/15; prequential test scoring (prior test responses update state for later attempts)",
        "counts": {
            "users": int(frame["user_id"].nunique()),
            "rows": int(len(frame)),
            "history": int((frame["split"] == "history").sum()),
            "validation": int((frame["split"] == "val").sum()),
            "test": int(test_mask.sum()),
        },
        "mapped_interaction_scope": {
            "all_interactions": int(len(all_frame)),
            "mapped_interactions_used": int(len(frame)),
            "excluded_unmapped_interactions": int(len(all_frame) - len(frame)),
            "mapped_share": float(len(frame) / len(all_frame)) if len(all_frame) else None,
        },
        "ema": {
            "selected_alpha": float(alpha),
            "selection_metric": "validation log-loss",
            "validation_log_loss_by_alpha": {str(key): float(value) for key, value in validation_log_loss.items()},
        },
        "test_metrics": {
            name: _metrics(y, values[test_mask]) for name, values in predictions.items()
        },
        "confidence_intervals_95": _cluster_bootstrap_intervals(
            y, test_predictions, test["user_id"].to_numpy(),
            n_bootstrap=n_bootstrap, seed=seed,
        ),
        "paired_tests_vs_ema": _paired_tests_vs_ema(frame, predictions, test_mask),
        "learner_group_analysis": _learner_group_analysis(frame, predictions, test_mask),
        "ci_method": f"paired percentile bootstrap by learner, {n_bootstrap} resamples, seed={seed}",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=config.ART / "assistments_evaluation.json")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recommender-sweep", action="store_true",
                        help="Tune hybrid-ranking alpha/beta on validation and evaluate on test")
    parser.add_argument("--next-item", action="store_true",
                        help="Evaluate native ASSISTments problem/assistment/template ID ranking")
    parser.add_argument("--adaptive-rag", action="store_true",
                        help="Compare semantic retrieval with history-profile/ZPD/CF adaptive ranking")
    parser.add_argument("--ranking-output", type=Path,
                        default=config.ART / "assistments_recommender_sweep.json")
    parser.add_argument("--next-item-output", type=Path,
                        default=config.ART / "assistments_next_item_evaluation.json")
    parser.add_argument("--adaptive-rag-output", type=Path,
                        default=config.ART / "assistments_adaptive_rag_evaluation.json")
    parser.add_argument("--include-unmapped", action="store_true",
                        help="Include interactions without a canonical skill mapping (comparison only)")
    args = parser.parse_args(argv)
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if args.adaptive_rag:
        report = evaluate_adaptive_rag(seed=args.seed, n_bootstrap=args.bootstrap)
        args.adaptive_rag_output.parent.mkdir(parents=True, exist_ok=True)
        args.adaptive_rag_output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.next_item:
        mapped_only = not args.include_unmapped
        if mapped_only and args.next_item_output == config.ART / "assistments_next_item_evaluation.json":
            args.next_item_output = config.ART / "assistments_next_item_mapped_evaluation.json"
        report = evaluate_assistments_next_item(
            seed=args.seed, n_bootstrap=args.bootstrap, mapped_only=mapped_only
        )
        args.next_item_output.parent.mkdir(parents=True, exist_ok=True)
        args.next_item_output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.recommender_sweep:
        report = evaluate_recommender_weight_sweep(seed=args.seed)
        args.ranking_output.parent.mkdir(parents=True, exist_ok=True)
        args.ranking_output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    report = evaluate_assistments(n_bootstrap=args.bootstrap, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
