"""ZPD-aware hybrid exercise ranking using skill-linked interaction history."""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .. import config
from ..layer1_data.learner_state import LearnerState
from ..layer1_data.sources import filter_exercise_bank
from ..layer2_embedding.profile_vector import InteractionVectorizer
from ..vectorstore import VectorStore


def zpd_score(delta: float) -> float:
    """Piecewise ZPD score from the project specification."""
    if not np.isfinite(delta):
        raise ValueError("delta must be finite")
    if 0.05 <= delta <= 0.30:
        return 1.0
    if -0.10 <= delta < 0.05:
        return 0.7
    if 0.30 < delta <= 0.50:
        return 0.5
    if delta < -0.10:
        return 0.3
    return 0.2


def canonical_skill_mappings(skill_catalog: dict) -> dict[int, str]:
    """Return source skill IDs mapped to canonical skill tags."""
    return {
        int(row["source_skill_id"]): row["canonical_skill_tag"]
        for row in skill_catalog["source_skill_mappings"]
        if row["canonical_skill_tag"] is not None
    }


def mastery_by_canonical_skill(state: LearnerState, source_skill_to_tag: dict[int, str]) -> dict[str, float]:
    """Merge source-skill EMA values under their canonical topic, weighted by attempt count."""
    weighted: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for source_id, tag in source_skill_to_tag.items():
        key = str(source_id)
        if key in state.M:
            weighted[tag].append((float(state.M[key]), max(1, int(state.attempts.get(key, 0)))))
    mastery = {
        tag: sum(value * count for value, count in values) / sum(count for _, count in values)
        for tag, values in weighted.items()
    }
    mastery.update(state.canonical_M)
    return mastery


def _normalized_score(value: float) -> float:
    return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))


class SkillCollaborativeFilter:
    """User-neighbour CF over canonical skills, learned only from interaction history.

    Since ASSISTments items and the external exercise bank do not share item IDs, this
    estimates preference at skill level. It is a skill-CF signal, not item-level CF.
    """

    def __init__(self, n_neighbors: int = config.CF_NEIGHBOURS):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        self.n_neighbors = n_neighbors
        self.user_ids: np.ndarray | None = None
        self.skill_tags: list[str] = []
        self.user_index: dict[int, int] = {}
        self.skill_index: dict[str, int] = {}
        self.ratings: np.ndarray | None = None
        self.observed: np.ndarray | None = None
        self.similarities: np.ndarray | None = None
        self.global_mean = 0.5
        self.skill_means: np.ndarray | None = None

    def fit(self, interactions: pd.DataFrame) -> "SkillCollaborativeFilter":
        required = {"user_id", "tag", "correct", "split"}
        missing = required - set(interactions.columns)
        if missing:
            raise ValueError(f"Interaction frame misses columns: {sorted(missing)}")
        history = interactions[
            interactions["split"].eq("history")
            & interactions["tag"].notna()
            & interactions["tag"].ne("unmapped")
        ]
        if history.empty:
            raise ValueError("No mapped history interactions available to fit skill-CF")

        grouped = history.groupby(["user_id", "tag"])["correct"].mean()
        self.user_ids = np.sort(grouped.index.get_level_values("user_id").unique().astype(int))
        self.skill_tags = sorted(grouped.index.get_level_values("tag").unique().astype(str))
        self.user_index = {int(user_id): index for index, user_id in enumerate(self.user_ids)}
        self.skill_index = {tag: index for index, tag in enumerate(self.skill_tags)}
        self.ratings = np.zeros((len(self.user_ids), len(self.skill_tags)), dtype="float32")
        self.observed = np.zeros_like(self.ratings, dtype=bool)
        for (user_id, tag), score in grouped.items():
            ui, si = self.user_index[int(user_id)], self.skill_index[str(tag)]
            self.ratings[ui, si] = float(score)
            self.observed[ui, si] = True

        self.global_mean = float(grouped.mean())
        counts = self.observed.sum(axis=0)
        sums = (self.ratings * self.observed).sum(axis=0)
        self.skill_means = np.divide(
            sums, counts, out=np.full(len(self.skill_tags), self.global_mean), where=counts > 0
        )
        self.similarities = cosine_similarity(self.ratings)
        np.fill_diagonal(self.similarities, 0.0)
        return self

    def predict(self, user_id: int, skill_tag: str) -> float:
        if self.ratings is None or self.observed is None or self.similarities is None or self.skill_means is None:
            raise RuntimeError("Call fit() before predict()")
        skill_index = self.skill_index.get(skill_tag)
        if skill_index is None:
            return self.global_mean
        user_index = self.user_index.get(int(user_id))
        if user_index is None:
            return float(self.skill_means[skill_index])

        eligible = self.observed[:, skill_index].copy()
        eligible[user_index] = False
        neighbour_ids = np.flatnonzero(eligible)
        if not len(neighbour_ids):
            return float(self.skill_means[skill_index])
        order = np.argsort(self.similarities[user_index, neighbour_ids])[::-1]
        neighbour_ids = neighbour_ids[order[:self.n_neighbors]]
        weights = self.similarities[user_index, neighbour_ids]
        if not np.any(weights > 0):
            return float(self.skill_means[skill_index])
        values = self.ratings[neighbour_ids, skill_index]
        return float(np.clip(np.average(values, weights=weights), 0.0, 1.0))


class CandidateCollaborativeFilter:
    """Predict candidate success by shared item ID, then by canonical skill/difficulty.

    External exercise-bank IDs usually do not occur in ASSISTments. The feature-level
    fallback transfers collaborative signal through the mapped skill and difficulty
    bucket, so different candidate difficulties can receive different scores.
    """

    def __init__(self, interactions: pd.DataFrame, n_neighbors: int = config.CF_NEIGHBOURS):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        self.n_neighbors = n_neighbors
        history = interactions[
            interactions["split"].eq("history")
            & interactions["tag"].notna()
            & interactions["tag"].ne("unmapped")
        ].copy()
        if history.empty:
            raise ValueError("No mapped history interactions available to fit candidate CF")
        self.user_skill_history_counts = (
            history.groupby(["user_id", "tag"]).size().to_dict()
        )
        if "problem_id" in history:
            exact = history.groupby(["user_id", "problem_id"], dropna=True)["correct"].mean()
            self.item_model = self._fit(exact)
            self.item_id_lookup = {
                self._item_key(feature): feature
                for feature in self.item_model["feature_index"]
            }
        else:
            self.item_model = None
            self.item_id_lookup = {}
        if "bucket" not in history:
            raise ValueError("Interactions must include difficulty buckets for candidate CF")
        history["candidate_feature"] = list(zip(history["tag"].astype(str), history["bucket"].astype(int)))
        grouped = history.groupby(["user_id", "candidate_feature"])["correct"].mean()
        self.feature_model = self._fit(grouped)

    def personalization_confidence(self, user_id: int, skill_tag: str,
                                   tau: float = config.RAG_CONFIDENCE_TAU) -> float:
        """Return skill-history confidence; tau=0 disables gating."""
        if not np.isfinite(tau) or tau < 0:
            raise ValueError("tau must be finite and non-negative")
        count = int(self.user_skill_history_counts.get((int(user_id), skill_tag), 0))
        if tau == 0:
            return 1.0
        return float(count / (count + tau))

    @staticmethod
    def _item_key(value: object) -> str:
        try:
            number = float(value)
            if np.isfinite(number) and number.is_integer():
                return str(int(number))
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    def _fit(self, grouped: pd.Series) -> dict:
        users = np.sort(grouped.index.get_level_values(0).unique().astype(int))
        features = list(grouped.index.get_level_values(1).unique())
        user_index = {int(value): i for i, value in enumerate(users)}
        feature_index = {value: i for i, value in enumerate(features)}
        ratings = np.zeros((len(users), len(features)), dtype="float32")
        observed = np.zeros_like(ratings, dtype=bool)
        for (user, feature), score in grouped.items():
            ui, fi = user_index[int(user)], feature_index[feature]
            ratings[ui, fi] = float(score)
            observed[ui, fi] = True
        centered = np.where(observed, ratings, 0.0)
        means = np.divide(
            centered.sum(axis=1), observed.sum(axis=1),
            out=np.zeros(len(users), dtype="float32"), where=observed.sum(axis=1) > 0,
        )
        centered = np.where(observed, ratings - means[:, None], 0.0)
        similarities = cosine_similarity(centered)
        np.fill_diagonal(similarities, 0.0)
        feature_means = np.divide(
            (ratings * observed).sum(axis=0), observed.sum(axis=0),
            out=np.full(len(features), float(grouped.mean())), where=observed.sum(axis=0) > 0,
        )
        return {
            "user_index": user_index, "feature_index": feature_index,
            "ratings": ratings, "observed": observed,
            "similarities": similarities, "means": means,
            "feature_means": feature_means, "global_mean": float(grouped.mean()),
        }

    def _predict(self, model: dict | None, user_id: int, feature: object) -> float | None:
        if model is None or feature not in model["feature_index"]:
            return None
        fi = model["feature_index"][feature]
        ui = model["user_index"].get(int(user_id))
        if ui is None:
            return float(model["feature_means"][fi])
        eligible = model["observed"][:, fi].copy()
        eligible[ui] = False
        neighbours = np.flatnonzero(eligible)
        if not len(neighbours):
            return float(model["feature_means"][fi])
        order = np.argsort(model["similarities"][ui, neighbours])[::-1]
        neighbours = neighbours[order[:self.n_neighbors]]
        weights = np.maximum(model["similarities"][ui, neighbours], 0.0)
        if not np.any(weights > 1e-12):
            return float(model["feature_means"][fi])
        values = model["ratings"][neighbours, fi]
        return float(np.clip(np.average(values, weights=weights), 0.0, 1.0))

    def predict_candidate(
        self, user_id: int, item: dict, skill_tag: str, difficulty: float,
    ) -> tuple[float, str]:
        item_id = str(item.get("item_id", ""))
        if self.item_model is not None:
            exact_id = self.item_id_lookup.get(self._item_key(item_id))
            if exact_id is not None:
                score = self._predict(self.item_model, user_id, exact_id)
                if score is not None:
                    return score, "shared_item_id"
        levels = np.asarray(config.DIFFICULTY_LEVELS, dtype="float32")
        bucket = int(np.argmin(np.abs(levels - float(difficulty))))
        feature_score = self._predict(self.feature_model, user_id, (skill_tag, bucket))
        if feature_score is not None:
            return feature_score, "canonical_skill_difficulty"
        return float(self.feature_model["global_mean"]), "global_history_fallback"

    def popularity(self, item: dict, skill_tag: str, difficulty: float) -> float:
        """History-only mean success for an item, or its skill/difficulty feature."""
        item_id = self._item_key(item.get("item_id", ""))
        exact_id = self.item_id_lookup.get(item_id)
        if exact_id is not None:
            model = self.item_model
            feature = exact_id
        else:
            levels = np.asarray(config.DIFFICULTY_LEVELS, dtype="float32")
            bucket = int(np.argmin(np.abs(levels - float(difficulty))))
            model = self.feature_model
            feature = (skill_tag, bucket)
        if model is not None and feature in model["feature_index"]:
            return float(model["feature_means"][model["feature_index"][feature]])
        return float(self.feature_model["global_mean"])


class HybridRecommender:
    """Rank tasks using the three-way semantic, profile-ZPD, and CF hybrid score."""

    def __init__(self, store: VectorStore, exercise_bank: list[dict], skill_catalog: dict,
                 interactions: pd.DataFrame,
                 alpha: float = config.RANK_SEMANTIC_WEIGHT,
                 beta: float = config.RANK_PROFILE_WEIGHT,
                 n_cf_neighbors: int = config.CF_NEIGHBOURS):
        if (not np.isfinite(alpha) or not np.isfinite(beta) or alpha < 0 or beta < 0
                or alpha + beta > 1):
            raise ValueError("alpha and beta must be non-negative and sum to at most 1")
        self.store = store
        self.exercise_bank = exercise_bank
        self.source_skill_to_tag = canonical_skill_mappings(skill_catalog)
        self.alpha = alpha
        self.beta = beta
        self.cf = CandidateCollaborativeFilter(interactions, n_cf_neighbors)
        self.interaction_vectorizer = InteractionVectorizer(store.vectors, store.meta)
        self.chunk_ids_by_doc: dict[str, list[int]] = defaultdict(list)
        for index, metadata in enumerate(store.meta):
            self.chunk_ids_by_doc[metadata["doc_id"]].append(index)

    def _profile_vector(self, state: LearnerState, target_skill: str) -> np.ndarray | None:
        total = np.zeros(self.store.vectors.shape[1], dtype="float32")
        found = False
        canonical_weights = state.canonical_prof.get(target_skill)
        if canonical_weights:
            for bucket, weight in enumerate(canonical_weights):
                if weight <= 0:
                    continue
                vector = self.interaction_vectorizer(target_skill, bucket)
                if vector is not None:
                    total += float(weight) * vector
                    found = True
            if found:
                norm = np.linalg.norm(total)
                return total / norm if norm > 1e-12 else None
        for source_id, tag in self.source_skill_to_tag.items():
            if tag != target_skill:
                continue
            weights = state.prof.get(str(source_id))
            if not weights:
                continue
            for bucket, weight in enumerate(weights):
                if weight <= 0:
                    continue
                vector = self.interaction_vectorizer(target_skill, bucket)
                if vector is not None:
                    total += float(weight) * vector
                    found = True
        norm = np.linalg.norm(total)
        return total / norm if found and norm > 1e-12 else None

    def profile_vector_for(self, state: LearnerState, target_skill: str) -> np.ndarray | None:
        """Expose the learner vector to Adaptive RAG using the same profile construction."""
        return self._profile_vector(state, target_skill)

    def mastery_for_skill(self, state: LearnerState, target_skill: str) -> float:
        mastery = mastery_by_canonical_skill(state, self.source_skill_to_tag)
        return float(mastery.get(target_skill, state.A if state.n else 0.5))

    def recommend(self, *, user_id: int, state: LearnerState, query_vector: np.ndarray,
                  target_subject: str, target_grade: str, target_skill: str,
                  top_k: int = 10, seen_item_ids: set[str] | None = None,
                  strategy: str = "hybrid") -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        strategies = {"hybrid", "random", "most_popular", "traditional_rag", "pure_cf", "pure_cb_zpd"}
        if strategy not in strategies:
            raise ValueError(f"strategy must be one of {sorted(strategies)}")
        if not target_subject or not target_grade or not target_skill:
            raise ValueError("target_subject, target_grade, and target_skill are required")
        if int(state.user_id) != int(user_id):
            raise ValueError("user_id must match the supplied learner state")
        query = np.asarray(query_vector, dtype="float32").reshape(-1)
        if query.shape[0] != self.store.vectors.shape[1] or not np.isfinite(query).all():
            raise ValueError("query_vector must be finite and match the embedding dimension")
        query_norm = np.linalg.norm(query)
        if query_norm <= 1e-12:
            raise ValueError("query_vector must be non-zero")
        query = query / query_norm

        candidates = filter_exercise_bank(
            self.exercise_bank, target_subject, target_grade, target_skill
        )
        seen = set(state.seen) if seen_item_ids is None else seen_item_ids
        candidates = [
            item for item in candidates
            if item["item_id"] not in seen and item["mapping_status"] == "mapped"
        ]
        mastery_map = mastery_by_canonical_skill(state, self.source_skill_to_tag)
        mastery = self.mastery_for_skill(state, target_skill)
        mastery_source = (
            "canonical_skill_ema" if target_skill in mastery_map
            else "overall_accuracy_fallback" if state.n
            else "neutral_cold_start"
        )
        profile_vector = self._profile_vector(state, target_skill)
        results = []
        for item in candidates:
            chunk_indices = self.chunk_ids_by_doc.get(item["doc_id"])
            if not chunk_indices:
                raise ValueError(f"Exercise {item['item_id']} has no indexed document chunks")
            item_vector = self.store.vectors[chunk_indices[0]]
            semantic_score = _normalized_score(float(np.dot(query, item_vector)))
            raw_profile_score = (
                _normalized_score(float(np.dot(profile_vector, item_vector)))
                if profile_vector is not None else 0.5
            )
            delta = float(item["difficulty"]) - mastery
            zpd = zpd_score(delta)
            profile_score = raw_profile_score * zpd
            cf_success, cf_granularity = self.cf.predict_candidate(
                user_id, item, target_skill, float(item["difficulty"])
            )
            cf_score = float(np.clip(1.0 - abs(cf_success - config.TARGET_SUCCESS), 0.0, 1.0))
            cf_weight = 1.0 - self.alpha - self.beta
            final_score = (
                self.alpha * semantic_score
                + self.beta * profile_score
                + cf_weight * cf_score
            )
            if strategy == "hybrid":
                ranking_score = final_score
            elif strategy == "random":
                random_key = hashlib.sha256(f"{user_id}:{item['item_id']}".encode()).digest()
                ranking_score = int.from_bytes(random_key[:8], "big") / (2**64 - 1)
            elif strategy == "most_popular":
                ranking_score = self.cf.popularity(
                    item, target_skill, float(item["difficulty"])
                )
            elif strategy == "traditional_rag":
                ranking_score = semantic_score
            elif strategy == "pure_cf":
                ranking_score = cf_success
            else:  # pure_cb_zpd
                ranking_score = profile_score
            results.append({
                **item,
                "mastery": mastery,
                "mastery_source": mastery_source,
                "delta": delta,
                "zpd_score": zpd,
                "semantic_score": semantic_score,
                "profile_score": profile_score,
                "profile_cosine_score": raw_profile_score,
                "cf_score": cf_score,
                "cf_predicted_success": cf_success,
                "cf_granularity": cf_granularity,
                "cf_used_for_item_ranking": cf_weight > 0,
                "semantic_weight": self.alpha,
                "profile_weight": self.beta,
                "cf_weight": cf_weight,
                "final_score": final_score,
                "ranking_strategy": strategy,
                "ranking_score": ranking_score,
            })
        results.sort(key=lambda item: item["ranking_score"], reverse=True)
        return results[:top_k]
