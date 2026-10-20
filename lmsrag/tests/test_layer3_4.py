import numpy as np
import pandas as pd
import pytest

from lmsrag.layer1_data.learner_state import LearnerState
from lmsrag.layer3_retrieval.query import QueryRewriter
from lmsrag.layer3_retrieval.retriever import AdaptiveRetriever
from lmsrag.layer4_recommender import HybridRecommender, SkillCollaborativeFilter, zpd_score
from lmsrag.vectorstore import VectorStore


class FakeEncoder:
    def encode_queries(self, texts):
        assert texts == ["Find fraction, greatest common factor, and \\frac{1}{2}"]
        return np.array([[1.0, 0.0, 0.0]], dtype="float32")


class FakeReranker:
    def score(self, query, documents):
        return [float(i) for i, _ in enumerate(documents)]


def test_query_rewriter_corrects_typos_expands_abbreviations_and_preserves_math():
    rewriter = QueryRewriter()

    assert rewriter.rewrite("  Find   fracion and GCF; \\frac{1}{2} ") == (
        "Find fraction and greatest common factor; \\frac{1}{2}"
    )
    assert rewriter.rewrite("SUBSTRACT 2") == "SUBTRACT 2"


def test_adaptive_retriever_hard_filters_and_reranks():
    store = VectorStore(
        np.eye(3, dtype="float32"),
        [
            {"chunk_id": "a", "doc_id": "a", "text": "right 1", "skill_tag": "fractions",
             "subject": "Mathematics", "grade": "lower_secondary", "difficulty": 0.3},
            {"chunk_id": "b", "doc_id": "b", "text": "right 2", "skill_tag": "fractions",
             "subject": "Mathematics", "grade": "lower_secondary", "difficulty": 0.6},
            {"chunk_id": "c", "doc_id": "c", "text": "wrong grade", "skill_tag": "fractions",
             "subject": "Mathematics", "grade": "upper_secondary", "difficulty": 0.3},
        ],
    )
    retriever = AdaptiveRetriever(store, FakeEncoder(), FakeReranker())

    results = retriever.retrieve(
        "Find fracion, GCF, and \\frac{1}{2}",
        target_subject="Mathematics",
        target_grade="lower_secondary",
        target_skill="fractions",
        top_k=2,
        candidate_k=2,
    )

    assert [row["chunk_id"] for row in results] == ["b", "a"]
    assert all(row["grade"] == "lower_secondary" for row in results)
    assert retriever.retrieve(
        "fraction", target_subject="English", target_grade="lower_secondary",
        target_skill="fractions"
    ) == []


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (-0.10001, 0.3),
        (-0.1, 0.7),
        (0.04999, 0.7),
        (0.05, 1.0),
        (0.30, 1.0),
        (0.30001, 0.5),
        (0.50, 0.5),
        (0.50001, 0.2),
    ],
)
def test_zpd_score_piecewise_boundaries(delta, expected):
    assert zpd_score(delta) == expected


def test_zpd_score_rejects_non_finite_delta():
    with pytest.raises(ValueError, match="finite"):
        zpd_score(float("nan"))


def test_skill_cf_uses_history_and_skill_level():
    interactions = pd.DataFrame(
        [
            {"user_id": 1, "tag": "x", "correct": 1, "split": "history"},
            {"user_id": 1, "tag": "y", "correct": 0, "split": "history"},
            {"user_id": 2, "tag": "x", "correct": 1, "split": "history"},
            {"user_id": 2, "tag": "y", "correct": 1, "split": "history"},
            {"user_id": 3, "tag": "x", "correct": 0, "split": "history"},
            {"user_id": 3, "tag": "y", "correct": 1, "split": "history"},
            {"user_id": 1, "tag": "y", "correct": 1, "split": "test"},
        ]
    )
    model = SkillCollaborativeFilter(n_neighbors=1).fit(interactions)

    assert model.predict(1, "y") == pytest.approx(1.0)
    assert model.predict(999, "y") == pytest.approx(2 / 3)
    assert model.predict(1, "unknown") == pytest.approx(2 / 3)


def test_hybrid_recommender_applies_constraints_zpd_and_hybrid_formula():
    store = VectorStore(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
        [
            {"chunk_id": "doc-a#0", "doc_id": "doc-a", "text": "easy fraction example",
             "skill_tag": "Percent-Of", "difficulty": 0.3},
            {"chunk_id": "doc-b#0", "doc_id": "doc-b", "text": "hard fraction example",
             "skill_tag": "Percent-Of", "difficulty": 0.9},
        ],
    )
    exercises = [
        {"item_id": "T-a", "doc_id": "doc-a", "skill_tag": "Percent-Of", "mapping_status": "mapped",
         "subject": "Mathematics", "grade": "lower_secondary", "difficulty": 0.3},
        {"item_id": "T-b", "doc_id": "doc-b", "skill_tag": "Percent-Of", "mapping_status": "mapped",
         "subject": "Mathematics", "grade": "lower_secondary", "difficulty": 0.9},
        {"item_id": "T-wrong-grade", "doc_id": "doc-a", "skill_tag": "Percent-Of",
         "mapping_status": "mapped", "subject": "Mathematics", "grade": "upper_secondary",
         "difficulty": 0.3},
    ]
    catalog = {"source_skill_mappings": [
        {"source_skill_id": 7, "canonical_skill_tag": "Percent-Of"},
    ]}
    interactions = pd.DataFrame([
        {"user_id": 1, "tag": "Percent-Of", "correct": 0, "split": "history"},
        {"user_id": 2, "tag": "Percent-Of", "correct": 1, "split": "history"},
    ])
    recommender = HybridRecommender(
        store, exercises, catalog, interactions, alpha=0.4, beta=0.3, n_cf_neighbors=1
    )
    state = LearnerState(user_id=1)
    for _ in range(5):
        state.update(7, 0.0, 10_000, alpha=0.1, prior=0.2, bucket=0, tau_base=10_000)
    state.update_canonical(
        "Percent-Of", 0.0, 10_000, alpha=0.5, prior=0.2, bucket=0, tau_base=10_000
    )

    results = recommender.recommend(
        user_id=1,
        state=state,
        query_vector=np.array([1.0, 0.0]),
        target_subject="Mathematics",
        target_grade="lower_secondary",
        target_skill="Percent-Of",
        top_k=10,
    )

    assert {row["item_id"] for row in results} == {"T-a", "T-b"}
    assert all(row["mastery_source"] == "canonical_skill_ema" for row in results)
    assert all(row["mastery"] == state.canonical_M["Percent-Of"] for row in results)
    easy, hard = next(row for row in results if row["item_id"] == "T-a"), next(
        row for row in results if row["item_id"] == "T-b"
    )
    assert easy["delta"] == pytest.approx(0.3 - easy["mastery"])
    assert easy["zpd_score"] == 1.0
    assert hard["delta"] == pytest.approx(0.9 - hard["mastery"])
    assert hard["zpd_score"] == 0.2
    for row in results:
        expected = 0.4 * row["semantic_score"] + 0.3 * row["profile_score"] + 0.3 * row["cf_score"]
        assert row["final_score"] == pytest.approx(expected)
        assert row["cf_granularity"] == "canonical_skill"
    assert [row["item_id"] for row in recommender.recommend(
        user_id=1, state=state, query_vector=np.array([1.0, 0.0]),
        target_subject="Mathematics", target_grade="lower_secondary",
        target_skill="Percent-Of", seen_item_ids={"T-a"},
    )] == ["T-b"]


def test_recommendations_prefer_online_canonical_mastery_over_source_ema():
    from lmsrag.layer4_recommender import mastery_by_canonical_skill

    state = LearnerState(user_id=2, M={"7": 0.8}, attempts={"7": 3})
    state.canonical_M["Percent-Of"] = 0.25

    assert mastery_by_canonical_skill(state, {7: "Percent-Of"}) == {"Percent-Of": 0.25}


def test_query_encoder_rejects_kind_mismatch_with_index_manifest(tmp_path, monkeypatch):
    import json

    from lmsrag import config
    from lmsrag.layer3_retrieval.retriever import load_query_encoder

    monkeypatch.setattr(config, "ART", tmp_path)
    (tmp_path / "index_manifest.json").write_text(
        json.dumps({"embedder": {"kind": "lsa"}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not match index embedder"):
        load_query_encoder("bge")
