import json

import numpy as np
from lmsrag.application import LearningPipeline
from lmsrag.layer1_data.state_store import StateStore
from lmsrag.layer5_generation import LLMGenerator


class FakeEncoder:
    def encode_queries(self, texts):
        assert texts == ["practice fractions"]
        return np.array([[1.0, 0.0]], dtype="float32")


class FakeRetriever:
    def retrieve(self, query, **constraints):
        assert query == "practice fractions"
        assert constraints["target_skill"] == "fractions"
        return [
            {
                "doc_id": "doc-1",
                "skill_tag": "fractions",
                "subject": "Mathematics",
                "grade": "lower_secondary",
                "difficulty": 0.6,
                "text": "Use a common denominator to add fractions.",
            },
            {
                "doc_id": "doc-2",
                "skill_tag": "fractions",
                "subject": "Mathematics",
                "grade": "lower_secondary",
                "difficulty": 0.6,
                "text": "Rewrite fractions with a shared denominator before adding.",
            },
        ]


class FakeCollaborativeFilter:
    def personalization_confidence(self, user_id, skill_tag):
        return 0.5

    def predict_candidate(self, user_id, item, skill_tag, difficulty):
        return 0.7, "skill"


class FakeRecommender:
    def __init__(self):
        self.item = {
            "item_id": "T-fraction-1",
            "doc_id": "doc-1",
            "skill_tag": "fractions",
            "mapping_status": "mapped",
            "subject": "Mathematics",
            "grade": "lower_secondary",
            "difficulty": 0.6,
            "statement": "What is 1/3 + 1/4?",
            "gold_answer": "7/12",
        }
        self.exercise_bank = [self.item]
        self.cf = FakeCollaborativeFilter()

    def profile_vector_for(self, state, skill_tag):
        return np.array([1.0, 0.0], dtype="float32")

    def mastery_for_skill(self, state, skill_tag):
        return 0.5

    def recommend(self, *, state, **kwargs):
        if self.item["item_id"] in state.seen:
            return []
        return [self.item]


def _response():
    return {
        "schema_version": "1.0",
        "status": "ok",
        "mode": "practice",
        "task": {
            "item_id": "untrusted-id",
            "skill_tag": "fractions",
            "difficulty": 0.6,
            "prompt": "What is 1/3 + 1/4?",
            "response_type": "exact",
            "hints": ["Find a common denominator."],
            "personalization_reason": "This skill is ready for practice.",
            "personalization_basis": ["overall_accuracy"],
            "supporting_doc_ids": ["doc-2"],
            "answer_key": "7/12",
        },
        "tutor_turn": None,
        "safety": {"contains_final_answer": False, "grounded": True},
    }


def _pipeline(tmp_path):
    state_store = StateStore(tmp_path / "learner.sqlite")
    generator = LLMGenerator(lambda prompt, **kwargs: json.dumps(_response()))
    return LearningPipeline(
        state_store=state_store,
        retriever=FakeRetriever(),
        recommender=FakeRecommender(),
        encoder=FakeEncoder(),
        generator=generator,
    )


def test_pipeline_generates_grades_and_persists_feedback(tmp_path):
    pipeline = _pipeline(tmp_path)
    try:
        activity = pipeline.prepare_activity(
            user_id=18,
            query="practice fractions",
            target_subject="Mathematics",
            target_grade="lower_secondary",
            target_skill="fractions",
        )

        assert activity.item_id.startswith("GEN-")
        assert activity.generation["task"]["item_id"] == activity.item_id
        assert "7/12" not in json.dumps(activity.generation)

        result = pipeline.submit_answer(
            user_id=18,
            task_id=activity.task_id,
            response="7/12",
            rt_ms=12000,
        )

        assert result["status"] == "graded"
        assert result["score"] == 1.0
        state = pipeline.state_store.get(18)
        assert state.canonical_attempts["fractions"] == 1
        assert state.seen == [activity.item_id]
    finally:
        pipeline.close()


def test_pipeline_resumes_an_issued_activity_after_reopen(tmp_path):
    pipeline = _pipeline(tmp_path)
    activity = pipeline.prepare_activity(
        user_id=18,
        query="practice fractions",
        target_subject="Mathematics",
        target_grade="lower_secondary",
        target_skill="fractions",
    )
    pipeline.close()

    reopened = _pipeline(tmp_path)
    try:
        pending = reopened.get_pending_activity(18)
        assert pending is not None
        assert pending.task_id == activity.task_id
        assert pending.item_id == activity.item_id
    finally:
        reopened.close()


def test_pipeline_recovers_submission_interrupted_after_persisting_response(tmp_path):
    pipeline = _pipeline(tmp_path)
    activity = pipeline.prepare_activity(
        user_id=18,
        query="practice fractions",
        target_subject="Mathematics",
        target_grade="lower_secondary",
        target_skill="fractions",
    )
    pipeline.issued_tasks.begin_submission(
        activity.task_id, "stable-event-id", "7/12", 12000
    )
    pipeline.close()

    reopened = _pipeline(tmp_path)
    try:
        result = reopened.submit_answer(
            user_id=18,
            task_id=activity.task_id,
        )
        assert result["status"] == "graded"
        assert result["score"] == 1.0
        assert reopened.state_store.get(18).canonical_attempts["fractions"] == 1
    finally:
        reopened.close()
