import json

import pytest

from lmsrag.layer1_data.learner_state import LearnerState
from lmsrag.layer1_data.state_store import StateStore
from lmsrag.layer5_generation import GenerationContext, LLMGenerator, SocraticTutor
from lmsrag.layer6_feedback import FeedbackLoop, GradingResult


def _context(mode="practice"):
    return GenerationContext(
        retrieved_docs=[{"doc_id": "doc-1", "text": "Find a common denominator."}],
        candidate_tasks=[{
            "item_id": "task-1",
            "skill_tag": "fractions",
            "difficulty": 0.6,
            "mapping_status": "mapped",
        }],
        learner_profile={"canonical_mastery": {"fractions": 0.4}},
        constraints={"mode": mode, "target_skill": "fractions"},
    )


def _generation(mode="practice", item_id="task-1", doc_id="doc-1"):
    return {
        "schema_version": "1.0",
        "status": "ok",
        "mode": mode,
        "task": {
            "item_id": item_id,
            "skill_tag": "fractions",
            "difficulty": 0.6,
            "prompt": "Add 1/3 and 1/4.",
            "response_type": "exact",
            "hints": ["Find a common denominator."],
            "personalization_reason": "Mastery is currently 0.4.",
            "personalization_basis": ["canonical_mastery.fractions"],
            "supporting_doc_ids": [doc_id],
        },
        "tutor_turn": {
            "turn_index": 1,
            "question": "What denominator can both fractions use?",
            "follow_up_questions": ["What do you do after finding a common denominator?"],
            "wait_for_learner": True,
        } if mode == "socratic" else None,
        "safety": {"contains_final_answer": False, "grounded": True},
    }


def test_generator_builds_context_and_validates_response():
    response = _generation()
    received = {}

    def fake_generate(prompt, *, json_mode):
        received["prompt"] = prompt
        received["json_mode"] = json_mode
        return json.dumps(response), 0.01

    result = LLMGenerator(fake_generate).generate(_context())

    assert result["task"]["personalization_reason"]
    assert received["json_mode"] is True
    assert '"canonical_mastery"' in received["prompt"]


def test_context_builder_limits_sources_and_hides_private_task_fields():
    state = LearnerState(user_id=4, seen=["seen-task"])
    context = GenerationContext.from_sources(
        retrieved_docs=[
            {"doc_id": "d1", "skill_tag": "fractions", "subject": "Mathematics", "grade": "lower_secondary"},
            {"doc_id": "d2", "skill_tag": "fractions", "subject": "Mathematics", "grade": "upper_secondary"},
        ],
        candidate_tasks=[
            {"item_id": "seen-task", "skill_tag": "fractions"},
            {
                "item_id": "task-1",
                "skill_tag": "fractions",
                "mapping_status": "mapped",
                "subject": "Mathematics",
                "grade": "lower_secondary",
                "grading": {"answerKey": "private", "rubric_id": "private"},
            },
            {"item_id": "wrong-skill", "skill_tag": "algebra", "mapping_status": "mapped"},
            {"item_id": "wrong-grade", "skill_tag": "fractions", "mapping_status": "mapped", "subject": "Mathematics", "grade": "upper_secondary"},
        ],
        learner_state=state,
        constraints={
            "mode": "practice",
            "target_skill": "fractions",
            "target_subject": "Mathematics",
            "target_grade": "lower_secondary",
        },
        top_k=1,
        top_n=2,
    )

    assert [doc["doc_id"] for doc in context.retrieved_docs] == ["d1"]
    assert [task["item_id"] for task in context.candidate_tasks] == ["task-1"]
    prompt = LLMGenerator._build_prompt(context)
    assert "private" not in prompt


@pytest.mark.parametrize(("top_k", "top_n"), [(0, 1), (1, 0), (1.5, 1), (True, 1)])
def test_context_builder_rejects_invalid_limits(top_k, top_n):
    with pytest.raises(ValueError, match="positive integer"):
        GenerationContext.from_sources(
            retrieved_docs=[],
            candidate_tasks=[],
            learner_state=LearnerState(user_id=1),
            constraints={},
            top_k=top_k,
            top_n=top_n,
        )


@pytest.mark.parametrize(
    ("item_id", "doc_id", "message"),
    [
        ("not-in-context", "doc-1", "candidate"),
        ("task-1", "not-in-context", "supporting_doc_ids"),
    ],
)
def test_generator_rejects_unverified_references(item_id, doc_id, message):
    response = _generation(item_id=item_id, doc_id=doc_id)
    generator = LLMGenerator(lambda prompt, **kwargs: json.dumps(response))

    with pytest.raises(ValueError, match=message):
        generator.generate(_context())


@pytest.mark.parametrize(
    ("candidate_update", "message"),
    [
        ({"mapping_status": "unmapped"}, "mapping_status=mapped"),
        ({"difficulty": 0.9}, "difficulty does not match"),
    ],
)
def test_generator_rejects_changed_or_unmapped_candidates(candidate_update, message):
    context = _context()
    context.candidate_tasks[0].update(candidate_update)
    generator = LLMGenerator(lambda prompt, **kwargs: json.dumps(_generation()))

    with pytest.raises(ValueError, match=message):
        generator.generate(context)


def test_generator_rejects_prompt_changed_for_bank_answer_key():
    context = _context()
    context.candidate_tasks[0]["statement"] = "Original issued problem"
    response = _generation()
    response["task"]["prompt"] = "A different problem with the same item ID"
    generator = LLMGenerator(lambda prompt, **kwargs: json.dumps(response))

    with pytest.raises(ValueError, match="preserve the selected candidate statement"):
        generator.generate(context)


def test_generator_rejects_inconsistent_insufficient_context_safety():
    response = {
        "schema_version": "1.0",
        "status": "insufficient_context",
        "mode": "practice",
        "task": None,
        "tutor_turn": None,
        "safety": {"contains_final_answer": False, "grounded": True},
    }
    generator = LLMGenerator(lambda prompt, **kwargs: json.dumps(response))

    with pytest.raises(ValueError, match="grounded=false"):
        generator.generate(_context())


def test_socratic_generation_requires_no_answer_leakage():
    response = _generation(mode="socratic")
    generator = LLMGenerator(lambda prompt, **kwargs: json.dumps(response))
    generated = generator.generate(_context(mode="socratic"))
    tutor = SocraticTutor()
    session = tutor.start(generated, max_steps=3)

    hint = tutor.submit(session, 0.0)
    assert hint["status"] == "hint"
    assert hint["hint"] == "Find a common denominator."
    assert hint["complete"] is False
    next_step = tutor.submit(session, 1.0)
    assert next_step["status"] == "next_step"
    assert next_step["question"] == "What do you do after finding a common denominator?"
    assert tutor.submit(session, 1.0)["status"] == "correct"


def test_socratic_generation_rejects_final_answer_flag():
    response = _generation(mode="socratic")
    response["safety"]["contains_final_answer"] = True
    generator = LLMGenerator(lambda prompt, **kwargs: json.dumps(response))

    with pytest.raises(ValueError, match="final answer"):
        generator.generate(_context(mode="socratic"))


def test_generator_rejects_ungrounded_response():
    response = _generation()
    response["safety"]["grounded"] = False
    generator = LLMGenerator(lambda prompt, **kwargs: json.dumps(response))

    with pytest.raises(ValueError, match="grounded=true"):
        generator.generate(_context())


def test_generator_requires_personalization_basis_from_profile():
    response = _generation()
    response["task"]["personalization_basis"] = ["canonical_mastery.algebra"]
    generator = LLMGenerator(lambda prompt, **kwargs: json.dumps(response))

    with pytest.raises(ValueError, match="not in learner_profile"):
        generator.generate(_context())


def test_exercise_bank_gold_answer_is_private_and_usable_by_grader(tmp_path):
    bank_task = {
        "item_id": "T-doc-1",
        "doc_id": "doc-1",
        "skill_tag": "fractions",
        "mapping_status": "mapped",
        "subject": "Mathematics",
        "grade": "lower_secondary",
        "difficulty": 0.6,
        "statement": "What is 1/2 + 1/4?",
        "gold_answer": "3/4",
    }
    context = GenerationContext(
        retrieved_docs=[{"doc_id": "doc-1", "skill_tag": "fractions"}],
        candidate_tasks=[bank_task],
        learner_profile={"canonical_mastery": {"fractions": 0.5}},
        constraints={"mode": "practice", "target_skill": "fractions"},
    )
    assert "3/4" not in LLMGenerator._build_prompt(context)

    store = StateStore(tmp_path / "learner.sqlite")
    issued_task = {
        **bank_task,
        "user_id": 7,
        "status": "issued",
        "response_type": "exact",
        "difficulty_bucket": 1,
    }
    result = FeedbackLoop(store).submit(
        user_id=7,
        event_id="event-gold-answer",
        task=issued_task,
        response="3/4",
        rt_ms=5000,
    )
    assert result["score"] == 1.0
    store.close()


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ('{"status":"ok","status":"bad"}', "duplicate JSON key"),
        ('{"difficulty":NaN}', "invalid JSON number"),
    ],
)
def test_generator_rejects_ambiguous_or_nonstandard_json(raw, message):
    with pytest.raises(ValueError, match=message):
        LLMGenerator._parse_json(raw)


def test_socratic_generation_rejects_questions_beyond_step_limit():
    response = _generation(mode="socratic")
    response["tutor_turn"]["follow_up_questions"] = ["Step two?", "Step three?"]
    context = _context(mode="socratic")
    context.constraints["max_steps"] = 2
    generator = LLMGenerator(lambda prompt, **kwargs: json.dumps(response))

    with pytest.raises(ValueError, match="exceed constraints.max_steps"):
        generator.generate(context)


def test_feedback_grades_and_applies_event_once(tmp_path):
    store = StateStore(tmp_path / "learner.sqlite")
    loop = FeedbackLoop(store, alpha=0.25)
    task = {
        "user_id": 7,
        "status": "issued",
        "item_id": "task-1",
        "skill_tag": "fractions",
        "difficulty": 0.6,
        "mapping_status": "mapped",
        "response_type": "numeric",
        "answer_key": 0.75,
        "tolerance": 0.001,
    }

    first = loop.submit(user_id=7, event_id="event-1", task=task, response="0.75", rt_ms=10_000)
    duplicate = loop.submit(user_id=7, event_id="event-1", task=task, response="0.75", rt_ms=10_000)

    assert first["score"] == 1.0
    assert duplicate["state"].canonical_attempts["fractions"] == 1
    assert duplicate["state"].canonical_M["fractions"] == pytest.approx(0.625)
    assert duplicate["context_snapshot"]["canonical_mastery"]["fractions"] == pytest.approx(0.625)
    with pytest.raises(ValueError, match="different submission"):
        loop.submit(user_id=7, event_id="event-1", task=task, response="0.0", rt_ms=10_000)
    with pytest.raises(ValueError, match="task was already applied"):
        loop.submit(user_id=7, event_id="event-2", task=task, response="0.75", rt_ms=10_000)
    store.close()


def test_feedback_does_not_update_ungraded_submission(tmp_path):
    store = StateStore(tmp_path / "learner.sqlite")
    loop = FeedbackLoop(store, grader=lambda task, response: GradingResult("needs_review", None))
    task = {"user_id": 7, "status": "issued", "item_id": "task-1", "skill_tag": "fractions", "difficulty_bucket": 1}

    result = loop.submit(user_id=7, event_id="event-review", task=task, response="unclear", rt_ms=10_000)

    assert result == {"status": "needs_review", "score": None, "state": None}
    assert store.get(7) is None
    store.close()


@pytest.mark.parametrize(
    ("task", "grader", "message"),
    [
        ({"user_id": 7, "status": "issued", "item_id": "task-1", "skill_tag": "fractions", "difficulty": 9}, None, "difficulty"),
        (
            {"user_id": 7, "status": "issued", "item_id": "task-1", "skill_tag": "fractions", "difficulty_bucket": 1},
            lambda task, response: GradingResult("unknown", None),
            "grader status",
        ),
    ],
)
def test_feedback_rejects_invalid_parameters_and_grader_status(tmp_path, task, grader, message):
    store = StateStore(tmp_path / "learner.sqlite")
    loop = FeedbackLoop(store, grader=grader)

    with pytest.raises(ValueError, match=message):
        loop.submit(user_id=7, event_id="event-invalid", task=task, response="x", rt_ms=1000)
    assert store.get(7) is None
    store.close()


def test_feedback_uses_neutral_speed_without_baseline(tmp_path):
    store = StateStore(tmp_path / "learner.sqlite")
    loop = FeedbackLoop(store, alpha=0.25)
    task = {
        "user_id": 7,
        "status": "issued",
        "item_id": "task-1",
        "skill_tag": "fractions",
        "difficulty_bucket": 1,
        "response_type": "exact",
        "answer_key": "correct",
    }

    result = loop.submit(user_id=7, event_id="event-neutral", task=task, response="correct", rt_ms=10_000)

    assert result["state"].canonical_prof["fractions"][1] == pytest.approx(0.85)
    store.close()


@pytest.mark.parametrize(
    ("rt_ms", "prior"),
    [
        (float("nan"), 0.5),
        (1000, float("inf")),
        (-1, 0.5),
        (1000, 1.1),
    ],
)
def test_feedback_rejects_non_finite_or_out_of_range_inputs(tmp_path, rt_ms, prior):
    store = StateStore(tmp_path / "learner.sqlite")
    loop = FeedbackLoop(store)
    task = {"user_id": 7, "status": "issued", "item_id": "task-1", "skill_tag": "fractions", "difficulty_bucket": 1}

    with pytest.raises(ValueError):
        loop.submit(user_id=7, event_id="event-invalid", task=task, response="x", rt_ms=rt_ms, prior=prior)
    assert store.get(7) is None
    store.close()