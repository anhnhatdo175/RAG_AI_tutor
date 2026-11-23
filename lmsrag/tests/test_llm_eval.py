from types import SimpleNamespace

import pytest

from lmsrag.llm_eval.answer_check import (
    answers_equivalent,
    evaluate_answer,
    is_numeric_expression,
    parse_math_expression,
    solver_majority,
)
from lmsrag.llm_eval.grounding import (
    build_llm_judge_prompt,
    evaluate_nli_grounding,
    parse_judge_verdict,
    split_claims,
)
from lmsrag.llm_eval.requests import build_requests, eligible_skill_tags, write_requests
from lmsrag.llm_eval.text_metrics import (
    copying_metrics,
    duplicate_pairs,
    evaluate_skill_coverage,
    generation_validation_metrics,
    lcs_length,
    longest_common_run,
    ollama_usage_metrics,
    rouge_l_f1,
    socratic_answer_leakage,
)


def _request_config():
    return {
        "skill_count": 60,
        "dev_skill_count": 10,
        "mastery_levels": [0.3, 0.55, 0.8],
        "modes": ["practice", "socratic"],
        "skill_map": "unused.csv",
        "output_dir": "unused",
    }


def test_request_set_is_360_stable_and_split_by_skill():
    skills = [f"Skill-{index:02d}" for index in range(79)]
    first = build_requests(skills, _request_config(), seed=2026)
    second = build_requests(skills, _request_config(), seed=2026)
    assert first == second
    assert len(first) == 360
    dev_skills = {request["skill_tag"] for request in first if request["split"] == "dev"}
    eval_skills = {request["skill_tag"] for request in first if request["split"] == "eval"}
    assert len(dev_skills) == 10 and len(eval_skills) == 50
    assert not dev_skills & eval_skills
    assert all(request["target_level"] == expected for request, expected in zip(
        first[:6], [3, 3, 4, 4, 5, 5]
    ))
    socratic = next(request for request in first if request["mode"] == "socratic")
    assert [turn["scenario"] for turn in socratic["student_turns"]] == [
        "common_error", "stuck", "correct_intermediate_step",
    ]
    assert len({request["request_id"] for request in first}) == 360


def test_eligible_skills_require_mapped_usable_templates(tmp_path):
    skill_map = tmp_path / "skill_map.csv"
    skill_map.write_text(
        "source_dataset,canonical_tag,method,template_status,usable_problems\n"
        "assist2009,Addition-Whole-Numbers,rule_text,usable,2\n"
        "assist2009,No-Template,rule_text,none,0\n"
        ",khan:template_only,template_only,usable,10\n"
        "assist2012,No-Problems,rule_text,usable,0\n",
        encoding="utf-8",
    )
    assert eligible_skill_tags(skill_map) == ["Addition-Whole-Numbers"]


def test_request_file_is_frozen_and_rejects_different_content(tmp_path):
    output_path = tmp_path / "requests_v1.jsonl"
    request = {"request_id": "r1", "split": "dev"}
    assert write_requests([request], output_path) is True
    assert write_requests([request], output_path) is False
    with pytest.raises(FileExistsError, match="different contents"):
        write_requests([{"request_id": "r2"}], output_path)


def test_answer_check_handles_numeric_radical_cuberoot_and_solver_majority():
    assert is_numeric_expression(r"\frac{1}{2} + \frac{1}{3}")
    assert not is_numeric_expression("A train travels 12 miles in 2 hours.")
    assert answers_equivalent("0.8333333333", "5/6")
    assert answers_equivalent("1e3", "1000")
    assert answers_equivalent("3π", "3*pi")
    assert not answers_equivalent("0.84", "5/6")
    assert answers_equivalent("7*sqrt(3)", "147", "radical")
    assert answers_equivalent("2", "8", "cuberoot")
    assert solver_majority(["1/2", "0.5", "2/4", "0.6", "0.4"]) == "1/2"
    assert solver_majority(["1", "1", "2", "2", "3"]) is None
    assert evaluate_answer("3 + 4", "7", "7")["correct"] is True
    expression_result = evaluate_answer("3 + 4", "8", "8")
    assert expression_result["correct"] is False
    assert expression_result["consensus_answer"] == "7"
    correct_word_problem = evaluate_answer(
        "A box has 4 balls.",
        "4",
        "4",
        solver_answers=["4", "4", "4", "4", "3"],
    )
    assert correct_word_problem["correct"] is True
    wrong = evaluate_answer(
        "A student has 2 apples and gets 1 more.",
        "4",
        "3",
        solver_answers=["3", "3", "3", "3", "2"],
    )
    assert wrong["correct"] is False
    wrong_candidate = evaluate_answer(
        "A box has 4 balls.",
        "4",
        "3",
        solver_answers=["4", "4", "4", "4", "4"],
    )
    assert wrong_candidate["correct"] is False
    assert answers_equivalent("√147", "147", "radical")
    with pytest.raises(ValueError, match="five"):
        solver_majority(["1", "1"])


def test_math_parser_rejects_executable_python():
    with pytest.raises(ValueError):
        parse_math_expression("__import__('os').system('whoami')")


def test_nli_grounding_splits_claims_and_marks_no_source_not_applicable():
    assert split_claims("First fact. Second fact.", "The solution follows.") == [
        "First fact.", "Second fact.", "The solution follows.",
    ]
    scored = evaluate_nli_grounding(
        ["The sum is 7.", "The result is negative."],
        [{"text": "The sum is 7."}],
        lambda pairs: [0.91, 0.12],
    )
    assert [claim["supported"] for claim in scored["claims"]] == [True, False]
    assert scored["support_rate"] == 0.5
    assert evaluate_nli_grounding(["No source"], [], lambda pairs: [])["applicable"] is False
    prompt = build_llm_judge_prompt("The sum is 7.", ["The sum is 7."])
    assert "do not use outside knowledge" in prompt
    assert parse_judge_verdict('{"supported": true}') is True
    assert parse_judge_verdict("unsupported") is False
    with pytest.raises(ValueError):
        parse_judge_verdict('{"supported": true, "extra": false}')


def test_copy_duplicate_skill_coverage_and_usage_metrics():
    text = "Find the sum by adding the two given values and simplify the result."
    copied = copying_metrics(text, [{"doc_id": "d1", "text": text}])
    assert copied["copied"] is True and copied["common_tokens"] >= 12
    assert copying_metrics("An unrelated short prompt.", [{"text": "No shared words here."}])["copied"] is False
    assert lcs_length(["a", "b", "c"], ["a", "x", "c"]) == 2
    assert longest_common_run(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert rouge_l_f1(["a", "b"], ["a", "b"]) == 1.0
    pairs = duplicate_pairs(
        [
            {"request_id": "a", "skill_tag": "s1"},
            {"request_id": "b", "skill_tag": "s1"},
            {"request_id": "c", "skill_tag": "s2"},
        ],
        [[1.0, 0.0], [0.995, 0.1], [1.0, 0.0]],
    )
    assert [(pair["left_id"], pair["right_id"]) for pair in pairs] == [("a", "b")]
    assert duplicate_pairs(
        [{"skill_tag": "s1"}, {"skill_tag": "s1"}],
        [[1.0, 0.0], [0.0, 1.0]],
    ) == []

    class Mapper:
        def tag_text(self, problem, subject, level, is_gsm8k):
            return "s1", "Skill one", "high"

    coverage = evaluate_skill_coverage(text, "s1", Mapper(), lambda problem, target: True)
    assert coverage["covered"] is True and coverage["agreement"] is True
    missed = evaluate_skill_coverage(text, "s2", Mapper(), lambda problem, target: False)
    assert missed["covered"] is False and missed["agreement"] is True
    assert generation_validation_metrics({
        "generation_attempts": 2,
        "normalized_safety": True,
    }) == {
        "json_valid": True,
        "retry_count": 1,
        "normalized_safety": True,
        "error": None,
    }
    failed = generation_validation_metrics(SimpleNamespace(attempts=2))
    assert failed["json_valid"] is False
    assert failed["retry_count"] == 1
    assert failed["normalized_safety"] is None
    usage = ollama_usage_metrics(
        {"total_duration": 10_000_000, "prompt_eval_count": 20, "eval_count": 7},
        retrieval_latency_ms=12.5,
    )
    assert usage == {
        "total_duration": 10_000_000,
        "prompt_eval_count": 20,
        "eval_count": 7,
        "retrieval_latency_ms": 12.5,
    }


def test_socratic_answer_leakage_detects_equivalent_answer_before_student():
    leaked = socratic_answer_leakage(
        [
            {"role": "assistant", "content": "The result is 0.5."},
            {"role": "student", "content": "I am not sure."},
        ],
        "1/2",
    )
    assert leaked["leaked"] is True and leaked["event_index"] == 0
    safe = socratic_answer_leakage(
        [
            {"role": "student", "content": "My answer is 1/2."},
            {"role": "assistant", "content": "Correct, the result is 0.5."},
        ],
        "1/2",
    )
    assert safe["leaked"] is False
    normalized_symbolic_leak = socratic_answer_leakage(
        [{"role": "assistant", "content": "We can write x = 2 here."}],
        "x=2",
    )
    assert normalized_symbolic_leak["leaked"] is True
    radical_leak = socratic_answer_leakage(
        [{"role": "assistant", "content": "The simplified result is 7√3."}],
        "147",
        answer_type="radical",
    )
    assert radical_leak["leaked"] is True
