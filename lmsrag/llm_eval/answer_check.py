"""Safe mathematical answer comparison and five-run solver consensus."""
from __future__ import annotations

import ast
import math
import re
from typing import Any

import sympy as sp

_MAX_EXPRESSION_LENGTH = 512
_MAX_AST_NODES = 96
_MAX_EXPONENT = 100


def _normalize_expression(value: str) -> str:
    expression = value.strip().strip("`$").strip()
    expression = expression.replace("−", "-").replace("×", "*").replace("·", "*").replace("÷", "/")
    expression = re.sub(r"\\(?:left|right)\b", "", expression)
    expression = expression.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    expression = expression.replace(r"\pi", "pi").replace("π", "pi")
    expression = re.sub(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", expression)
    expression = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", expression)
    expression = re.sub(r"\\sqrt\s*\(([^()]*)\)", r"sqrt(\1)", expression)
    expression = re.sub(r"\\sqrt\s*(\d+)", r"sqrt(\1)", expression)
    expression = expression.replace("∛", "cbrt").replace("√", "sqrt")
    expression = re.sub(r"(?<![A-Za-z])sqrt\s*(\d+)", r"sqrt(\1)", expression)
    expression = re.sub(r"(?<![A-Za-z])cbrt\s*(\d+)", r"cbrt(\1)", expression)
    expression = expression.replace("^", "**")
    expression = re.sub(
        r"(?<=\d)\s*(?!(?:[eE][+-]?\d))(?=(?:sqrt|cbrt|[A-Za-z]|\())",
        "*",
        expression,
    )
    expression = re.sub(r"(?<=\))\s*(?=(?:[A-Za-z]|\())", "*", expression)
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Math expression exceeds {_MAX_EXPRESSION_LENGTH} characters")
    return expression


def _parse_node(node: ast.AST) -> sp.Expr:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are supported")
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise ValueError("Non-finite numeric constants are not supported")
        return sp.Rational(str(node.value)) if isinstance(node.value, float) else sp.Integer(node.value)
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return sp.pi
        if node.id == "e":
            return sp.E
        if node.id in {"sqrt", "cbrt"}:
            raise ValueError(f"{node.id} must be called with one argument")
        return sp.Symbol(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _parse_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _parse_node(node.left)
        right = _parse_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            if right.is_number and right.is_real is True and abs(right) > _MAX_EXPONENT:
                raise ValueError(f"Exponent magnitude must not exceed {_MAX_EXPONENT}")
            return left ** right
        raise ValueError(f"Unsupported math operator: {type(node.op).__name__}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords and len(node.args) == 1:
        argument = _parse_node(node.args[0])
        if node.func.id == "sqrt":
            return sp.sqrt(argument)
        if node.func.id == "cbrt":
            return sp.real_root(argument, 3)
        raise ValueError(f"Unsupported math function: {node.func.id}")
    raise ValueError(f"Unsupported math syntax: {type(node).__name__}")


def parse_math_expression(value: str | int | float) -> sp.Expr:
    """Parse arithmetic using a small AST allowlist; never evaluate input as Python."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Math answers must be strings or finite numbers")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Math answers must be finite")
    expression = _normalize_expression(str(value))
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid math expression: {value!r}") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise ValueError(f"Math expression exceeds {_MAX_AST_NODES} syntax nodes")
    return _parse_node(tree.body)


def _expected_expression(answer_key: str | int | float, answer_type: str | None) -> sp.Expr:
    expected = parse_math_expression(answer_key)
    normalized_type = (answer_type or "").casefold()
    if normalized_type == "radical":
        return sp.sqrt(expected)
    if normalized_type == "cuberoot":
        return sp.real_root(expected, 3)
    return expected


def answers_equivalent(
    candidate: str | int | float,
    answer_key: str | int | float,
    answer_type: str | None = None,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
) -> bool:
    """Compare exact/symbolic answers, restoring Khan radical answer formats."""
    if (answer_type or "").casefold() not in {"radical", "cuberoot"}:
        if isinstance(candidate, str) and isinstance(answer_key, str):
            if re.sub(r"\s+", "", candidate).casefold() == re.sub(r"\s+", "", answer_key).casefold():
                return True
    try:
        actual = parse_math_expression(candidate)
        expected = _expected_expression(answer_key, answer_type)
        difference = sp.simplify(actual - expected)
        if difference == 0 or difference.is_zero is True:
            return True
        if not difference.free_symbols:
            actual_value = float(sp.N(actual, 30))
            expected_value = float(sp.N(expected, 30))
            return math.isclose(actual_value, expected_value, rel_tol=rel_tol, abs_tol=abs_tol)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return False
    return False


def is_numeric_expression(problem: str) -> bool:
    """True only when the entire prompt is a parseable, symbol-free expression."""
    if not isinstance(problem, str) or not problem.strip():
        return False
    if not re.fullmatch(r"[0-9A-Za-z\s()+\-*/^.,{}\\$×·÷−√∛π]+", problem):
        return False
    try:
        return not parse_math_expression(problem).free_symbols
    except ValueError:
        return False


def solver_majority(
    solver_answers: list[str | int | float],
) -> str | int | float | None:
    """Return a strict majority answer from exactly five independent solver outputs."""
    if len(solver_answers) != 5:
        raise ValueError("Exactly five independent solver answers are required")
    groups: list[list[str | int | float]] = []
    for answer in solver_answers:
        if isinstance(answer, str) and not answer.strip():
            continue
        for group in groups:
            if answers_equivalent(answer, group[0]):
                group.append(answer)
                break
        else:
            groups.append([answer])
    winners = [group for group in groups if len(group) >= 3]
    return winners[0][0] if len(winners) == 1 else None


def evaluate_answer(
    problem: str,
    answer_key: str | int | float,
    candidate: str | int | float,
    *,
    answer_type: str | None = None,
    solver_answers: list[str | int | float] | None = None,
) -> dict[str, Any]:
    """Verify the candidate and answer key against SymPy or five-run consensus."""
    if is_numeric_expression(problem):
        solved = sp.simplify(parse_math_expression(problem))
        expected = str(solved)
        return {
            "method": "sympy",
            "correct": (
                answers_equivalent(expected, answer_key, answer_type)
                and answers_equivalent(candidate, answer_key, answer_type)
            ),
            "consensus_answer": expected,
        }
    if solver_answers is None:
        raise ValueError("Non-expression problems require five independent solver answers")
    consensus = solver_majority(solver_answers)
    return {
        "method": "solver_majority",
        "correct": (
            consensus is not None
            and answers_equivalent(consensus, answer_key, answer_type)
            and answers_equivalent(candidate, answer_key, answer_type)
        ),
        "consensus_answer": consensus,
    }
