"""Đánh giá faithfulness độc lập với correctness trên toàn bộ benchmark."""

import json
import re

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from src import config
from src.llm_client import generate_answer


LABELS = ("grounded", "fabricated", "contradicts")
METHODS = ("RAG", "LLM thuần")

FAITHFULNESS_PROMPT = """Bạn là một LLM-judge đánh giá độ trung thực của câu trả lời.
Không được dùng đáp án vàng/gold answer. Chỉ dùng support text khi support text được cung cấp.

Quy tắc:
- grounded: mọi thông tin trong câu trả lời đều có căn cứ trong support text.
  Nếu support text trống (LLM thuần), nhãn này nghĩa là các thông tin là kiến thức
  phổ thông đúng và có thể kiểm chứng.
- fabricated: câu trả lời có chi tiết hoặc số liệu không xuất hiện trong support text
  và không thể kiểm chứng là kiến thức phổ thông đúng.
- contradicts: câu trả lời mâu thuẫn trực tiếp với support text.
  Nếu vừa mâu thuẫn vừa bịa, ưu tiên nhãn contradicts.

Hãy trả về duy nhất JSON hợp lệ, không markdown, đúng schema:
{{"label": "grounded|fabricated|contradicts", "reason": "giải thích ngắn gọn"}}

Câu hỏi:
{question}

Support text (có thể trống với LLM thuần):
{support_text}

Câu trả lời cần chấm:
{answer}
"""


def _parse_judgment(raw: str) -> dict[str, str]:
    """Parse strict JSON while tolerating a judge wrapping it in a code fence."""
    candidate = raw.strip()
    fenced = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(0)
    try:
        judgment = json.loads(candidate)
    except json.JSONDecodeError as exc:
        # Small local models occasionally close an otherwise valid JSON object
        # with a square bracket. Repair only this unambiguous terminal typo.
        if candidate.startswith("{") and candidate.endswith("]"):
            try:
                judgment = json.loads(candidate[:-1] + "}")
            except json.JSONDecodeError:
                raise ValueError(
                    f"Judge trả về JSON không hợp lệ: {raw!r}"
                ) from exc
        else:
            raise ValueError(f"Judge trả về JSON không hợp lệ: {raw!r}") from exc

    if not isinstance(judgment, dict):
        raise ValueError("Judge response phải là một JSON object.")
    label = judgment.get("label")
    reason = judgment.get("reason")
    if label not in LABELS:
        raise ValueError(f"Nhãn faithfulness không hợp lệ: {label!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Judge response thiếu reason dạng chuỗi.")
    return {"label": label, "reason": reason.strip()}


def judge_faithfulness(
    question: str, support_text: str, answer: str
) -> dict[str, str]:
    prompt = FAITHFULNESS_PROMPT.format(
        question=question,
        support_text=support_text,
        answer=answer,
    )
    raw, _ = generate_answer(prompt, model=config.JUDGE_MODEL, json_mode=True)
    return _parse_judgment(raw)


def _load_checkpoint() -> list[dict]:
    if not config.FAITHFULNESS_JSON.exists():
        return []
    with config.FAITHFULNESS_JSON.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"{config.FAITHFULNESS_JSON} phải chứa JSON array.")
    return rows


def _save_checkpoint(rows: list[dict]) -> None:
    temporary_path = config.FAITHFULNESS_JSON.with_suffix(".tmp.json")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    temporary_path.replace(config.FAITHFULNESS_JSON)


def _write_summary(rows: list[dict]) -> None:
    judgments = pd.DataFrame(rows)
    judgments.to_csv(config.FAITHFULNESS_CSV, index=False, encoding="utf-8-sig")

    rates = (
        judgments.groupby(["method", "label"]).size()
        .unstack(fill_value=0)
        .reindex(index=METHODS, columns=LABELS, fill_value=0)
    )
    percentages = rates.div(rates.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(7, 4.5))
    x = range(len(LABELS))
    width = 0.36
    colors = {"RAG": "#1D9E75", "LLM thuần": "#D85A30"}
    for offset, method in [(-width / 2, "RAG"), (width / 2, "LLM thuần")]:
        values = percentages.loc[method].tolist()
        bars = plt.bar(
            [position + offset for position in x],
            values,
            width,
            label=method,
            color=colors[method],
        )
        plt.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
    plt.xticks(list(x), LABELS)
    plt.ylabel("Tỷ lệ (%)")
    plt.title("So sánh faithfulness: RAG vs LLM thuần")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.FAITHFULNESS_CHART, dpi=150)
    plt.close()

    markdown = [
        "# Đánh giá faithfulness",
        "",
        f"- Judge model: `{config.JUDGE_MODEL}`",
        "- Context RAG: retrieved support thực tế được đưa vào prompt",
        f"- Số câu đã chấm: {len(judgments)} (mỗi câu gồm cả RAG và LLM thuần)",
        "",
        "| Phương pháp | grounded (%) | fabricated (%) | contradicts (%) | Tổng câu |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        markdown.append(
            f"| {method} | {percentages.loc[method, 'grounded']:.1f} | "
            f"{percentages.loc[method, 'fabricated']:.1f} | "
            f"{percentages.loc[method, 'contradicts']:.1f} | "
            f"{int(rates.loc[method].sum())} |"
        )
    markdown.extend(
        [
            "",
            f"![Biểu đồ faithfulness]({config.FAITHFULNESS_CHART.name})",
            "",
            "Chi tiết từng câu (gồm câu hỏi, câu trả lời và lý do của judge) nằm trong "
            f"`{config.FAITHFULNESS_CSV.name}` và `{config.FAITHFULNESS_JSON.name}`.",
        ]
    )
    config.FAITHFULNESS_MARKDOWN.write_text("\n".join(markdown) + "\n", encoding="utf-8")


def run_faithfulness() -> pd.DataFrame:
    benchmark = pd.read_csv(config.RESULTS_CSV)
    required = {
        "question",
        "rag_answer",
        "baseline_answer",
        "rag_retrieved_texts",
    }
    missing = required - set(benchmark.columns)
    if missing:
        raise ValueError(
            "Benchmark thiếu retrieved support thực tế. Hãy xóa checkpoint cũ "
            f"và chạy lại src.evaluate. Thiếu cột: {sorted(missing)}"
        )

    rows = _load_checkpoint()
    if any(row.get("context_source") != "retrieved_support" for row in rows):
        raise ValueError(
            "Faithfulness checkpoint không dùng retrieved support thực tế; "
            "hãy xóa faithfulness_judgments.json rồi chạy lại."
        )
    completed = {(row["question"], row["method"]) for row in rows}

    for _, item in tqdm(benchmark.iterrows(), total=len(benchmark)):
        question = str(item["question"])
        retrieved_texts = json.loads(item["rag_retrieved_texts"])
        if not isinstance(retrieved_texts, list) or not retrieved_texts:
            raise ValueError(f"RAG không có retrieved support cho câu hỏi: {question}")
        rag_support = "\n\n".join(
            f"- {str(text)}" for text in retrieved_texts
        )
        for method, answer, judge_support in (
            ("RAG", item["rag_answer"], rag_support),
            ("LLM thuần", item["baseline_answer"], ""),
        ):
            key = (question, method)
            if key in completed:
                continue
            judgment = judge_faithfulness(question, judge_support, str(answer))
            rows.append(
                {
                    "question": question,
                    "method": method,
                    "answer": str(answer),
                    "support_text": judge_support,
                    "context_source": "retrieved_support",
                    "judge_model": config.JUDGE_MODEL,
                    "label": judgment["label"],
                    "reason": judgment["reason"],
                }
            )
            completed.add(key)
            _save_checkpoint(rows)

    _write_summary(rows)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = run_faithfulness()
    print(f"Saved {len(result)} faithfulness judgments to {config.OUT_DIR}")
