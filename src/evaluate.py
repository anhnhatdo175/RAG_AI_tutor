"""Run the RAG and baseline benchmark and save answer and retrieval metrics."""
import json
import pandas as pd
from tqdm import tqdm

from src import config
from src.ingestion import load_all, build_corpus, build_support_to_docid, sample_test_questions
from src.retriever import load_index
from src.pipelines import rag_answer, baseline_answer
from src.llm_client import generate_answer


JUDGE_PROMPT = """Câu hỏi: {question}
Đáp án tham chiếu: {correct_answer}
Câu trả lời cần chấm: {model_answer}

Chấm câu trả lời trên theo đúng một trong ba nhãn sau, dựa CHỈ vào tính chính xác \
nội dung so với đáp án tham chiếu, bỏ qua khác biệt về cách diễn đạt:
DUNG / SAI / MOT_PHAN

Chỉ trả lời đúng một từ trong ba nhãn trên, không giải thích thêm."""


def judge_answer(question: str, correct_answer: str, model_answer: str) -> str:
    prompt = JUDGE_PROMPT.format(
        question=question, correct_answer=correct_answer, model_answer=model_answer
    )
    verdict, _ = generate_answer(prompt)
    verdict = verdict.strip().upper()
    if "DUNG" in verdict:
        return "DUNG"
    if "SAI" in verdict:
        return "SAI"
    return "MOT_PHAN"


def precision_recall_at_k(retrieved_doc_ids: list[str], gold_doc_id: str, k_values: list[int]):
    result = {}
    for k in k_values:
        top_k = retrieved_doc_ids[:k]
        hit = 1 if gold_doc_id in top_k else 0
        result[f"precision@{k}"] = hit / k
        result[f"recall@{k}"] = hit
    return result


def _save_checkpoint(rows: list[dict]) -> None:
    """Persist completed rows atomically so an interrupted run can resume."""
    results_df = pd.DataFrame(rows)
    temporary_path = config.RESULTS_CSV.with_suffix(".tmp.csv")
    results_df.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    temporary_path.replace(config.RESULTS_CSV)


def run_benchmark():
    full_df = load_all()
    corpus = build_corpus(full_df)
    support_to_docid = build_support_to_docid(corpus)

    index, docs = load_index()
    test_questions = sample_test_questions(full_df, config.SAMPLE_SIZE)

    rows = []
    if config.RESULTS_CSV.exists():
        checkpoint_df = pd.read_csv(config.RESULTS_CSV)
        if "question" not in checkpoint_df.columns:
            raise ValueError(
                f"Checkpoint {config.RESULTS_CSV} thiếu cột question và không thể tiếp tục."
            )
        rows = checkpoint_df.to_dict("records")

    completed_questions = {str(item["question"]) for item in rows}
    pending_questions = test_questions[
        ~test_questions["question"].astype(str).isin(completed_questions)
    ]
    print(
        f"Checkpoint: đã có {len(completed_questions)}/{len(test_questions)} câu; "
        f"còn {len(pending_questions)} câu."
    )

    for _, row in tqdm(pending_questions.iterrows(), total=len(pending_questions)):
        question = row["question"]
        correct_answer = row["correct_answer"]
        gold_doc_id = support_to_docid.get(row["support"])
        if gold_doc_id is None:
            raise ValueError(
                "Gold support không có trong corpus; hãy chạy ingestion theo cách 1."
            )

        rag_result = rag_answer(question, index, docs, k=max(config.K_VALUES))
        rag_verdict = judge_answer(question, correct_answer, rag_result["answer"])
        pr_at_k = precision_recall_at_k(
            rag_result["retrieved_doc_ids"], gold_doc_id, config.K_VALUES
        )

        base_result = baseline_answer(question)
        base_verdict = judge_answer(question, correct_answer, base_result["answer"])

        row_data = {
            "question": question,
            "correct_answer": correct_answer,
            "rag_answer": rag_result["answer"],
            "rag_verdict": rag_verdict,
            "rag_latency": rag_result["latency"],
            "rag_retrieved_doc_ids": json.dumps(
                rag_result["retrieved_doc_ids"], ensure_ascii=False
            ),
            "rag_retrieved_texts": json.dumps(
                rag_result["retrieved_texts"], ensure_ascii=False
            ),
            "rag_retrieved_scores": json.dumps(rag_result["retrieved_scores"]),
            "baseline_answer": base_result["answer"],
            "baseline_verdict": base_verdict,
            "baseline_latency": base_result["latency"],
        }
        row_data.update(pr_at_k)
        rows.append(row_data)
        _save_checkpoint(rows)

    results_df = pd.DataFrame(rows)
    print(f"Đã lưu kết quả vào {config.RESULTS_CSV}")
    return results_df


if __name__ == "__main__":
    run_benchmark()
