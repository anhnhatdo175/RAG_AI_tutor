"""
rag_answer(): retrieve top-K support -> ghép prompt -> model sinh câu trả lời
baseline_answer(): gọi thẳng model, không có ngữ cảnh
"""
from src import config
from src.llm_client import generate_answer
from src.retriever import retrieve


RAG_PROMPT_TEMPLATE = """Bạn là một AI Tutor. Dựa vào các đoạn tài liệu tham khảo dưới đây, \
hãy trả lời câu hỏi một cách ngắn gọn và chính xác. Nếu tài liệu không đủ thông tin, \
hãy nói rõ là không có đủ dữ liệu để trả lời, không được bịa thông tin.

Tài liệu tham khảo:
{context}

Câu hỏi: {question}

Trả lời:"""

BASELINE_PROMPT_TEMPLATE = """Trả lời ngắn gọn và chính xác câu hỏi sau:

Câu hỏi: {question}

Trả lời:"""


def rag_answer(question: str, index, docs, k: int = config.TOP_K_FOR_ANSWER) -> dict:
    retrieved = retrieve(question, index, docs, k=k)
    context = "\n\n".join(f"- {r['text']}" for r in retrieved)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    answer, latency = generate_answer(prompt)
    return {
        "answer": answer,
        "latency": latency,
        "retrieved_doc_ids": [r["doc_id"] for r in retrieved],
        "retrieved_texts": [r["text"] for r in retrieved],
        "retrieved_scores": [r["score"] for r in retrieved],
    }


def baseline_answer(question: str) -> dict:
    prompt = BASELINE_PROMPT_TEMPLATE.format(question=question)
    answer, latency = generate_answer(prompt)
    return {"answer": answer, "latency": latency}
