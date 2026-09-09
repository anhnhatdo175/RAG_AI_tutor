"""
Embed toàn bộ corpus (đoạn support) bằng model local, lưu vào FAISS index.
Cung cấp hàm retrieve(query, k) trả về top-K doc_id kèm điểm tương đồng.
"""
import json
import numpy as np
import faiss
from src import config
from src.llm_client import embed_texts


def load_corpus() -> list[dict]:
    docs = []
    with open(config.CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def build_index(docs: list[dict]):
    texts = [d["text"] for d in docs]
    if config.EMBEDDINGS_PARTIAL_PATH.exists():
        vectors = np.load(config.EMBEDDINGS_PARTIAL_PATH).tolist()
    else:
        vectors = []

    for start in range(len(vectors), len(texts), config.EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + config.EMBEDDING_BATCH_SIZE]
        vectors.extend(embed_texts(batch, batch_size=len(batch)))
        np.save(
            config.EMBEDDINGS_PARTIAL_PATH,
            np.asarray(vectors, dtype="float32"),
        )

    vectors = np.array(vectors, dtype="float32")
    faiss.normalize_L2(vectors)  # để dùng cosine similarity qua inner product

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    np.save(config.EMBEDDINGS_PATH, vectors)
    with open(config.INDEX_META_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    config.EMBEDDINGS_PARTIAL_PATH.unlink(missing_ok=True)
    return index, docs


def load_index():
    vectors = np.load(config.EMBEDDINGS_PATH)
    docs = []
    with open(config.INDEX_META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, docs


def retrieve(query: str, index, docs: list[dict], k: int = 10) -> list[dict]:
    q_vec = np.array(embed_texts([query]), dtype="float32")
    faiss.normalize_L2(q_vec)
    scores, ids = index.search(q_vec, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({**docs[idx], "score": float(score)})
    return results


if __name__ == "__main__":
    docs = load_corpus()
    print(f"Đang embed {len(docs)} đoạn support, có thể mất vài phút...")
    build_index(docs)
    print("Đã lưu index vào thư mục output/")
