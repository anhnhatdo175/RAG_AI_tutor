"""Load SciQ splits, build the retrieval corpus, and sample test questions."""
import json
import pandas as pd
from src import config


def load_split(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "support" not in df.columns:
        df = pd.read_csv(path, sep="\t")
    return df


def load_all() -> pd.DataFrame:
    train = load_split(config.TRAIN_CSV)
    test = load_split(config.TEST_CSV)
    valid = load_split(config.VALID_CSV)
    for name, d in [("train", train), ("test", test), ("validation", valid)]:
        d["split"] = name
    full = pd.concat([train, test, valid], ignore_index=True)
    return full


def build_corpus(df: pd.DataFrame) -> list[dict]:
    """Create one deduplicated document for each non-empty support passage."""
    df = df.dropna(subset=["support"])
    df = df[df["support"].str.strip() != ""]

    unique_supports = df["support"].drop_duplicates().reset_index(drop=True)
    corpus = []
    for i, text in unique_supports.items():
        corpus.append({"doc_id": f"doc_{i}", "text": text})
    return corpus


def build_support_to_docid(corpus: list[dict]) -> dict:
    return {doc["text"]: doc["doc_id"] for doc in corpus}


def save_corpus(corpus: list[dict]):
    with open(config.CORPUS_PATH, "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def sample_test_questions(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Lấy mẫu câu hỏi test từ split 'test', chỉ giữ câu có support để có ground truth."""
    test_df = df[(df["split"] == "test") & (df["support"].notna())]
    test_df = test_df[test_df["support"].str.strip() != ""]
    n = min(n, len(test_df))
    return test_df.sample(n=n, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    full_df = load_all()
    corpus = build_corpus(full_df)
    save_corpus(corpus)
    print(f"Tổng số dòng dữ liệu: {len(full_df)}")
    print(f"Số đoạn support duy nhất (corpus size): {len(corpus)}")
