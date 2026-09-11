import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DATA_DIR = Path(os.environ.get("SCIQ_DATA_DIR", r"D:\SciQ"))

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
VALID_CSV = DATA_DIR / "validation.csv"

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = OUT_DIR / "corpus.jsonl"
EMBEDDINGS_PATH = OUT_DIR / "embeddings.npy"
EMBEDDINGS_PARTIAL_PATH = OUT_DIR / "embeddings.partial.npy"
INDEX_META_PATH = OUT_DIR / "index_meta.jsonl"
RESULTS_CSV = OUT_DIR / "benchmark_results.csv"
FAITHFULNESS_JSON = OUT_DIR / "faithfulness_judgments.json"
FAITHFULNESS_CSV = OUT_DIR / "faithfulness_judgments.csv"
FAITHFULNESS_MARKDOWN = OUT_DIR / "faithfulness_report.md"
FAITHFULNESS_CHART = OUT_DIR / "chart_faithfulness.png"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_BACKEND = os.environ.get("LLM_BACKEND", "transformers").lower()
GENERATION_MODEL = os.environ.get(
    "GENERATION_MODEL", "Qwen/Qwen2.5-7B-Instruct"
)
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
TRANSFORMERS_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
TRANSFORMERS_CONTEXT_LENGTH = int(os.environ.get("TRANSFORMERS_CONTEXT_LENGTH", "4096"))

SAMPLE_SIZE = 200

K_VALUES = [1, 3, 5, 10]
TOP_K_FOR_ANSWER = 3

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
RATE_LIMIT_SECONDS = float(os.environ.get("GROQ_RATE_LIMIT", "2.1"))
EMBEDDING_BATCH_SIZE = int(os.environ.get("LOCAL_EMBEDDING_BATCH_SIZE", "64"))
RETRY_ATTEMPTS = int(os.environ.get("GROQ_RETRY_ATTEMPTS", "5"))