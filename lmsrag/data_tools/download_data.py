"""Fetch the real corpora (GSM8K from OpenAI, MATH test split from the Qwen2.5-Math mirror). Khan Academy: add manually."""
import urllib.request
from .. import config

URLS = {
    "gsm8k_train.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "gsm8k_test.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "math_test.jsonl": "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/main/evaluation/data/math/test.jsonl",
}


def download_missing() -> None:
    config.RAW.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        path = config.RAW / name
        if path.is_file() and path.stat().st_size:
            print("using existing", name)
            continue
        urllib.request.urlretrieve(url, path)
        print("downloaded", name)


if __name__ == "__main__":
    download_missing()
