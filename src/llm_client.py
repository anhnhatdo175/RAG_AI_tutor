"""LLM backends for local Transformers, Ollama, and Groq inference."""
import re
import time
import json
from urllib import request
from groq import Groq
from sentence_transformers import SentenceTransformer
from src import config

_client = None
_embedding_model = None
_transformers_models = {}
_transformers_tokenizers = {}

_last_request_time = 0.0

def _wait_for_rate_limit():
    """Đảm bảo không gọi API quá 15 request/phút."""
    global _last_request_time
    elapsed = time.perf_counter() - _last_request_time
    if elapsed < config.RATE_LIMIT_SECONDS:
        sleep_time = config.RATE_LIMIT_SECONDS - elapsed
        time.sleep(sleep_time)
    _last_request_time = time.perf_counter()


def _retry_delay(error: Exception, attempt: int) -> float:
    """Honor an API retry hint when available, otherwise use exponential backoff."""
    match = re.search(
        r"(?:retry in|try again in)\s*([0-9]+(?:\.[0-9]+)?)\s*s",
        str(error),
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1)) + 1.0
    return float(2 ** attempt)


def _get_client():
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY chưa được cấu hình trong file .env.")
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def embed_texts(
    texts: list[str],
    batch_size: int = config.EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    """Return normalized local BGE embeddings without using an API quota."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    vectors = _embedding_model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return vectors.tolist()


def _load_transformers_model(model_name: str):
    if model_name in _transformers_models:
        return _transformers_tokenizers[model_name], _transformers_models[model_name]

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Backend transformers cần cài transformers, accelerate, "
            "bitsandbytes và torch."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Backend transformers cần GPU CUDA. Hãy bật GPU runtime trong Colab."
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _transformers_tokenizers[model_name] = tokenizer
    _transformers_models[model_name] = model
    return tokenizer, model


def _generate_transformers(
    prompt: str, model_name: str, start: float
) -> tuple[str, float]:
    import torch

    tokenizer, model = _load_transformers_model(model_name)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        truncation=True,
        max_length=config.TRANSFORMERS_CONTEXT_LENGTH,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=config.TRANSFORMERS_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_tokens = output[0, inputs["input_ids"].shape[-1] :]
    content = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    if not content:
        raise ValueError("Transformers model trả về phản hồi rỗng.")
    return content, time.perf_counter() - start


def generate_answer(
    prompt: str, model: str | None = None, json_mode: bool = False
) -> tuple[str, float]:
    """Generate text with the configured backend."""
    start = time.perf_counter()
    selected_model = model or config.GENERATION_MODEL
    if config.LLM_BACKEND == "transformers":
        return _generate_transformers(prompt, selected_model, start)
    if config.LLM_BACKEND == "ollama":
        payload = json.dumps(
            {
                "model": selected_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_ctx": 4096},
            }
        ).encode("utf-8")
        if json_mode:
            payload = json.dumps(
                {
                    "model": selected_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json",
                    "options": {"num_ctx": 4096},
                }
            ).encode("utf-8")
        http_request = request.Request(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                "Không gọi được Ollama. Hãy cài Ollama, chạy service và "
                f"tải model bằng `ollama pull {selected_model}`."
            ) from exc
        content = result.get("message", {}).get("content")
        if not content:
            raise ValueError("Ollama trả về phản hồi rỗng.")
        return content.strip(), time.perf_counter() - start

    client = _get_client()
    for attempt in range(config.RETRY_ATTEMPTS):
        try:
            _wait_for_rate_limit()
            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
            )
            latency = time.perf_counter() - start
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Groq trả về phản hồi rỗng.")
            return content.strip(), latency
        except Exception as e:
            if attempt == config.RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_retry_delay(e, attempt))