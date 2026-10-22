"""Layer 5 LLM backends owned by the lmsrag package."""
from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib import request

os.environ.setdefault("USE_TF", "0")

from lmsrag import config

_models: dict[str, tuple[Any, Any]] = {}
_groq_client: Any = None
_last_groq_request = 0.0


def _transformers_model(model_name: str) -> tuple[Any, Any]:
    if model_name not in _models:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("Backend transformers requires a CUDA GPU; enable GPU in Colab.")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _models[model_name] = tokenizer, model
    return _models[model_name]


def _generate_transformers(prompt: str, model_name: str) -> str:
    import torch

    tokenizer, model = _transformers_model(model_name)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
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
    answer = tokenizer.decode(
        output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()
    if not answer:
        raise ValueError("Transformers model returned an empty response.")
    return answer


def _generate_ollama(prompt: str, model_name: str, json_mode: bool,
                     json_schema: dict[str, Any] | None) -> str:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_ctx": config.OLLAMA_CONTEXT_LENGTH,
                    "num_predict": config.OLLAMA_MAX_NEW_TOKENS, "temperature": 0},
    }
    if json_mode:
        payload["format"] = json_schema or "json"
    if model_name.casefold().startswith("qwen3"):
        payload["think"] = False
    req = request.Request(
        f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not call Ollama at {config.OLLAMA_BASE_URL}.") from exc
    answer = result.get("message", {}).get("content")
    if not answer:
        raise ValueError("Ollama returned an empty response.")
    return answer.strip()


def _generate_groq(prompt: str, model_name: str) -> str:
    global _groq_client, _last_groq_request
    if not config.GROQ_API_KEY:
        raise ValueError("Set GROQ_API_KEY in the environment or .env file.")
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    for attempt in range(config.GROQ_RETRY_ATTEMPTS):
        delay = config.GROQ_RATE_LIMIT - (time.perf_counter() - _last_groq_request)
        if delay > 0:
            time.sleep(delay)
        _last_groq_request = time.perf_counter()
        try:
            response = _groq_client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content
            if not answer:
                raise ValueError("Groq returned an empty response.")
            return answer.strip()
        except Exception:
            if attempt == config.GROQ_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Groq retry loop ended unexpectedly.")


def generate_answer(prompt: str, model: str | None = None, json_mode: bool = False,
                    json_schema: dict[str, Any] | None = None) -> tuple[str, float]:
    """Generate text with the configured Transformers, Ollama, or Groq backend."""
    started = time.perf_counter()
    model_name = model or config.GENERATION_MODEL
    backend = config.LLM_BACKEND
    if backend == "transformers":
        answer = _generate_transformers(prompt, model_name)
    elif backend == "ollama":
        answer = _generate_ollama(prompt, model_name, json_mode, json_schema)
    elif backend == "groq":
        answer = _generate_groq(prompt, model_name)
    else:
        raise ValueError(f"Unsupported LLM_BACKEND: {backend!r}")
    return answer, time.perf_counter() - started
