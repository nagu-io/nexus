"""
Provider logging and retry helpers adapted from AEON ai_core.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Any, Awaitable, Callable, TypeVar


T = TypeVar("T")

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGGER = logging.getLogger("nexus.providers")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_DIR / "token_usage.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)


def approximate_tokens(text: str) -> int:
    """Estimate token count cheaply using the AEON 4-char heuristic."""
    return max(1, len(text) // 4)


def log_token_usage(
    provider: str,
    model: str,
    prompt: str,
    response_text: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """Persist token usage telemetry for local/cloud providers."""
    payload = {
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens if input_tokens is not None else approximate_tokens(prompt),
        "output_tokens": output_tokens if output_tokens is not None else approximate_tokens(response_text),
        "timestamp": int(time.time()),
    }
    LOGGER.info(json.dumps(payload, ensure_ascii=True))


async def retry_async(
    provider: str,
    operation: Callable[[], Awaitable[T]],
    attempts: int = 3,
    base_delay: float = 1.5,
) -> T:
    """Retry an async provider operation with linear backoff."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as error:  # pragma: no cover - defensive retry wrapper
            last_error = error
            if attempt == attempts:
                break
            delay = base_delay * attempt
            LOGGER.warning("%s attempt %s failed: %s", provider, attempt, error)
            await asyncio.sleep(delay)
    assert last_error is not None
    raise last_error


def preferred_cloud_provider(config: Any) -> str | None:
    """Return the currently preferred cloud provider for fallback calls."""
    if getattr(config, "openrouter_api_key", ""):
        return "openrouter"
    if getattr(config, "anthropic_api_key", ""):
        return "anthropic"
    return None


def preferred_cloud_model(config: Any) -> str:
    """Return the active cloud model label for status surfaces."""
    provider = preferred_cloud_provider(config)
    if provider == "openrouter":
        return getattr(config, "openrouter_model", "openrouter/auto") or "openrouter/auto"
    return "claude-sonnet-4-20250514"


def configured_local_backend(config: Any) -> str:
    """Return the active local runtime backend."""
    backend = str(getattr(config, "local_model_backend", "ollama") or "ollama").strip().lower()
    return backend if backend in {"ollama", "adapter"} else "ollama"


def active_local_model_label(config: Any) -> str:
    """Return the user-facing label for the current local model runtime."""
    if configured_local_backend(config) == "adapter":
        model_dir = Path(str(getattr(config, "local_model_dir", "lora_model") or "lora_model"))
        return f"nexus-local ({model_dir.name})"
    return str(getattr(config, "nexus_model", "phi3:mini") or "phi3:mini")


def extract_message_text(content: Any) -> str:
    """Normalize provider content blocks into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        text = content.get("text")
        if text:
            return str(text)
    return str(content or "")


async def call_openrouter_chat(
    *,
    api_key: str,
    model: str,
    prompt: str,
    system: str | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
    app_name: str = "NEXUS",
) -> tuple[str, dict[str, Any]]:
    """Call the OpenRouter chat completions API."""
    import httpx

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured.")

    async def operation() -> tuple[str, dict[str, Any]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-Title": app_name,
                },
                json={
                    "model": model or "openrouter/auto",
                    "messages": messages,
                },
            )
            response.raise_for_status()
            payload = response.json()

        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("OpenRouter returned no choices.")
        message = choices[0].get("message", {})
        text = extract_message_text(message.get("content"))
        usage = payload.get("usage") or {}
        return text or "No response", usage

    return await retry_async("openrouter", operation)


class _AdapterRuntime:
    """Lazy singleton wrapper for the local NEXUS adapter runtime."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded_dir: str | None = None
        self._load_lock = Lock()
        self._generate_lock = Lock()

    def _ensure_loaded(self, model_dir: str) -> None:
        resolved_dir = str(Path(model_dir).expanduser().resolve())
        if self._model is not None and self._tokenizer is not None and self._loaded_dir == resolved_dir:
            return

        with self._load_lock:
            if self._model is not None and self._tokenizer is not None and self._loaded_dir == resolved_dir:
                return

            import json
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            config_path = Path(resolved_dir) / "adapter_config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Adapter config not found in {resolved_dir}")
            
            with open(config_path) as f:
                adapter_cfg = json.load(f)
            base_model = adapter_cfg["base_model_name_or_path"]

            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            model = PeftModel.from_pretrained(base, resolved_dir)
            tokenizer = AutoTokenizer.from_pretrained(resolved_dir)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            self._model = model
            self._tokenizer = tokenizer
            self._loaded_dir = resolved_dir

    def generate(self, *, model_dir: str, prompt: str, system: str | None = None) -> str:
        self._ensure_loaded(model_dir)
        assert self._model is not None
        assert self._tokenizer is not None

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        rendered_prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        with self._generate_lock:
            import torch

            inputs = self._tokenizer(rendered_prompt, return_tensors="pt").to(self._model.device)
            prompt_length = int(inputs["input_ids"].shape[-1])

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=3000,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

        generated_tokens = outputs[0][prompt_length:]
        text = self._tokenizer.decode(generated_tokens, skip_special_tokens=False)
        return (
            text.replace("<|assistant|>", "")
            .replace("<|end|>", "")
            .replace("</s>", "")
            .strip()
        )


_ADAPTER_RUNTIME = _AdapterRuntime()


async def call_local_chat(
    *,
    config: Any,
    prompt: str,
    system: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Call the active local runtime, either Ollama or the NEXUS adapter."""
    backend = configured_local_backend(config)
    model_label = active_local_model_label(config)

    if backend == "adapter":
        text = await asyncio.to_thread(
            _ADAPTER_RUNTIME.generate,
            model_dir=str(getattr(config, "local_model_dir", "lora_model") or "lora_model"),
            prompt=prompt,
            system=system,
        )
        return (
            text or "No response from local adapter",
            {
                "provider": "nexus_adapter",
                "model": model_label,
                "prompt_tokens": approximate_tokens((system or "") + prompt),
                "completion_tokens": approximate_tokens(text or ""),
            },
        )

    import httpx

    async def operation() -> tuple[str, dict[str, Any]]:
        request_prompt = prompt
        if system:
            request_prompt = f"System: {system}\n\nUser: {prompt}\n\nAssistant:"
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{config.ollama_base_url}/api/generate",
                json={"model": config.nexus_model, "prompt": request_prompt, "stream": False},
            )
            response.raise_for_status()
            payload = response.json()
        text = payload.get("response", "No response")
        return (
            text,
            {
                "provider": "ollama",
                "model": config.nexus_model,
                "prompt_tokens": payload.get("prompt_eval_count"),
                "completion_tokens": payload.get("eval_count"),
            },
        )

    return await retry_async("ollama", operation)
