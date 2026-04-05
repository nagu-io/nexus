"""
Provider logging and retry helpers adapted from AEON ai_core.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable, TypeVar


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
