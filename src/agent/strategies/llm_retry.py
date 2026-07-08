"""Helper de retry con backoff para llamadas LLM de OpenAI.

Las strategies corren en paralelo con asyncio.gather, lo que hace que
multiples LLM calls peguen a la API al mismo tiempo. OpenAI tiene un
limite de 200K tokens por minuto (TPM) en el tier default. Si nos
pasamos, las llamadas fallan con 429 RateLimitError. Este helper
reintenta con backoff leyendo el delay sugerido por la API misma.
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable, TypeVar

from openai import APIError, RateLimitError

T = TypeVar("T")

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0

# Match: "Please try again in 173ms" o "in 1.5s"
_RETRY_AFTER_MS = re.compile(r"try again in (\d+(?:\.\d+)?)(ms|s)\b", re.IGNORECASE)


def _parse_retry_after(error_msg: str) -> float | None:
    """Extrae el delay sugerido del mensaje de OpenAI, en segundos."""
    m = _RETRY_AFTER_MS.search(error_msg)
    if not m:
        return None
    value = float(m.group(1))
    if m.group(2).lower() == "ms":
        return value / 1000.0
    return value


def call_with_retry(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    **kwargs: Any,
) -> T:
    """Ejecuta fn() con retry exponencial ante RateLimitError.

    Si OpenAI devuelve un delay sugerido (en el mensaje "try again in Xms"),
    lo respeta. Si no, usa backoff exponencial: base_delay * 2^attempt.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            last_error = e
            if attempt == max_retries:
                break
            # Respetar el delay sugerido por la API si lo da
            delay = _parse_retry_after(str(e))
            if delay is None:
                delay = base_delay * (2 ** attempt)
            # Agregar un pequeño jitter para no thunder-herd
            delay += 0.1
            time.sleep(delay)
        except APIError as e:
            # 5xx u otros errores de la API: reintentar tambien
            last_error = e
            if attempt == max_retries or (e.status_code and e.status_code < 500):
                break
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
    raise last_error  # type: ignore[misc]
