"""Langfuse SDK → Langfuse local only (3003) — deprecated shim.

Solo se activa si LANGFUSE_HOST está seteado (en .env.local). En Railway (sin esa var)
no hace nada — prod no traza. Usa `from observability import get_langfuse` en los handlers.
Este archivo queda por compatibilidad con `import instrumentation` en api/main.py;
la logica nueva vive en `observability.py` (manual SDK 2.80, sin OTel).
"""
from __future__ import annotations

# Re-export from observability (canonical)
try:
    from observability import get_langfuse, is_enabled  # noqa: F401

    _ENABLED = is_enabled()
except Exception:
    _ENABLED = False


def setup_instrumentation() -> bool:
    """No OTel — solo valida que Langfuse esté configurado.

    Evita `Connection reset by peer` del exporter
    `http://localhost:3003/api/public/otel/v1/traces`.
    """
    try:
        from observability import is_enabled as _is_enabled

        return _is_enabled()
    except Exception:
        return False


# Auto-setup al importar (no bloquea si no hay env)
setup_instrumentation()
