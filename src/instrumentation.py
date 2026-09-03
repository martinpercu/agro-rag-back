"""Langfuse SDK → Langfuse local only (3003).

Solo se activa si LANGFUSE_HOST está seteado (en .env.local). En Railway (sin esa var)
no hace nada — prod no traza. Usa `from langfuse import observe` en los handlers.
"""
from __future__ import annotations

import os

_ENABLED = False


def setup_instrumentation() -> bool:
    """Valida Langfuse env y prepara flush.

    No hace OTel por defecto para evitar `Connection reset by peer` del exporter
    `http://localhost:3003/api/public/otel/v1/traces`. El trazado real lo hace
    `@observe(name=...)` de `from langfuse import observe` en `api/main.py`.
    """
    global _ENABLED
    if _ENABLED:
        return True

    host = os.getenv("LANGFUSE_HOST", "").strip()
    if not host:
        return False

    # Verificar que las keys existen y el SDK puede instanciarse
    try:
        from langfuse import Langfuse

        pk = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        sk = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        if pk and sk:
            Langfuse()  # valida auth_check internamente
    except Exception:
        pass

    _ENABLED = True
    return True


# Auto-setup al importar (no bloquea si no hay env)
setup_instrumentation()
