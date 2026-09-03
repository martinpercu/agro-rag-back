"""OTel instrumentation → Langfuse local only (3003).

Solo se activa si LANGFUSE_HOST está seteado (en .env.local). En Railway (sin esa var)
no hace nada — prod no traza.
"""
from __future__ import annotations

import os

_ENABLED = False


def setup_instrumentation() -> bool:
    """Registra instrumentación OpenAI + LangChain hacia Langfuse/OTel.

    Returns True si se activó, False si está deshabilitado (sin env).
    """
    global _ENABLED
    if _ENABLED:
        return True

    host = os.getenv("LANGFUSE_HOST", "").strip()
    if not host:
        return False

    # Langfuse envs ya deben estar en .env.local: PUBLIC_KEY / SECRET_KEY
    # Para OpenInference, usamos el endpoint OTLP de Langfuse si existe,
    # sino el SDK de Langfuse directamente.
    try:
        # 1) OpenTelemetry base
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # 2) Langfuse OTLP exporter (si langfuse está disponible)
        # Intentamos con OpenInference si está, sino fallback silencioso
        provider = TracerProvider()
        trace.set_tracer_provider(provider)

        # Langfuse 3 expone OTLP en /api/public/otel/v1/traces — derivar de HOST si no hay OTLP_URL
        otlp_url = os.getenv("LANGFUSE_OTLP_URL", "").strip() or f"{host.rstrip('/')}/api/public/otel/v1/traces"
        # Solo configurar exporter si host está y hay keys (evita ruido en dev sin keys)
        has_keys = bool(os.getenv("LANGFUSE_PUBLIC_KEY", "").strip() and os.getenv("LANGFUSE_SECRET_KEY", "").strip())
        if has_keys:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                import base64

                # Langfuse OTel requiere Basic auth pk+sk
                pk = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
                sk = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
                auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
                exporter = OTLPSpanExporter(
                    endpoint=otlp_url,
                    headers={"Authorization": f"Basic {auth}"},
                )
                provider.add_span_processor(BatchSpanProcessor(exporter))
            except Exception:
                pass

        # 3) Instrumentadores OpenInference (no rompen si no hay exporter)
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor

            LangChainInstrumentor().instrument()
        except Exception:
            pass
        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor

            OpenAIInstrumentor().instrument()
        except Exception:
            pass

        # 4) Langfuse SDK para trazas explícitas (opcional)
        # Se usa via `from langfuse import get_client` en los nodos si se quiere
        _ENABLED = True
        return True
    except Exception:
        return False


# Auto-setup al importar (no bloquea si no hay env)
setup_instrumentation()
