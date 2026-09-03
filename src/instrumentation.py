"""OTel + Langfuse SDK → Langfuse local only (3003).

Solo se activa si LANGFUSE_HOST está seteado (en .env.local). En Railway (sin esa var)
no hace nada — prod no traza.
"""
from __future__ import annotations

import os

_ENABLED = False


def setup_instrumentation() -> bool:
    """Registra instrumentación OpenAI + LangChain hacia Langfuse.

    Soporta dos vías:
    - Langfuse SDK (observe) — vía LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY (recomendado)
    - OTel OpenInference — vía OTLP http://localhost:3003/api/public/otel
    """
    global _ENABLED
    if _ENABLED:
        return True

    host = os.getenv("LANGFUSE_HOST", "").strip()
    if not host:
        return False

    # 1) Langfuse SDK — verifica keys (no bloquea si faltan)
    try:
        from langfuse import Langfuse

        # Langfuse SDK lee LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY del env automáticamente
        # Solo instanciar para validar que las keys existen
        pk = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        sk = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        if pk and sk:
            # No hacemos flush aquí, solo validar que el SDK puede crearse
            Langfuse()
    except Exception:
        pass

    # 2) OTel base (opcional, para OpenInference)
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()
        try:
            trace.set_tracer_provider(provider)
        except Exception:
            # Ya hay provider (ej. pytest), reutilizar
            provider = trace.get_tracer_provider()  # type: ignore

        # Langfuse OTel: http://localhost:3003/api/public/otel (exporter añade /v1/traces)
        otlp_base = os.getenv("LANGFUSE_OTLP_URL", "").strip() or f"{host.rstrip('/')}/api/public/otel"
        has_keys = bool(os.getenv("LANGFUSE_PUBLIC_KEY", "").strip() and os.getenv("LANGFUSE_SECRET_KEY", "").strip())
        if has_keys:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                import base64

                pk = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
                sk = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
                auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
                # OTLP HTTP exporter espera base sin /v1/traces, él lo añade
                exporter = OTLPSpanExporter(
                    endpoint=otlp_base,
                    headers={"Authorization": f"Basic {auth}"},
                )
                # type: ignore — provider puede ser ProxyTracerProvider en tests
                if hasattr(provider, "add_span_processor"):
                    provider.add_span_processor(BatchSpanProcessor(exporter))  # type: ignore
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

        _ENABLED = True
        return True
    except Exception:
        # Aunque falle OTel, el SDK Langfuse ya está listo para @observe
        _ENABLED = True
        return True


# Auto-setup al importar (no bloquea si no hay env)
setup_instrumentation()
