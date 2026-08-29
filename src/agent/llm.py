"""Factory de clientes LLM y embeddings con soporte multi-provider.

El default es OpenAI (como siempre). Con env vars se puede apuntar a
cualquier servidor OpenAI-compatible (LM Studio, Ollama, vLLM, etc.):

    AGROPOSTA_LLM_BASE_URL      base_url del servidor (default: OpenAI oficial)
    AGROPOSTA_LLM_MODEL         modelo de chat (default: gpt-4.1-nano)
    AGROPOSTA_EMBEDDING_MODEL   modelo de embeddings (default: text-embedding-3-small)
    AGROPOSTA_EMBEDDINGS_BASE_URL  base_url del servidor de embeddings (default:
                                   el mismo de AGROPOSTA_LLM_BASE_URL; util para
                                   apuntar a un servicio dedicado, ej: bge-m3)

El nombre de la coleccion se deriva del modelo de embeddings: la default
conserva el nombre historico (para no re-ingestar el vector store OpenAI
existente) y los otros modelos crean colecciones separadas, porque
ChromaDB fija la dimension con el primer embedding que entra y no se
pueden mezclar dimensiones dentro de una misma coleccion.
"""
from __future__ import annotations

import os
import re
from typing import Any

from openai import AsyncOpenAI, OpenAI

DEFAULT_LLM_MODEL = "gpt-4.1-nano"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
BASE_COLLECTION = "margenes_agropecuarios"

LLM_SEED = 42


def embedding_dimensions() -> int | None:
    """Dims Matryoshka para text-embedding-3-* (768/512). None = native.

    Solo aplica a OpenAI text-embedding-3-*. Modelos locales (bge-m3, nomic)
    rechazan el param `dimensions` y deben ignorarlo.
    """
    v = os.getenv("AGROPOSTA_EMBEDDING_DIMS", "").strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def seed_for_temperature(temperature: float | None) -> int | None:
    """Seed fijo para temperatura 0: decoding greedy determinista a nivel GPU.

    Los motores locales (LM Studio/MLX) muestran indeterminismo numerico con
    contextos largos incluso en greedy (Metal resuelve logits empatados distinto
    en cada run). Con seed fijo y temp=0, mismo input -> misma respuesta.
    """
    return LLM_SEED if temperature == 0 else None


class EmbeddingsClient:
    """Adapter minimo sobre el SDK de OpenAI para embeddings.

    Reemplaza a langchain's OpenAIEmbeddings: ese tokeniza el input con
    tiktoken y envia IDs de tokens, que los servidores locales
    OpenAI-compatibles (LM Studio) rechazan (solo aceptan strings).
    """

    def __init__(self, model: str | None = None):
        self.model = model or embedding_model()
        self._client = OpenAI(base_url=embeddings_base_url())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        kwargs: dict[str, Any] = {}
        dims = embedding_dimensions()
        if dims and self.model.startswith("text-embedding-3"):
            kwargs["dimensions"] = dims
        resp = self._client.embeddings.create(model=self.model, input=texts, **kwargs)
        out: list[list[float]] = [d.embedding for d in resp.data]
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def llm_model() -> str:
    return os.getenv("AGROPOSTA_LLM_MODEL", DEFAULT_LLM_MODEL)


def embedding_model() -> str:
    return os.getenv("AGROPOSTA_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def llm_base_url() -> str | None:
    v = os.getenv("AGROPOSTA_LLM_BASE_URL", "").strip()
    return v or None


def embeddings_base_url() -> str | None:
    v = os.getenv("AGROPOSTA_EMBEDDINGS_BASE_URL", "").strip()
    return v or llm_base_url()


def is_default_embedding(model: str | None = None) -> bool:
    return (model or embedding_model()) == DEFAULT_EMBEDDING_MODEL


def get_chat_client() -> OpenAI:
    """Cliente sync de chat (completions). Lee AGROPOSTA_LLM_BASE_URL."""
    return OpenAI(base_url=llm_base_url())


def get_async_chat_client() -> AsyncOpenAI:
    """Cliente async de chat (streaming). Lee AGROPOSTA_LLM_BASE_URL."""
    return AsyncOpenAI(base_url=llm_base_url())


def get_embeddings(model: str | None = None) -> EmbeddingsClient:
    """Embeddings del modelo configurado (o el pasado como argumento)."""
    return EmbeddingsClient(model)


def collection_name(embed_model: str | None = None) -> str:
    """Nombre de coleccion para el modelo de embeddings dado.

    - text-embedding-3-small (default) sin dims: conserva el nombre historico.
    - Con AGROPOSTA_EMBEDDING_DIMS: sufija __d768 (evita colision 1536 vs 768).
    - Cualquier otro modelo: `margenes_agropecuarios__<modelo_saneado>[__d768]`.
    """
    m = embed_model or embedding_model()
    dims = embedding_dimensions()
    if m == DEFAULT_EMBEDDING_MODEL and not dims:
        return BASE_COLLECTION
    if m == DEFAULT_EMBEDDING_MODEL:
        # 3-small con dims → base + sufijo dims (solo Matryoshka)
        base = BASE_COLLECTION
        if dims and dims != 1536:
            base = f"{base}__d{dims}"
    else:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", m).strip("-")
        base = f"{BASE_COLLECTION}__{safe}"
        # Para modelos no-Matryoshka (bge, nomic) no sufijamos dims — native dims ya está en el modelo
    return base