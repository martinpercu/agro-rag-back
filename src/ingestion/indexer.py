"""Indexa chunks en ChromaDB con embeddings de OpenAI."""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from schemas import Chunk

load_dotenv()

COLLECTION_NAME = "margenes_agropecuarios"


def get_chroma_path(base: Path | None = None) -> Path:
    """Path por defecto del vector store."""
    base = base or Path("data/vector")
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_client(base: Path | None = None) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=str(get_chroma_path(base)),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(base: Path | None = None) -> chromadb.Collection:
    client = get_client(base)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def get_embeddings() -> OpenAIEmbeddings:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY no encontrada. Defina la variable de entorno o cree un .env en la raiz del proyecto."
        )
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _sanitize_metadata(meta: dict) -> dict:
    """ChromaDB no acepta None ni strings vacios en metadata."""
    return {k: v for k, v in meta.items() if v is not None and v != ""}


def index_chunks(
    chunks: list[Chunk],
    base: Path | None = None,
    batch_size: int = 64,
) -> int:
    """Embebe e indexa los chunks. Devuelve la cantidad indexada.

    Si el chunk ya existe (mismo id), lo saltea. Asi correr dos veces
    el script no duplica.
    """
    if not chunks:
        return 0

    embeddings_fn = get_embeddings()
    collection = get_collection(base)

    existing = set(collection.get(ids=[c.id for c in chunks]).get("ids", []))
    to_add = [c for c in chunks if c.id not in existing]
    if not to_add:
        return 0

    n = 0
    for start in range(0, len(to_add), batch_size):
        batch = to_add[start:start + batch_size]
        vectors = embeddings_fn.embed_documents([c.text for c in batch])
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=vectors,
            metadatas=[_sanitize_metadata(c.metadata.model_dump(mode="json")) for c in batch],
        )
        n += len(batch)
    return n


def search(
    query: str | None = None,
    k: int = 6,
    where: dict | None = None,
    base: Path | None = None,
    query_vector: list[float] | None = None,
) -> list[tuple[Chunk, float]]:
    """Devuelve los k chunks mas similares, con score.

    Acepta una `query` (string) o un `query_vector` (lista de floats)
    pre-computado. Si se pasan ambos o ninguno, error.
    """
    if (query is None) == (query_vector is None):
        raise ValueError("search: pasar exactamente uno de `query` o `query_vector`")
    embeddings_fn = get_embeddings()
    collection = get_collection(base)
    if query_vector is None:
        query_vector = embeddings_fn.embed_query(query)  # type: ignore[arg-type]
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        where=where,
    )
    out: list[tuple[Chunk, float]] = []
    for i, cid in enumerate(result["ids"][0]):
        meta = result["metadatas"][0][i]
        text = result["documents"][0][i]
        distance = result["distances"][0][i]
        # distancia coseno -> similitud coseno
        score = 1.0 - distance
        out.append((Chunk(id=cid, text=text, metadata=meta), score))
    return out


def collection_stats(base: Path | None = None) -> dict:
    """Metricas basicas del vector store."""
    collection = get_collection(base)
    count = collection.count()
    metadatas = collection.get(include=["metadatas"]).get("metadatas", [])
    by_section: dict[str, int] = {}
    by_tipo: dict[str, int] = {}
    by_cultivo: dict[str, int] = {}
    for m in metadatas:
        by_section[m.get("seccion", "?")] = by_section.get(m.get("seccion", "?"), 0) + 1
        by_tipo[m.get("tipo", "?")] = by_tipo.get(m.get("tipo", "?"), 0) + 1
        c = m.get("cultivo")
        if c:
            by_cultivo[c] = by_cultivo.get(c, 0) + 1
    return {
        "total": count,
        "by_section": by_section,
        "by_tipo": by_tipo,
        "by_cultivo": by_cultivo,
    }
