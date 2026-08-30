"""Indexa chunks en ChromaDB con embeddings del provider configurado.

El provider (OpenAI por default, o un servidor OpenAI-compatible como
LM Studio) y el modelo de embeddings se configuran via env vars en
agent.llm. El nombre de la coleccion se deriva del modelo de embeddings
para poder tener varios vector stores (por ejemplo uno por modelo en el
bakeoff) sin mezclar dimensiones.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from agent.llm import collection_name, get_embeddings
from schemas import Chunk

load_dotenv()

# Singleton de clientes por path + lock de operaciones: las strategies
# corren en paralelo (asyncio.to_thread) y ChromaDB no es thread-safe.
# Crear un PersistentClient por llamada en paralelo produce errores
# de "tenant" (carrera en la creacion del sqlite). Un solo cliente
# compartido + lock serializa el acceso al vector store (muy rapido
# comparado con las llamadas LLM).
_clients: dict[str, chromadb.PersistentClient] = {}
_clients_lock = threading.Lock()
_ops_lock = threading.Lock()


def get_chroma_path(base: Path | None = None) -> Path:
    """Path por defecto del vector store."""
    base = base or Path("data/vector")
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_client(base: Path | None = None) -> chromadb.PersistentClient:
    path = str(get_chroma_path(base))
    with _clients_lock:
        client = _clients.get(path)
        if client is None:
            client = chromadb.PersistentClient(
                path=path,
                settings=Settings(anonymized_telemetry=False),
            )
            _clients[path] = client
        return client


def get_collection(base: Path | None = None) -> chromadb.Collection:
    client = get_client(base)
    with _ops_lock:
        return client.get_or_create_collection(
            name=collection_name(),
            metadata={"hnsw:space": "cosine"},
        )


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

    Delega a PineconeStore si VECTOR_STORE=pinecone.
    """
    try:
        from ingestion.vector_store import is_pinecone

        if is_pinecone():
            from ingestion.vector_store import get_vector_store

            return get_vector_store().index_chunks(chunks, batch_size=batch_size)
    except Exception:
        pass
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
    timings: dict | None = None,
) -> list[tuple[Chunk, float]]:
    """Devuelve los k chunks mas similares, con score.

    Acepta una `query` (string) o un `query_vector` (lista de floats)
    pre-computado. Si se pasan ambos o ninguno, error.
    Si `timings` es un dict, lo llena con {"embed_ms", "chroma_ms",
    "embed_model"} para el trace de las strategies.

    Switch VECTOR_STORE: si es pinecone, delega a PineconeStore (mantiene
    compat con strategies que importan desde indexer).
    """
    # Switch vector store — permite ver todos los sistemas independientemente
    # sin tocar las strategies (que importan desde este modulo).
    try:
        from ingestion.vector_store import is_pinecone

        if is_pinecone():
            from ingestion.vector_store import get_vector_store

            return get_vector_store().search(
                query=query, k=k, where=where, query_vector=query_vector, timings=timings
            )
    except Exception:
        # Si pinecone no esta instalado/configurado, fallback a Chroma
        pass
    if (query is None) == (query_vector is None):
        raise ValueError("search: pasar exactamente uno de `query` o `query_vector`")
    embeddings_fn = get_embeddings()
    collection = get_collection(base)
    embed_ms = 0.0
    if query_vector is None:
        t_embed = time.time()
        query_vector = embeddings_fn.embed_query(query)  # type: ignore[arg-type]
        embed_ms = (time.time() - t_embed) * 1000
    with _ops_lock:
        t_chroma = time.time()
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=where,
        )
        chroma_ms = (time.time() - t_chroma) * 1000
    if timings is not None:
        timings.update(
            {
                "embed_ms": embed_ms,
                "chroma_ms": chroma_ms,
                "embed_model": embeddings_fn.model,
                "embed_dims": len(query_vector) if query_vector else None,
            }
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
    """Metricas basicas del vector store (delega a Pinecone si VECTOR_STORE=pinecone)."""
    try:
        from ingestion.vector_store import is_pinecone

        if is_pinecone():
            from ingestion.vector_store import get_vector_store

            return get_vector_store().collection_stats()
    except Exception:
        pass
    collection = get_collection(base)
    with _ops_lock:
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
