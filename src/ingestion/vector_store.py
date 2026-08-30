"""Abstraccion VectorStore: Chroma (local) vs Pinecone (prod).

Switch via env:
  VECTOR_STORE=chroma|pinecone  (default: chroma)
  PINECONE_API_KEY, PINECONE_INDEX, PINECONE_CLOUD (serverless)

Permite ver todos los sistemas independientemente:
  - local + chroma: .env.local con bge-m3 + VECTOR_STORE=chroma
  - local + chroma + OpenAI 768: .env con OPENAI_API_KEY + AGROPOSTA_EMBEDDING_DIMS=768 + VECTOR_STORE=chroma  (ya funciona)
  - prod: VECTOR_STORE=pinecone + PINECONE_* + OPENAI_API_KEY  (Pinecone Serverless 768d)

Para no romper dev, Chroma sigue siendo default. Pinecone es lazy import.
"""
from __future__ import annotations

import os
from typing import Protocol

from schemas import Chunk


class VectorStore(Protocol):
    def index_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int: ...
    def search(
        self,
        query: str | None = None,
        k: int = 6,
        where: dict | None = None,
        query_vector: list[float] | None = None,
        timings: dict | None = None,
    ) -> list[tuple[Chunk, float]]: ...
    def collection_stats(self) -> dict: ...
    def health(self) -> dict: ...


def vector_store_name() -> str:
    return os.getenv("VECTOR_STORE", "chroma").strip().lower() or "chroma"


def is_pinecone() -> bool:
    return vector_store_name() == "pinecone"


def get_vector_store() -> VectorStore:
    name = vector_store_name()
    if name == "pinecone":
        return PineconeStore()
    return ChromaStore()


# ------------------------------------------------------------------
# Chroma (local) — delega a indexer.py existente
# ------------------------------------------------------------------

class ChromaStore:
    def index_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        from ingestion.indexer import index_chunks as _index_chunks

        return _index_chunks(chunks, batch_size=batch_size)

    def search(self, query=None, k=6, where=None, query_vector=None, timings=None):
        from ingestion.indexer import search as _search

        return _search(query=query, k=k, where=where, query_vector=query_vector, timings=timings)

    def collection_stats(self) -> dict:
        from ingestion.indexer import collection_stats as _stats

        return _stats()

    def health(self) -> dict:
        from agent.llm import collection_name, embedding_dimensions, embedding_model

        return {
            "store": "chroma",
            "collection": collection_name(),
            "embedding_model": embedding_model(),
            "embedding_dims": embedding_dimensions() or "native",
        }


# ------------------------------------------------------------------
# Pinecone (prod) — lazy, requiere pinecone>=5 y envs
# ------------------------------------------------------------------

class PineconeStore:
    def _require(self):
        api_key = os.getenv("PINECONE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY no configurada (VECTOR_STORE=pinecone)")
        try:
            from pinecone import Pinecone  # type: ignore
        except ImportError as e:
            raise RuntimeError("pinecone package no instalado: uv add pinecone>=5.0") from e
        return Pinecone, api_key

    def _index(self):
        from pinecone import Pinecone  # type: ignore

        api_key = os.getenv("PINECONE_API_KEY", "").strip()
        index_name = os.getenv("PINECONE_INDEX", "margenes-agropecuarios").strip()
        pc = Pinecone(api_key=api_key)
        return pc.Index(index_name)

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        from agent.llm import get_embeddings

        Pinecone, _ = self._require()
        index = self._index()
        embeddings_fn = get_embeddings()
        # Pinecone upsert en batches
        n = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embeddings_fn.embed_documents([c.text for c in batch])
            to_upsert = []
            for c, v in zip(batch, vectors):
                meta = {k: v for k, v in c.metadata.model_dump(mode="json").items() if v is not None and v != ""}
                # Pinecone metadata debe ser flat, strings/numbers
                to_upsert.append({"id": c.id, "values": v, "metadata": {**meta, "text": c.text[:1000]}})
            index.upsert(vectors=to_upsert)
            n += len(batch)
        return n

    def search(self, query=None, k=6, where=None, query_vector=None, timings=None):
        import time

        from agent.llm import get_embeddings

        if (query is None) == (query_vector is None):
            raise ValueError("search: pasar exactamente uno de query o query_vector")
        embeddings_fn = get_embeddings()
        embed_ms = 0.0
        if query_vector is None:
            t = time.time()
            query_vector = embeddings_fn.embed_query(query)  # type: ignore
            embed_ms = (time.time() - t) * 1000
        # Pinecone filter: {"seccion": {"$in": [...]}} ya es compatible
        t2 = time.time()
        index = self._index()
        # Pinecone query
        res = index.query(vector=query_vector, top_k=k, filter=where, include_metadata=True)
        pinecone_ms = (time.time() - t2) * 1000
        if timings is not None:
            timings.update(
                {
                    "embed_ms": embed_ms,
                    "chroma_ms": pinecone_ms,  # compat key, realmente pinecone_ms
                    "pinecone_ms": pinecone_ms,
                    "embed_model": embeddings_fn.model,
                    "embed_dims": len(query_vector) if query_vector else None,
                }
            )
        out: list[tuple[Chunk, float]] = []
        for m in res.matches:  # type: ignore
            meta = m.metadata or {}
            text = meta.pop("text", "")
            # reconstruir Chunk
            from schemas import Chunk as C
            from schemas import ChunkMetadata

            # metadata ya viene con seccion/pagina etc.
            try:
                md = ChunkMetadata(**{k: meta.get(k) for k in ChunkMetadata.model_fields})
            except Exception:
                md = meta
                # fallback: usar dict
                out.append((C(id=m.id, text=text, metadata=md), float(m.score)))  # type: ignore
                continue
            out.append((C(id=m.id, text=text, metadata=md), float(m.score)))
        return out

    def collection_stats(self) -> dict:
        index = self._index()
        stats = index.describe_index_stats()
        # Pinecone stats no da breakdown por seccion; devolvemos total
        total = stats.total_vector_count  # type: ignore
        return {"total": total, "by_section": {}, "by_tipo": {}, "by_cultivo": {}, "store": "pinecone"}

    def health(self) -> dict:
        from agent.llm import collection_name, embedding_dimensions, embedding_model

        return {
            "store": "pinecone",
            "index": os.getenv("PINECONE_INDEX", "margenes-agropecuarios"),
            "collection": collection_name(),
            "embedding_model": embedding_model(),
            "embedding_dims": embedding_dimensions() or "native",
        }
