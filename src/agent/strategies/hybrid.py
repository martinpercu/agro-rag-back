"""Strategy 2: hybrid (BM25 + semantico, merge con RRF).

Combina dos retrievers:
1. BM25 lexico (rank_bm25) sobre el texto completo de los chunks.
2. Busqueda semantica en ChromaDB (mismo embedding que el baseline).

Los dos rankings se fusionan con Reciprocal Rank Fusion (RRF). El filtro
de secciones del clasificador se aplica a ambos retrievers antes del
merge, asi las dos "ramas" apuntan a la misma region del vector store.

El indice BM25 se construye una vez sobre todos los chunks del Chroma
y se cachea en memoria. Si la cantidad de chunks cambia (reingesta), se
reconstruye automaticamente.
"""
from __future__ import annotations

import re
import time
import unicodedata

from rank_bm25 import BM25Okapi

from agent.nodes.classifier import _classify
from agent.nodes.retriever import INTENT_TO_SECTION
from agent.strategies.base import RetrievedItem, Strategy, StrategyResult
from ingestion.indexer import get_collection, search


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_RRF_K = 60
_BM25_TOP_K = 20
_CHROMA_TOP_K = 20


def _tokenize(text: str) -> list[str]:
    """Tokenizacion simple para espanol: lowercase + remover acentos + split.

    Removemos acentos (NFD + drop combining marks) para que "soja" matchee
    "SOJA" y "costos" matchee "COSTOS Y MARGENES".
    """
    if not text:
        return []
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return _TOKEN_RE.findall(text)


# Cache: {count: int, bm25: BM25Okapi, ids: list[str], metas: list[dict], docs: list[str]}
_bm25_cache: dict = {}


def _get_bm25_index():
    """Devuelve (bm25, ids, metas, docs) y reconstruye si la coleccion cambio."""
    col = get_collection()
    current_count = col.count()
    if _bm25_cache.get("count") != current_count or _bm25_cache.get("bm25") is None:
        data = col.get(include=["documents", "metadatas"])
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        ids = data.get("ids") or []
        tokenized_corpus = [_tokenize(d) for d in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        _bm25_cache["count"] = current_count
        _bm25_cache["bm25"] = bm25
        _bm25_cache["ids"] = ids
        _bm25_cache["metas"] = metas
        _bm25_cache["docs"] = docs
    return _bm25_cache["bm25"], _bm25_cache["ids"], _bm25_cache["metas"], _bm25_cache["docs"]


def _bm25_search(query: str, k: int, allowed_sections: list[str] | None) -> list[tuple[int, float]]:
    """Top-k de BM25. Si allowed_sections, filtra por seccion."""
    bm25, _, metas, _ = _get_bm25_index()
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    indexed = list(enumerate(scores))
    if allowed_sections:
        indexed = [
            (i, s) for i, s in indexed
            if metas[i].get("seccion") in allowed_sections
        ]
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [(i, float(s)) for i, s in indexed[:k] if s > 0]


def _rrf_merge(
    rankings: list[list[str]],
    k: int = _RRF_K,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: cada ranking contribuye 1/(k+rank+1)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridStrategy(Strategy):
    name = "hybrid"

    def retrieve(
        self,
        question: str,
        history: list[dict] | None = None,
        k: int = 6,
    ) -> StrategyResult:
        intent = _classify(question)
        allowed = INTENT_TO_SECTION.get(intent)

        t0 = time.time()

        # 1) BM25 top-N
        bm25_hits = _bm25_search(question, k=_BM25_TOP_K, allowed_sections=allowed)
        _, ids, metas, docs = _get_bm25_index()
        bm25_ranking = [ids[i] for i, _ in bm25_hits]

        # 2) ChromaDB top-N
        if allowed is None:
            chroma_hits = search(question, k=_CHROMA_TOP_K)
        else:
            chroma_hits = search(
                question,
                k=_CHROMA_TOP_K,
                where={"seccion": {"$in": allowed}},
            )
        chroma_ranking = [chunk.id for chunk, _ in chroma_hits]

        # 3) RRF merge
        merged = _rrf_merge([bm25_ranking, chroma_ranking], k=_RRF_K)

        elapsed_ms = (time.time() - t0) * 1000

        # 4) Top-k final
        items = []
        for rank, (cid, rrf_score) in enumerate(merged[:k]):
            try:
                idx = ids.index(cid)
            except ValueError:
                continue
            meta = metas[idx]
            items.append(
                RetrievedItem(
                    chunk_id=cid,
                    text=docs[idx],
                    seccion=meta.get("seccion"),
                    pagina=meta.get("pagina"),
                    cultivo=meta.get("cultivo"),
                    campana=meta.get("campana"),
                    tipo=meta.get("tipo"),
                    score=float(rrf_score),
                    rank=rank,
                )
            )

        overlap = len(set(bm25_ranking) & set(chroma_ranking))

        return StrategyResult(
            name=self.name,
            items=items,
            intent=intent,
            retrieval_ms=elapsed_ms,
            extra={
                "rrf_k": _RRF_K,
                "bm25_top_n": _BM25_TOP_K,
                "chroma_top_n": _CHROMA_TOP_K,
                "bm25_returned": len(bm25_hits),
                "chroma_returned": len(chroma_hits),
                "overlap": overlap,
                "filter_sections": allowed,
            },
        )
