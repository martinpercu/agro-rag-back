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
from agent.strategies.base import (
    RetrievedItem,
    Strategy,
    StrategyResult,
    TraceRecorder,
)
from ingestion.indexer import get_collection, search


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_RRF_K = 60
_DEFAULT_BM25_TOP_K = 20
_DEFAULT_CHROMA_TOP_K = 20


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


# Cache: {key: (count, bm25, ids, metas, docs)} con key = coleccion + count
_bm25_cache: dict[str, tuple] = {}


def _get_bm25_index():
    """Devuelve (bm25, ids, metas, docs) y reconstruye si la coleccion cambio."""
    col = get_collection()
    current_count = col.count()
    col_name = col.name
    key = f"{col_name}:{current_count}"
    if _bm25_cache.get(key) is None:
        data = col.get(include=["documents", "metadatas"])
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        ids = data.get("ids") or []
        tokenized_corpus = [_tokenize(d) for d in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        _bm25_cache[key] = (bm25, ids, metas, docs)
    return _bm25_cache[key]


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

    def __init__(self, bm25_top_k: int = _DEFAULT_BM25_TOP_K, chroma_top_k: int = _DEFAULT_CHROMA_TOP_K):
        self.bm25_top_k = bm25_top_k
        self.chroma_top_k = chroma_top_k

    def retrieve(
        self,
        question: str,
        history: list[dict] | None = None,
        k: int = 6,
    ) -> StrategyResult:
        tr = TraceRecorder()

        intent = _classify(question)
        allowed = INTENT_TO_SECTION.get(intent)
        filtro = f", filtro={allowed}" if allowed else ", sin filtro de seccion"
        tr.step("classify", f"intent={intent}{filtro}", at_t=time.monotonic())

        # 1) BM25 top-N
        bm25_hits = _bm25_search(question, k=self.bm25_top_k, allowed_sections=allowed)
        tr.step(
            "bm25_search",
            f"top-{self.bm25_top_k}, hits={len(bm25_hits)}",
            at_t=time.monotonic(),
        )
        _, ids, metas, docs = _get_bm25_index()
        bm25_ranking = [ids[i] for i, _ in bm25_hits]

        # 2) ChromaDB top-N
        timings: dict = {}
        t_search = time.monotonic()
        if allowed is None:
            chroma_hits = search(question, k=self.chroma_top_k, timings=timings)
        else:
            chroma_hits = search(
                question,
                k=self.chroma_top_k,
                where={"seccion": {"$in": allowed}},
                timings=timings,
            )
        chroma_ranking = [chunk.id for chunk, _ in chroma_hits]
        embed_ms = timings.get("embed_ms", 0) / 1000
        chroma_ms = timings.get("chroma_ms", 0) / 1000
        tr.step(
            "embed_query",
            f"model={timings.get('embed_model', '?')}, dims={timings.get('embed_dims', '?')}",
            at_t=t_search + embed_ms,
        )
        tr.step(
            "chroma_search",
            f"top-{self.chroma_top_k}, hits={len(chroma_hits)}",
            at_t=t_search + embed_ms + chroma_ms,
        )

        # 3) RRF merge
        merged = _rrf_merge([bm25_ranking, chroma_ranking], k=_RRF_K)
        tr.step("rrf_merge", f"k={_RRF_K}, rankings=2", at_t=time.monotonic())

        elapsed_ms = tr.steps[-1].acc_ms if tr.steps else 0.0

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
                "bm25_top_n": self.bm25_top_k,
                "chroma_top_n": self.chroma_top_k,
                "bm25_returned": len(bm25_hits),
                "chroma_returned": len(chroma_hits),
                "overlap": overlap,
                "filter_sections": allowed,
            },
            trace=tr.steps,
        )
