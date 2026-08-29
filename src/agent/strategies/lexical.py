"""Strategy 8: lexical puro (solo BM25, sin embeddings ni chroma).

Recupera los chunks unicamente por coincidencia de palabras exactas
(BM25Okapi sobre el corpus tokenizado), sin tocar el vector store.

El `k` que recibe la strategy es exactamente la cantidad de chunks que
devuelve (el mismo `k` del slider del comparador).

Reusa el indice BM25 cacheado de hybrid.py (misma construccion, mismo cache).
"""
from __future__ import annotations

import time

from agent.nodes.classifier import _classify
from agent.nodes.retriever import INTENT_TO_SECTION
from agent.strategies.base import RetrievedItem, Strategy, StrategyResult, TraceRecorder
from agent.strategies.hybrid import _bm25_search, _get_bm25_index


class LexicalStrategy(Strategy):
    name = "lexical"

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

        # 1) BM25 puro: top-k por palabras exactas
        hits = _bm25_search(question, k=k, allowed_sections=allowed)
        tr.step("bm25_search", f"k={k}, hits={len(hits)}", at_t=time.monotonic())

        # 2) Map (corpus_index, score) -> RetrievedItem usando el indice cacheado
        _, ids, metas, docs = _get_bm25_index()
        items: list[RetrievedItem] = []
        for rank, (idx, score) in enumerate(hits):
            meta = metas[idx] if idx < len(metas) else {}
            items.append(
                RetrievedItem(
                    chunk_id=ids[idx],
                    text=docs[idx],
                    seccion=meta.get("seccion"),
                    pagina=meta.get("pagina"),
                    cultivo=meta.get("cultivo"),
                    campana=meta.get("campana"),
                    tipo=meta.get("tipo"),
                    score=score,
                    rank=rank,
                )
            )

        elapsed_ms = tr.steps[-1].acc_ms if tr.steps else 0.0

        return StrategyResult(
            name=self.name,
            items=items,
            intent=intent,
            retrieval_ms=elapsed_ms,
            extra={
                "bm25_top_k": k,
                "filter_sections": allowed,
                "corpus_chunks": len(ids),
                "embedding_used": False,
            },
            trace=tr.steps,
        )