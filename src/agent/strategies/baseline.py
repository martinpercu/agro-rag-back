"""Strategy 1: baseline.

Wrapper sobre el retriever actual de Agroposta: clasifica el intent, busca
los top-k chunks por similitud coseno en ChromaDB, opcionalmente filtrando
por secciones segun el intent. Es lo que ya tenemos en produccion.
"""
from __future__ import annotations

import time

from agent.nodes.classifier import _classify
from agent.nodes.retriever import INTENT_TO_SECTION
from agent.strategies.base import (
    RetrievedItem,
    Strategy,
    StrategyResult,
    TraceRecorder,
)
from ingestion.indexer import search


class BaselineStrategy(Strategy):
    name = "baseline"

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

        timings: dict = {}
        t_search = time.monotonic()
        if allowed is None:
            hits = search(question, k=k, timings=timings)
        else:
            hits = search(
                question,
                k=k,
                where={"seccion": {"$in": allowed}},
                timings=timings,
            )
        embed_ms = timings.get("embed_ms", 0) / 1000
        chroma_ms = timings.get("chroma_ms", 0) / 1000
        tr.step(
            "embed_query",
            f"model={timings.get('embed_model', '?')}, dims={timings.get('embed_dims', '?')}",
            at_t=t_search + embed_ms,
        )
        tr.step(
            "chroma_search",
            f"k={k}, hits={len(hits)}",
            at_t=t_search + embed_ms + chroma_ms,
        )
        elapsed_ms = tr.steps[-1].acc_ms if tr.steps else 0.0

        items = [
            RetrievedItem(
                chunk_id=chunk.id,
                text=chunk.text,
                seccion=chunk.metadata.seccion,
                pagina=chunk.metadata.pagina,
                cultivo=chunk.metadata.cultivo,
                campana=chunk.metadata.campana,
                tipo=chunk.metadata.tipo,
                score=float(score),
                rank=rank,
            )
            for rank, (chunk, score) in enumerate(hits)
        ]

        return StrategyResult(
            name=self.name,
            items=items,
            intent=intent,
            retrieval_ms=elapsed_ms,
            extra={
                "chroma_top_k": k,
                "filter_sections": allowed,
            },
            trace=tr.steps,
        )
