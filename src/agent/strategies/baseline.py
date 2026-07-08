"""Strategy 1: baseline.

Wrapper sobre el retriever actual de Agroposta: clasifica el intent, busca
los top-k chunks por similitud coseno en ChromaDB, opcionalmente filtrando
por secciones segun el intent. Es lo que ya tenemos en produccion.
"""
from __future__ import annotations

import time

from agent.nodes.classifier import _classify
from agent.nodes.retriever import INTENT_TO_SECTION
from agent.strategies.base import RetrievedItem, Strategy, StrategyResult
from ingestion.indexer import search


class BaselineStrategy(Strategy):
    name = "baseline"

    def retrieve(
        self,
        question: str,
        history: list[dict] | None = None,
        k: int = 6,
    ) -> StrategyResult:
        intent = _classify(question)
        allowed = INTENT_TO_SECTION.get(intent)

        t0 = time.time()
        if allowed is None:
            hits = search(question, k=k)
        else:
            hits = search(
                question,
                k=k,
                where={"seccion": {"$in": allowed}},
            )
        elapsed_ms = (time.time() - t0) * 1000

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
        )
