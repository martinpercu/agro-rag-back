"""Strategy 7: rerank con cross-encoder (servicio dedicado en la Mac mini).

Igual que `rerank` (LLM) pero el ranking lo hace un cross-encoder real
(Qwen3-Reranker-0.6B via mlx-lm, servicio en :8001 de la Mac mini) en vez
de un LLM generativo.

Pasos:
1. Recupera top-N del chroma (semantico, con filtro por intent).
2. POST a /v1/rerank (Jina-compatible) con los N fragmentos.
3. Reordena segun relevance_score; si el servicio falla, cae al orden
   original como fallback (nunca rompe la response).

El servicio corre con 4GB de RAM y ~130ms por request, sin tocar OpenAI.
"""
from __future__ import annotations

import time

from agent.nodes.classifier import _classify
from agent.nodes.retriever import INTENT_TO_SECTION
from agent.rerank_client import RerankClient, RerankResult
from agent.strategies.base import RetrievedItem, Strategy, StrategyResult, TraceRecorder
from ingestion.indexer import search


DEFAULT_RETRIEVE_TOP_N = 20
DEFAULT_RERANK_TOP_K = 6


class RerankCEStrategy(Strategy):
    name = "rerank_ce"

    def __init__(
        self,
        top_n: int = DEFAULT_RETRIEVE_TOP_N,
        client: RerankClient | None = None,
    ):
        self.top_n = top_n
        self.client = client or RerankClient()

    def retrieve(
        self,
        question: str,
        history: list[dict] | None = None,
        k: int = DEFAULT_RERANK_TOP_K,
    ) -> StrategyResult:
        tr = TraceRecorder()
        intent = _classify(question)
        allowed = INTENT_TO_SECTION.get(intent)
        filtro = f", filtro={allowed}" if allowed else ", sin filtro de seccion"
        tr.step("classify", f"intent={intent}{filtro}", at_t=time.monotonic())

        # 1) Retrieve top-N from chroma (mismo pipeline que rerank LLM)
        timings: dict = {}
        t_search = time.monotonic()
        if allowed is None:
            chroma_hits = search(question, k=self.top_n, timings=timings)
        else:
            chroma_hits = search(
                question,
                k=self.top_n,
                where={"seccion": {"$in": allowed}},
                timings=timings,
            )
        candidates = [
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
            for rank, (chunk, score) in enumerate(chroma_hits)
        ]
        embed_ms = timings.get("embed_ms", 0) / 1000
        chroma_ms = timings.get("chroma_ms", 0) / 1000
        tr.step(
            "embed_query",
            f"model={timings.get('embed_model', '?')}, dims={timings.get('embed_dims', '?')}",
            at_t=t_search + embed_ms,
        )
        tr.step(
            "chroma_search",
            f"top-{self.top_n}, hits={len(chroma_hits)}",
            at_t=t_search + embed_ms + chroma_ms,
        )

        if not candidates:
            return StrategyResult(
                name=self.name,
                items=[],
                intent=intent,
                retrieval_ms=tr.steps[-1].acc_ms if tr.steps else 0.0,
                trace=tr.steps,
            )

        # 2) Cross-encoder rerank
        texts = [c.text for c in candidates]
        rr: RerankResult | None = None
        error: str | None = None
        try:
            rr = self.client.rerank(question, texts, top_n=k)
        except Exception as e:  # noqa: BLE001 - fallback al orden original
            error = f"rerank_service_failed: {e}"

        # 3) Build final ranking
        if rr is not None and rr.indices:
            index_to_item = {i: c for i, c in enumerate(candidates)}
            score_by_index = dict(zip(rr.indices, rr.scores))
            final_items = [index_to_item[i] for i in rr.indices if i in index_to_item]
            seen_ids = {i.chunk_id for i in final_items}
            for c in candidates:
                if c.chunk_id not in seen_ids and len(final_items) < k:
                    final_items.append(c)
                    seen_ids.add(c.chunk_id)
            fallback = None
        else:
            final_items = list(candidates[:k])
            fallback = "service_error" if error else "empty_ranking"
        tr.step(
            "ce_rerank",
            f"k={k}, n={len(candidates)}, model={rr.model if rr else '?'}, "
            f"service_ms={round(rr.service_ms, 1) if rr else '?'}",
            at_t=time.monotonic(),
        )

        # 4) Score final: relevance_score del cross-encoder (o 1/(rank+1))
        for item in final_items:
            if rr is not None and item.chunk_id in {
                c.chunk_id for c in candidates
            }:
                idx = next(i for i, c in enumerate(candidates) if c.chunk_id == item.chunk_id)
                if idx in score_by_index:
                    item.score = score_by_index[idx]
            item.rank = final_items.index(item)

        tr.step(
            "build_ranking",
            f"k={k}, fallback={fallback}",
            at_t=time.monotonic(),
        )
        elapsed_ms = tr.steps[-1].acc_ms if tr.steps else 0.0

        return StrategyResult(
            name=self.name,
            items=final_items,
            intent=intent,
            retrieval_ms=elapsed_ms,
            extra={
                "rerank_model": rr.model if rr else None,
                "rerank_service_ms": round(rr.service_ms, 1) if rr else None,
                "rerank_url": self.client.base_url,
                "retrieved_top_n": self.top_n,
                "rerank_top_k": k,
                "scores": [round(s, 4) for s in (rr.scores if rr else [])],
                "fallback": fallback,
                "error": error,
                "filter_sections": allowed,
            },
            trace=tr.steps,
        )