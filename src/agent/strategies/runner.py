"""Runner: ejecuta las 6 strategies de retrieval + answerer en paralelo.

Uso:
    result = await run_compare(question, history, k=6)
    # result es un ComparisonResponse (dict) con 6 StrategyComparisonResult
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from agent.nodes.answerer import answer, stream_answer_async, _format_sources_from_items
from agent.strategies.base import RetrievedItem, Strategy, StrategyResult


# ---------- Pydantic schemas (dataclasses para evitar pydantic) ----------

from dataclasses import dataclass, field


@dataclass
class StrategyComparisonResult:
    """Resultado de UNA strategy: retrieval + answerer + metricas agregadas."""

    name: str
    intent: str | None = None
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    items: list[RetrievedItem] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "intent": self.intent,
            "answer": self.answer,
            "sources": self.sources,
            "items": [item_to_dict(i) for i in self.items],
            "metrics": self.metrics,
        }


@dataclass
class ComparisonResponse:
    """Resultado del comparador: las 6 strategies con sus metricas."""

    question: str
    strategies: dict[str, StrategyComparisonResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "strategies": {n: r.to_dict() for n, r in self.strategies.items()},
        }


def item_to_dict(item: RetrievedItem) -> dict:
    return {
        "chunk_id": item.chunk_id,
        "seccion": item.seccion,
        "pagina": item.pagina,
        "cultivo": item.cultivo,
        "campana": item.campana,
        "tipo": item.tipo,
        "score": round(item.score, 4),
        "rank": item.rank,
    }


# ---------- Registry de las 6 strategies ----------

def get_all_strategies() -> list[Strategy]:
    """Devuelve las 6 strategies (singleton lazy)."""
    global _STRATEGIES_SINGLETON
    if _STRATEGIES_SINGLETON is None:
        from agent.strategies.baseline import BaselineStrategy
        from agent.strategies.hybrid import HybridStrategy
        from agent.strategies.hyde import HydeStrategy
        from agent.strategies.multi_query import MultiQueryStrategy
        from agent.strategies.query_rewrite import QueryRewriteStrategy
        from agent.strategies.rerank import RerankStrategy

        _STRATEGIES_SINGLETON = [
            BaselineStrategy(),
            HybridStrategy(),
            RerankStrategy(),
            QueryRewriteStrategy(),
            MultiQueryStrategy(),
            HydeStrategy(),
        ]
    return _STRATEGIES_SINGLETON


_STRATEGIES_SINGLETON: list[Strategy] | None = None


def get_extra_strategies() -> list[Strategy]:
    """Strategies fuera del comparador default (ej: rerank_ce, lexical)."""
    global _EXTRA_STRATEGIES_SINGLETON
    if _EXTRA_STRATEGIES_SINGLETON is None:
        from agent.strategies.lexical import LexicalStrategy
        from agent.strategies.rerank_ce import RerankCEStrategy

        _EXTRA_STRATEGIES_SINGLETON = [RerankCEStrategy(), LexicalStrategy()]
    return _EXTRA_STRATEGIES_SINGLETON


_EXTRA_STRATEGIES_SINGLETON: list[Strategy] | None = None


def get_strategies_by_names(names: list[str], **hybrid_kwargs) -> list[Strategy]:
    """Busca strategies (default + extra) por nombre, respetando el orden.

    Si hybrid_kwargs no esta vacio, el hybrid se construye nuevo con esos
    parametros (anchos de rama BM25/semantica) en vez del singleton.
    """
    by_name = {s.name: s for s in get_all_strategies() + get_extra_strategies()}
    out: list[Strategy] = []
    for n in names:
        if n not in by_name:
            continue
        if n == "hybrid" and hybrid_kwargs:
            from agent.strategies.hybrid import HybridStrategy

            out.append(HybridStrategy(**hybrid_kwargs))
        else:
            out.append(by_name[n])
    return out


# ---------- Async runner ----------

async def _run_one(
    strategy: Strategy,
    question: str,
    history: list[dict] | None,
    k: int,
) -> StrategyComparisonResult:
    """Corre UNA strategy (retrieve + answerer) en un thread y devuelve el resultado.

    Medimos:
    - retrieval_ms: tiempo del retrieve() (ya viene en el StrategyResult)
    - answerer_ms: tiempo del answer()
    - total_ms: suma
    - tokens del answerer (input/output)
    - tokens auxiliares de la strategy (input/output, para LLM-augmented)
    - num_sources, distinct_sources
    - intent detectado
    - extra metadata de la strategy
    """
    t0 = time.time()
    metrics: dict[str, Any] = {}
    result = StrategyResult(name=strategy.name)

    # 1) Retrieve
    try:
        result = await asyncio.to_thread(
            strategy.retrieve, question, history, k
        )
    except Exception as e:
        return StrategyComparisonResult(
            name=strategy.name,
            metrics={
                "error": f"retrieve_failed: {e}",
                "total_ms": (time.time() - t0) * 1000,
            },
        )

    retrieval_ms = result.retrieval_ms
    items = result.items
    intent = result.intent

    # 2) Answerer
    t_answer = time.time()
    try:
        answer_result = await asyncio.to_thread(answer, question, items)
        answerer_ms = (time.time() - t_answer) * 1000
    except Exception as e:
        # Tenemos los items pero el answerer fallo. Devolvemos los items
        # para que se puedan ver y un error claro.
        return StrategyComparisonResult(
            name=strategy.name,
            intent=intent,
            items=items,
            metrics={
                "error": f"answer_failed: {e}",
                "retrieval_ms": retrieval_ms,
                "answerer_ms": (time.time() - t_answer) * 1000,
                "total_ms": (time.time() - t0) * 1000,
                "intent": intent,
                "num_sources": result.num_sources,
                "distinct_sources": result.distinct_sources,
                "aux_llm_input_tokens": result.llm_input_tokens,
                "aux_llm_output_tokens": result.llm_output_tokens,
                "extra": result.extra,
                "trace": result.trace_to_dict(),
            },
        )

    answerer_input_tokens = answer_result["input_tokens"]
    answerer_output_tokens = answer_result["output_tokens"]
    sources = answer_result["sources"]

    total_ms = (time.time() - t0) * 1000

    metrics: dict[str, Any] = {
        # Latencias
        "retrieval_ms": round(retrieval_ms, 2),
        "answerer_ms": round(answerer_ms, 2),
        "total_ms": round(total_ms, 2),
        # Tokens
        "answerer_input_tokens": answerer_input_tokens,
        "answerer_output_tokens": answerer_output_tokens,
        "total_input_tokens": answerer_input_tokens + result.llm_input_tokens,
        "total_output_tokens": answerer_output_tokens + result.llm_output_tokens,
        "aux_llm_input_tokens": result.llm_input_tokens,
        "aux_llm_output_tokens": result.llm_output_tokens,
        # Cobertura
        "num_sources": result.num_sources,
        "distinct_sources": result.distinct_sources,
        # Intent
        "intent": intent,
        # Extra metadata de la strategy
        "extra": result.extra,
        # Trace paso a paso del retrieve
        "trace": result.trace_to_dict(),
    }

    # Propagar error de la strategy (si el retrieve fallo pero devolvio
    # un StrategyResult con items=[] y extra.error). Sin esto, el report
    # mostraria 0/0 sources sin el motivo.
    if result.extra.get("error"):
        metrics["error"] = result.extra["error"]

    return StrategyComparisonResult(
        name=strategy.name,
        intent=intent,
        answer=answer_result["answer"],
        sources=sources,
        items=items,
        metrics=metrics,
    )


async def run_compare(
    question: str,
    history: list[dict] | None = None,
    k: int = 6,
    strategies: list[Strategy] | None = None,
) -> ComparisonResponse:
    """Corre todas las strategies en paralelo y devuelve el comparison completo."""
    if strategies is None:
        strategies = get_all_strategies()

    tasks = [_run_one(s, question, history, k) for s in strategies]
    results = await asyncio.gather(*tasks)

    return ComparisonResponse(
        question=question,
        strategies={r.name: r for r in results},
    )


# ---------------------------------------------------------------------------
# Streaming compare: corre N strategies en paralelo y emite eventos SSE
# ---------------------------------------------------------------------------

async def _run_one_stream(
    strategy: Strategy,
    question: str,
    history: list[dict] | None,
    k: int,
    queue: asyncio.Queue,
    temperature: float | None = None,
):
    """Corre UNA strategy con streaming y mete los eventos en la queue."""
    try:
        # 1) Retrieve (sync, va en un thread)
        t0 = time.time()
        result = await asyncio.to_thread(strategy.retrieve, question, history, k)
        retrieval_ms = (time.time() - t0) * 1000
        items = result.items
        intent = result.intent

        sources = _format_sources_from_items(items)
        await queue.put({
            "strategy": strategy.name,
            "type": "retrieve_done",
            "data": {
                "intent": intent,
                "sources": sources,
                "num_sources": result.num_sources,
                "distinct_sources": result.distinct_sources,
                "retrieval_ms": round(retrieval_ms, 2),
                "aux_llm_input_tokens": result.llm_input_tokens,
                "aux_llm_output_tokens": result.llm_output_tokens,
                "extra": result.extra,
                "trace": result.trace_to_dict(),
            },
        })

        # 2) Answerer streaming (async)
        t_answer = time.time()
        token_gen, usage = await stream_answer_async(question, items, temperature)
        full_text = ""
        async for token in token_gen:
            full_text += token
            await queue.put({
                "strategy": strategy.name,
                "type": "token",
                "data": token,
            })
        answerer_ms = (time.time() - t_answer) * 1000

        await queue.put({
            "strategy": strategy.name,
            "type": "done",
            "data": {
                "answer": full_text,
                "sources": sources,
                "answerer_ms": round(answerer_ms, 2),
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "items": [item_to_dict(i) for i in items],
            },
        })
    except Exception as e:
        await queue.put({
            "strategy": strategy.name,
            "type": "error",
            "data": str(e),
        })


async def run_compare_stream(
    question: str,
    history: list[dict] | None = None,
    k: int = 6,
    strategies: list[Strategy] | None = None,
    temperature: float | None = None,
):
    """Async generator: corre las strategies en paralelo y yield eventos.

    Cada evento es un dict:
        {"strategy": ..., "type": "retrieve_done"|"token"|"done"|"error", "data": ...}
    """
    if strategies is None:
        strategies = get_all_strategies()

    queue: asyncio.Queue = asyncio.Queue()
    total = len(strategies)

    for s in strategies:
        asyncio.create_task(_run_one_stream(s, question, history, k, queue, temperature))

    finished = 0
    while finished < total:
        msg = await queue.get()
        yield msg
        if msg["type"] in ("done", "error"):
            finished += 1
