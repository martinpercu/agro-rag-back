"""Strategy 6: HyDe (Hypothetical Document Embeddings).

Idea: el LLM genera un parrafo breve que SIMULA la respuesta a la
pregunta (sin usar su conocimiento del mundo real, solo el formato y
estilo). Ese parrafo se embebe y se usa como query en el chroma.

La hipotesis (del paper original de Gao et al. 2022) es que el embedding
de una respuesta "bien formada" matchea mejor los chunks reales que
el embedding de la pregunta cruda del productor (corta y coloquial).
"""
from __future__ import annotations

import time
from typing import Callable

from openai import OpenAI

from agent.llm import (
    embedding_model as env_embedding_model,
    get_chat_client,
    get_embeddings,
    llm_model,
)
from agent.nodes.classifier import _classify
from agent.nodes.retriever import INTENT_TO_SECTION
from agent.strategies.base import RetrievedItem, Strategy, StrategyResult, TraceRecorder
from agent.strategies.llm_retry import call_with_retry
from ingestion.indexer import search


DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_K = 6
MAX_TOKENS = 300
TEMPERATURE = 0.3

HYDE_PROMPT = """\
Imaginate que tenes que responder esta pregunta de un productor
agropecuario argentino. NO uses tu conocimiento del mundo real.
Solo escribi un parrafo breve (100-200 palabras) con el FORMATO y
ESTILO de la respuesta, como si la respuesta existiera en una revista
tecnica del sector. Inclui numeros con unidades tipicas (US$/ha,
qq/ha, US$/kg, US$/tn) aunque sean placeholders. El parrafo tiene
que sonar como un fragmento real de la publicacion, no como una
respuesta generada.

Pregunta: "{query}"

Parrafo hipotetico:"""


def _default_embed_fn(text: str) -> list[float]:
    """Default embed function: usa el embedding configurado (env)."""
    return get_embeddings().embed_query(text)


class HydeStrategy(Strategy):
    name = "hyde"

    def __init__(
        self,
        model: str | None = None,
        embedding_model: str | None = None,
        client: OpenAI | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
    ):
        self.model = model or llm_model()
        self.embedding_model = embedding_model or env_embedding_model()
        self.client = client or get_chat_client()
        if embed_fn is None:
            self._embed_fn = lambda text: get_embeddings(self.embedding_model).embed_query(text)
        else:
            self._embed_fn = embed_fn

    def retrieve(
        self,
        question: str,
        history: list[dict] | None = None,
        k: int = DEFAULT_K,
    ) -> StrategyResult:
        tr = TraceRecorder()
        intent = _classify(question)
        allowed = INTENT_TO_SECTION.get(intent)
        filtro = f", filtro={allowed}" if allowed else ", sin filtro de seccion"
        tr.step("classify", f"intent={intent}{filtro}", at_t=time.monotonic())

        # 1) LLM genera el parrafo hipotetico (con retry ante rate limit)
        gen_in_tok = 0
        gen_out_tok = 0
        hypothetical = ""
        try:
            response = call_with_retry(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": HYDE_PROMPT.format(query=question),
                }],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            hypothetical = (response.choices[0].message.content or "").strip()
            if response.usage:
                gen_in_tok = response.usage.prompt_tokens
                gen_out_tok = response.usage.completion_tokens
        except Exception as e:
            return StrategyResult(
                name=self.name,
                items=[],
                intent=intent,
                retrieval_ms=tr.steps[-1].acc_ms if tr.steps else 0.0,
                extra={"error": f"llm_failed: {e}"},
                trace=tr.steps,
            )
        tr.step(
            "llm_hypothetical",
            f"chars={len(hypothetical)}, model={self.model}",
            at_t=time.monotonic(),
        )

        if not hypothetical:
            return StrategyResult(
                name=self.name,
                items=[],
                intent=intent,
                retrieval_ms=tr.steps[-1].acc_ms if tr.steps else 0.0,
                extra={"error": "llm_returned_empty"},
                trace=tr.steps,
            )

        # 2) Embed el hipotetico
        try:
            query_vector = self._embed_fn(hypothetical)
        except Exception as e:
            return StrategyResult(
                name=self.name,
                items=[],
                intent=intent,
                retrieval_ms=tr.steps[-1].acc_ms if tr.steps else 0.0,
                extra={"error": f"embedding_failed: {e}"},
                trace=tr.steps,
            )
        tr.step(
            "embed_query",
            f"model={self.embedding_model}, dims={len(query_vector)}",
            at_t=time.monotonic(),
        )

        # 3) Chroma search con el vector del hipotetico
        timings: dict = {}
        t_search = time.monotonic()
        if allowed is None:
            hits = search(k=k, query_vector=query_vector, timings=timings)
        else:
            hits = search(
                k=k,
                where={"seccion": {"$in": allowed}},
                query_vector=query_vector,
                timings=timings,
            )
        chroma_ms = timings.get("chroma_ms", 0) / 1000
        tr.step(
            "chroma_search",
            f"k={k}, hits={len(hits)}",
            at_t=t_search + chroma_ms,
        )

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

        elapsed_ms = tr.steps[-1].acc_ms if tr.steps else 0.0

        return StrategyResult(
            name=self.name,
            items=items,
            intent=intent,
            retrieval_ms=elapsed_ms,
            llm_input_tokens=gen_in_tok,
            llm_output_tokens=gen_out_tok,
            extra={
                "model": self.model,
                "embedding_model": self.embedding_model,
                "hypothetical_chars": len(hypothetical),
                "hypothetical_preview": hypothetical[:200],
                "filter_sections": allowed,
            },
            trace=tr.steps,
        )
