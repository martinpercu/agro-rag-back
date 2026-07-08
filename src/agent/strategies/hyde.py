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

from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from agent.nodes.classifier import _classify
from agent.nodes.retriever import INTENT_TO_SECTION
from agent.strategies.base import RetrievedItem, Strategy, StrategyResult
from agent.strategies.llm_retry import call_with_retry
from ingestion.indexer import search


DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
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
    """Default embed function: usa OpenAIEmbeddings."""
    return OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL).embed_query(text)


class HydeStrategy(Strategy):
    name = "hyde"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        client: OpenAI | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.client = client or OpenAI()
        if embed_fn is None:
            self._embed_fn = lambda text: OpenAIEmbeddings(  # noqa: E731
                model=embedding_model
            ).embed_query(text)
        else:
            self._embed_fn = embed_fn

    def retrieve(
        self,
        question: str,
        history: list[dict] | None = None,
        k: int = DEFAULT_K,
    ) -> StrategyResult:
        t0 = time.time()
        intent = _classify(question)
        allowed = INTENT_TO_SECTION.get(intent)

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
                retrieval_ms=(time.time() - t0) * 1000,
                extra={"error": f"llm_failed: {e}"},
            )

        if not hypothetical:
            return StrategyResult(
                name=self.name,
                items=[],
                intent=intent,
                retrieval_ms=(time.time() - t0) * 1000,
                extra={"error": "llm_returned_empty"},
            )

        # 2) Embed el hipotetico
        try:
            query_vector = self._embed_fn(hypothetical)
        except Exception as e:
            return StrategyResult(
                name=self.name,
                items=[],
                intent=intent,
                retrieval_ms=(time.time() - t0) * 1000,
                extra={"error": f"embedding_failed: {e}"},
            )

        # 3) Chroma search con el vector del hipotetico
        if allowed is None:
            hits = search(k=k, query_vector=query_vector)
        else:
            hits = search(
                k=k,
                where={"seccion": {"$in": allowed}},
                query_vector=query_vector,
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

        elapsed_ms = (time.time() - t0) * 1000

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
        )
