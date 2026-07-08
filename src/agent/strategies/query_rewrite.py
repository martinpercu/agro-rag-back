"""Strategy 4: query rewriting con history (PROTOTIPO).

Idea: dado el historial de la conversacion y la ultima pregunta del
productor, el LLM reescribe la pregunta para que sea autonoma y clara
(sin referencias vagas como "y eso?", "esa zona", "el de antes").

Despues corre el retrieval (chroma + filtro por intent) con la pregunta
reescrita.

PROTOTIPO: implementacion minima para validar el concepto. Se
completa con tests + tuning de prompt en el Paso 4 propiamente dicho.
"""
from __future__ import annotations

import os
import time

from openai import OpenAI

from agent.nodes.classifier import _classify
from agent.nodes.retriever import INTENT_TO_SECTION
from agent.strategies.base import RetrievedItem, Strategy, StrategyResult
from agent.strategies.llm_retry import call_with_retry
from ingestion.indexer import search


DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_K = 6
MAX_HISTORY = 8
MAX_TOKENS = 200
TEMPERATURE = 0.0


REWRITE_PROMPT = """\
Sos un reescritor de queries para un sistema RAG agropecuario. Dada la
conversacion y la ultima pregunta del productor, tu unica tarea es
devolver la pregunta REESCRITA para que sea autonoma y clara.

Reglas:
- Si la pregunta ya es autonoma y no tiene referencias vagas,
  devolvela tal cual.
- Si tiene referencias como "y eso?", "esa zona", "el de antes",
  "el mismo", reescribila incluyendo el contexto que sacas de la
  conversacion.
- NO agregues informacion que no este en la conversacion.
- Devolvé SOLO la pregunta reescrita, sin comillas, sin prosa,
  sin "Pregunta:" ni nada extra.

Conversacion:
{history}

Ultima pregunta del productor: {question}

Pregunta reescrita:"""


def _format_history(history: list[dict] | None) -> str:
    if not history:
        return "(sin conversacion previa)"
    # Tomamos los ultimos MAX_HISTORY mensajes
    recent = history[-MAX_HISTORY:]
    lines = []
    for m in recent:
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"[{role}] {content}")
    return "\n".join(lines) if lines else "(sin conversacion previa)"


def _build_query(question: str, history: list[dict] | None) -> tuple[str, int, int, str]:
    """Llama al LLM para reescribir. Devuelve (rewritten, in_tok, out_tok, raw).

    Si no hay API key o falla, devuelve la pregunta original.
    """
    if not history:
        return question, 0, 0, ""

    client = OpenAI()
    prompt = REWRITE_PROMPT.format(
        history=_format_history(history),
        question=question,
    )
    try:
        response = call_with_retry(
            client.chat.completions.create,
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        content = (response.choices[0].message.content or "").strip()
        in_tok = response.usage.prompt_tokens if response.usage else 0
        out_tok = response.usage.completion_tokens if response.usage else 0
        # Si el LLM devuelve vacio o solo whitespace, fallback
        if not content:
            return question, in_tok, out_tok, ""
        return content, in_tok, out_tok, content
    except Exception as e:
        return question, 0, 0, f"error: {e}"


class QueryRewriteStrategy(Strategy):
    name = "query_rewrite"

    def retrieve(
        self,
        question: str,
        history: list[dict] | None = None,
        k: int = DEFAULT_K,
    ) -> StrategyResult:
        t0 = time.time()
        rewritten, in_tok, out_tok, raw = _build_query(question, history)
        intent = _classify(rewritten)
        allowed = INTENT_TO_SECTION.get(intent)

        # Retrieval con la pregunta reescrita
        if allowed is None:
            hits = search(rewritten, k=k)
        else:
            hits = search(rewritten, k=k, where={"seccion": {"$in": allowed}})

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
        changed = rewritten.strip().lower() != question.strip().lower()

        return StrategyResult(
            name=self.name,
            items=items,
            intent=intent,
            retrieval_ms=elapsed_ms,
            llm_input_tokens=in_tok,
            llm_output_tokens=out_tok,
            extra={
                "model": DEFAULT_MODEL,
                "original_query": question,
                "rewritten_query": rewritten,
                "rewritten_changed": changed,
                "history_used": len(history) if history else 0,
                "filter_sections": allowed,
            },
        )
