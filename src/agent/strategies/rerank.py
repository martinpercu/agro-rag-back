"""Strategy 3: rerank con LLM (gpt-4.1-nano por default, configurable).

Pasos:
1. Recupera top-N del chroma (semantico, con filtro por intent).
2. Envia los N fragmentos al LLM con un prompt estricto que pide
   devolver un JSON array con los K indices mas relevantes.
3. Reordena los chunks segun ese ranking; si el JSON no parsea,
   cae al orden original como fallback.

La idea del comparador: usamos el MISMO LLM que escribe las respuestas
para hacer de reranker. Asi demostramos que con el mismo modelo
y distintos prompts/formatos, se obtienen outputs diferentes para RAG.
"""
from __future__ import annotations

import json
import re
import time

from openai import OpenAI

from agent.llm import get_chat_client, llm_model, seed_for_temperature
from agent.nodes.classifier import _classify
from agent.nodes.retriever import INTENT_TO_SECTION
from agent.strategies.base import RetrievedItem, Strategy, StrategyResult, TraceRecorder
from agent.strategies.llm_retry import call_with_retry
from ingestion.indexer import search


DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_RETRIEVE_TOP_N = 20
DEFAULT_RERANK_TOP_K = 6
MAX_TOKENS = 200
TEMPERATURE = 0.0  # determinismo en el rerank
TRUNCATE_PER_CHUNK = 800

RERANK_PROMPT = """\
Sos un reranker de fragmentos para un sistema RAG. Dada la pregunta de un
productor agropecuario y {n} fragmentos de la revista Margenes Agropecuarios,
tu UNICA tarea es devolver un JSON array (sin prose, sin markdown, sin
explicaciones) con los indices de los {k} fragmentos MAS RELEVANTES para
responder esa pregunta, en orden de relevancia descendente.

Reglas estrictas:
- Devolve SOLO el JSON array, nada mas.
- Los indices son enteros de 0 a {n_minus_1}.
- Si un fragmento no aporta a la pregunta, no lo incluyas.
- No te inventes informacion: solo juzga relevancia.

Pregunta del productor: "{query}"

Fragmentos:
{fragments}

JSON array:"""


def _format_fragments(items: list[RetrievedItem]) -> str:
    """Genera el bloque de fragmentos para el prompt. Trunca a TRUNCATE_PER_CHUNK."""
    parts: list[str] = []
    for i, item in enumerate(items):
        text = item.text.strip().replace("\n", " ")[:TRUNCATE_PER_CHUNK]
        meta_pag = f"pag. {item.pagina}" if item.pagina is not None else "s/p"
        meta_sec = item.seccion or "s/seccion"
        parts.append(f"[{i}] ({meta_pag}, {meta_sec}) {text}")
    return "\n".join(parts)


def _parse_indices(content: str, n: int) -> list[int]:
    """Extrae y valida una lista de indices del output del LLM.

    Robusto a:
    - Markdown fences ```json ... ```
    - Prose alrededor del JSON
    - Indices fuera de rango (los descarta)
    - Duplicados (los dedupa preservando orden)
    """
    if not content:
        return []
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    parsed: object
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\[[\d,\s]*\]", s)
        if not m:
            return []
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, list):
        return []
    seen: set[int] = set()
    out: list[int] = []
    for x in parsed:
        if isinstance(x, int) and not isinstance(x, bool) and 0 <= x < n and x not in seen:
            seen.add(x)
            out.append(x)
    return out


class RerankStrategy(Strategy):
    name = "rerank"

    def __init__(
        self,
        model: str | None = None,
        top_n: int = DEFAULT_RETRIEVE_TOP_N,
        client: OpenAI | None = None,
    ):
        self.model = model or llm_model()
        self.top_n = top_n
        self.client = client or get_chat_client()

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

        # 1) Retrieve top-N from chroma
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

        # 2) LLM rerank
        prompt = RERANK_PROMPT.format(
            k=k,
            n=len(candidates),
            n_minus_1=len(candidates) - 1,
            query=question,
            fragments=_format_fragments(candidates),
        )

        input_tokens = 0
        output_tokens = 0
        content = ""
        try:
            response = call_with_retry(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                seed=seed_for_temperature(TEMPERATURE),
            )
            content = response.choices[0].message.content or ""
            if response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
        except Exception as e:
            return StrategyResult(
                name=self.name,
                items=candidates[:k],
                intent=intent,
                retrieval_ms=tr.steps[-1].acc_ms if tr.steps else 0.0,
                extra={"error": f"llm_call_failed: {e}"},
                trace=tr.steps,
            )
        tr.step(
            "llm_rerank",
            f"k={k}, n={len(candidates)}, model={self.model}",
            at_t=time.monotonic(),
        )

        # 3) Parse indices
        indices = _parse_indices(content, len(candidates))

        # 4) Build final ranking
        if not indices:
            final_items = list(candidates[:k])
            fallback = "parse_failed"
        else:
            final_items = [candidates[i] for i in indices[:k]]
            seen_ids = {i.chunk_id for i in final_items}
            for c in candidates:
                if c.chunk_id not in seen_ids and len(final_items) < k:
                    final_items.append(c)
                    seen_ids.add(c.chunk_id)
            fallback = None

        # 5) Re-rank + 1/(rank+1) score (rank-confidence)
        for rank, item in enumerate(final_items):
            item.rank = rank
            item.score = 1.0 / (rank + 1)

        tr.step(
            "build_ranking",
            f"k={k}, indices={len(indices)}, fallback={fallback}",
            at_t=time.monotonic(),
        )
        elapsed_ms = tr.steps[-1].acc_ms if tr.steps else 0.0

        return StrategyResult(
            name=self.name,
            items=final_items,
            intent=intent,
            retrieval_ms=elapsed_ms,
            llm_input_tokens=input_tokens,
            llm_output_tokens=output_tokens,
            extra={
                "model": self.model,
                "retrieved_top_n": self.top_n,
                "rerank_top_k": k,
                "llm_indices_returned": len(indices),
                "llm_raw_response_preview": content[:200],
                "fallback": fallback,
                "filter_sections": allowed,
            },
            trace=tr.steps,
        )
