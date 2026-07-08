"""Strategy 5: multi query con RRF.

Idea: el LLM genera 3 reformulaciones de la pregunta original. Cada
reformulacion se busca por separado en el chroma. Los 3 rankings se
fusionan con Reciprocal Rank Fusion (RRF) y nos quedamos con el top-k.

El objetivo es expandir recall: una sola pregunta puede no cubrir todos
los angulos. Tres reformulaciones cubren mas superficie del vector store
y el RRF elige lo que aparece en varios rankings (señal de relevancia).
"""
from __future__ import annotations

import json
import re
import time

from openai import OpenAI

from agent.nodes.classifier import _classify
from agent.nodes.retriever import INTENT_TO_SECTION
from agent.strategies.base import RetrievedItem, Strategy, StrategyResult
from agent.strategies.llm_retry import call_with_retry
from ingestion.indexer import search


DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_N_QUERIES = 3
DEFAULT_K = 6
MAX_TOKENS = 300
TEMPERATURE = 0.3  # algo de variabilidad para queries distintas
_RRF_K = 60
_CHROMA_TOP_K = 10  # por sub-query, antes del RRF

GENERATE_PROMPT = """\
Sos un generador de busquedas alternativas para un sistema RAG agropecuario.
Dada una pregunta de un productor argentino, tu UNICA tarea es generar
{exactamente_n} reformulaciones de la pregunta que podrian recuperar
informacion complementaria desde una revista de margenes agropecuarios.

Reglas:
- Cada reformulacion debe ser una pregunta completa en espanol rioplatense.
- Las {exactamente_n} versiones deben ser DISTINTAS entre si (cubrir
  angulos diferentes, sinonimos, o aspectos relacionados).
- NO respondas la pregunta, solo reformulala.
- Devolvé SOLO un JSON array de {exactamente_n} strings, sin prose,
  sin markdown, sin explicacion.

Pregunta del productor: "{query}"

JSON array:"""


def _parse_queries(content: str, n: int) -> list[str]:
    """Extrae N strings del output del LLM. Robusto a markdown y prose."""
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
        m = re.search(r"\[[\s\S]*?\]", s)
        if not m:
            return []
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, list):
        return []
    out: list[str] = []
    for x in parsed:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
        if len(out) >= n:
            break
    return out


def _rrf_merge(rankings: list[list[str]], k: int = _RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class MultiQueryStrategy(Strategy):
    name = "multi_query"

    def __init__(self, model: str = DEFAULT_MODEL, n_queries: int = DEFAULT_N_QUERIES):
        self.model = model
        self.n_queries = n_queries
        self.client = OpenAI()

    def retrieve(
        self,
        question: str,
        history: list[dict] | None = None,
        k: int = DEFAULT_K,
    ) -> StrategyResult:
        t0 = time.time()

        # 1) Clasificamos el intent UNA vez sobre la pregunta original
        # y lo aplicamos a las N sub-queries (consistente con el resto
        # de las strategies)
        intent = _classify(question)
        allowed = INTENT_TO_SECTION.get(intent)

        # 2) LLM genera N reformulaciones
        gen_in_tok = 0
        gen_out_tok = 0
        generated: list[str] = []
        try:
            response = call_with_retry(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": GENERATE_PROMPT.format(
                        exactamente_n=self.n_queries,
                        query=question,
                    ),
                }],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            content = response.choices[0].message.content or ""
            if response.usage:
                gen_in_tok = response.usage.prompt_tokens
                gen_out_tok = response.usage.completion_tokens
            generated = _parse_queries(content, self.n_queries)
        except Exception as e:
            generated = []
            llm_error = f"generation_failed: {e}"
        else:
            llm_error = None

        # Si el LLM fallo o dio < N, fallback a busqueda unica con la
        # pregunta original (igual que el baseline)
        if not generated:
            generated = [question]
            fallback = "llm_failed_or_parse"
        elif len(generated) < self.n_queries:
            # Pad con la pregunta original hasta tener N
            while len(generated) < self.n_queries:
                generated.append(question)
            fallback = "partial_rewrites"
        else:
            fallback = None

        # 3) Retrieval para cada sub-query
        rankings: list[list[str]] = []
        per_query_hits: dict[str, list[tuple]] = {}
        for q in generated:
            if allowed is None:
                hits = search(q, k=_CHROMA_TOP_K)
            else:
                hits = search(q, k=_CHROMA_TOP_K, where={"seccion": {"$in": allowed}})
            rk = [chunk.id for chunk, _ in hits]
            rankings.append(rk)
            per_query_hits[q] = hits

        # 4) RRF merge
        merged = _rrf_merge(rankings, k=_RRF_K)

        # 5) Map a RetrievedItem. Necesitamos el chunk completo, lo
        # buscamos en cualquiera de los rankings (todos apuntan al chroma)
        # Construimos un map chunk_id -> (chunk, score_original)
        chunk_map: dict[str, tuple] = {}
        for q, hits in per_query_hits.items():
            for chunk, score in hits:
                if chunk.id not in chunk_map:
                    chunk_map[chunk.id] = (chunk, score)

        items: list[RetrievedItem] = []
        for rank, (cid, rrf_score) in enumerate(merged[:k]):
            entry = chunk_map.get(cid)
            if not entry:
                continue
            chunk, _ = entry
            items.append(
                RetrievedItem(
                    chunk_id=cid,
                    text=chunk.text,
                    seccion=chunk.metadata.seccion,
                    pagina=chunk.metadata.pagina,
                    cultivo=chunk.metadata.cultivo,
                    campana=chunk.metadata.campana,
                    tipo=chunk.metadata.tipo,
                    score=float(rrf_score),
                    rank=rank,
                )
            )

        elapsed_ms = (time.time() - t0) * 1000

        # Total tokens (generator + los embeddings de cada sub-query,
        # que chroma hace internamente). Solo contamos los del generator.
        return StrategyResult(
            name=self.name,
            items=items,
            intent=intent,
            retrieval_ms=elapsed_ms,
            llm_input_tokens=gen_in_tok,
            llm_output_tokens=gen_out_tok,
            extra={
                "model": self.model,
                "n_queries": len(generated),
                "generated_queries": generated,
                "chroma_top_k_per_query": _CHROMA_TOP_K,
                "rrf_k": _RRF_K,
                "fallback": fallback,
                "llm_error": llm_error,
                "filter_sections": allowed,
            },
        )
