"""Nodo retriever: busca chunks relevantes segun la intencion clasificada."""
from __future__ import annotations

from agent.state import AgentState, Intent
from ingestion.indexer import search

# Mapa intent -> filtro de seccion para el vector store.
# Si no matchea, no filtra (deja que la busqueda semantica sola decida).
INTENT_TO_SECTION: dict[Intent, list[str] | None] = {
    "costos": [
        "costos_margenes",
        "costos_operativos",
        "siembras",
    ],
    "mercado": [
        "mercado_precios",
        "analisis_mercado",
        "fas_teorico",
    ],
    "proyecciones": [
        "proyecciones",
        "siembras",
    ],
    "tecnologia": [
        "tecnologia",
        "insumos_maquinaria",
    ],
    "ganaderia": [
        "ganaderia_costos",
    ],
    "siembras": [
        "siembras",
    ],
    "general": None,  # sin filtro
}

DEFAULT_K = 6


def retriever_node(state: AgentState) -> AgentState:
    question = state["question"]
    intent = state.get("intent", "general")
    history = state.get("history")
    allowed_sections = INTENT_TO_SECTION.get(intent)

    # Helper to run search (kept separate for tracing)
    def _do_search() -> list[tuple[dict, float]]:
        chunks: list[tuple[dict, float]] = []
        if allowed_sections is None:
            for chunk, score in search(question, k=DEFAULT_K):
                chunks.append((chunk.model_dump(), score))
        else:
            for chunk, score in search(
                question,
                k=DEFAULT_K,
                where={"seccion": {"$in": allowed_sections}},
            ):
                chunks.append((chunk.model_dump(), score))
        return chunks

    # Langfuse retriever span
    try:
        from observability import get_langfuse

        lf = get_langfuse()
        if lf is not None:
            with lf.start_as_current_observation(
                name="retriever",
                as_type="retriever",
                input={"question": question, "history": history, "intent": intent, "allowed_sections": allowed_sections},
            ) as _span:
                chunks = _do_search()
                state["retrieved"] = chunks
                try:
                    sources = [
                        {
                            "seccion": c[0].get("metadata", {}).get("seccion"),
                            "pagina": c[0].get("metadata", {}).get("pagina"),
                            "score": round(c[1], 3),
                        }
                        for c in chunks
                    ]
                    lf.update_current_span(
                        output={"retrieved_count": len(chunks), "sources": sources},
                        metadata={"intent": intent},
                    )
                except Exception:
                    pass
                return state
    except Exception:
        pass

    state["retrieved"] = _do_search()
    return state
