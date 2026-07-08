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
    allowed_sections = INTENT_TO_SECTION.get(intent)

    chunks: list[tuple[dict, float]] = []

    if allowed_sections is None:
        # Sin filtro: busqueda semantica libre
        for chunk, score in search(question, k=DEFAULT_K):
            chunks.append((chunk.model_dump(), score))
    else:
        # Filtrado por secciones. ChromaDB $in permite OR.
        for chunk, score in search(
            question,
            k=DEFAULT_K,
            where={"seccion": {"$in": allowed_sections}},
        ):
            chunks.append((chunk.model_dump(), score))

    state["retrieved"] = chunks
    return state
