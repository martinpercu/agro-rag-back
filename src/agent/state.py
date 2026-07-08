"""Estado del agente LangGraph."""
from __future__ import annotations

from typing import Literal, TypedDict

Intent = Literal[
    "costos",
    "mercado",
    "proyecciones",
    "tecnologia",
    "ganaderia",
    "siembras",
    "general",
]


class AgentState(TypedDict, total=False):
    """Estado que pasa por los nodos del grafo.

    - question: la pregunta cruda del productor
    - intent: clasificacion rapida de la intencion
    - retrieved: lista de (Chunk, score) recuperadas del vector store
    - answer: respuesta final generada por el LLM
    - sources: lista de paginas/secciones citadas (para mostrar al productor y para el PDF)
    """

    question: str
    intent: Intent
    retrieved: list[tuple[dict, float]]
    answer: str
    sources: list[dict]
