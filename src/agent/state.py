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
    - plan_intent: si el productor quiere ayuda para decidir qué plantar (Fase 2 baby)
    - divisions: potreros detectados [{hectares: str, cultivo: str|None}]
    - location: ubicación opcional {label?} (Fase 2 abierto, por ahora vacío)
    - history: historial reciente [{role, content}] para detectar intención sutil
    """

    question: str
    intent: Intent
    retrieved: list[tuple[dict, float]]
    answer: str
    sources: list[dict]
    # Fase 2 — dueño de tierra, sin cliente/precio por ahora
    plan_intent: bool
    divisions: list[dict]
    location: dict | None
    history: list[dict]
