"""Nodo plan_intent: detecta si el productor quiere armar un plan de siembra.

Fase 2 baby-step: solo para dueño de tierra, sin client_hint / price_variants.
Rule-based liviano, sin LLM, para no gastar tokens.
"""
from __future__ import annotations

import re

from agent.state import AgentState

# Keywords que indican intención de planificar siembra/división de campo
_PLAN_KEYWORDS = (
    "plan de siembra",
    "plan siembra",
    "que siembro",
    "qué siembro",
    "que sembrar",
    "qué sembrar",
    "que plantar",
    "qué plantar",
    "que me conviene",
    "qué me conviene",
    "como divido",
    "cómo divido",
    "como dividir",
    "cómo dividir",
    "campaña 26/27",
    "campaña 2026/27",
    "campaña que viene",
    "campana que viene",
    "proxima siembra",
    "próxima siembra",
    "tengo",
    "dispongo",
    "cuento con",
)

# Si la pregunta contiene hectáreas, es señal fuerte de plan (aunque no diga "plan")
_HA_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(ha|hectarea|hectárea|hectareas|hectáreas)\b", re.I)

# Señal de división explícita
_DIVIDE_RE = re.compile(r"\b(dividir|division|división|repartir|distribuir|lote|potrero|parcel)\b", re.I)


def is_plan_intent(question: str, history: list[dict] | None = None) -> bool:
    """Heurística sutil: ¿el user quiere ayuda para decidir qué plantar?

    - No fuerza: si no hay señal, devuelve False y el grafo sigue RAG normal.
    - Usa question + última history si está.
    """
    if not question or not question.strip():
        return False
    q = question.lower()

    # 1) keywords directas de plan
    for kw in _PLAN_KEYWORDS:
        if kw in q:
            # "tengo" solo cuenta si además hay ha o siembra/cultivo cerca
            if kw in ("tengo", "dispongo", "cuento con"):
                if _HA_RE.search(q) or any(c in q for c in ("siembra", "sembrar", "cultivo", "soja", "maiz", "maíz", "trigo", "girasol")):
                    return True
                continue
            return True

    # 2) mención de hectáreas + cultivo/siembra => plan
    if _HA_RE.search(q):
        if any(c in q for c in ("soja", "maiz", "maíz", "trigo", "girasol", "cebada", "sorgo", "cultivo", "siembra", "sembrar", "lote", "potrero")):
            return True
        # hectáreas + "qué hago / qué me conviene / dividir"
        if _DIVIDE_RE.search(q) or "conviene" in q or "recom" in q:
            return True

    # 3) historial: si ya venía hablando de plan, mantener intent
    if history:
        last = " ".join(str(m.get("content", "")) for m in history[-3:]).lower()
        if any(kw in last for kw in ("plan de siembra", "hectareas", "hectáreas", "ha ")) and _HA_RE.search(q):
            return True

    return False


def plan_intent_node(state: AgentState) -> AgentState:
    """Nodo LangGraph: anota plan_intent en el state."""
    question = state.get("question", "")
    # history puede venir en state como lista de dicts {role, content}
    history = state.get("history")  # type: ignore[attr-defined]
    state["plan_intent"] = is_plan_intent(question, history if isinstance(history, list) else None)  # type: ignore
    return state
