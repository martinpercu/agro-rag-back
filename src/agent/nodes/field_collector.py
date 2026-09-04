"""Nodo field_collector abierto: extrae hectáreas/cultivo de lenguaje natural.

Fase 2 baby-step dueño: solo divisions + location vacío, sin client_hint/price.
Todo opcional: si no hay hectáreas, divisions=[] y no bloquea el flujo.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from agent.state import AgentState

# hectáreas: 120ha, 120 ha, 80,5 ha, 150 hectáreas
_HA_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(ha\b|hectareas\b|hectáreas\b|hectarea\b|hectárea\b)",
    re.I,
)

# cultivos conocidos (align con classifier.py)
_CULTIVOS = [
    "soja",
    "maiz",
    "maíz",
    "trigo",
    "girasol",
    "cebada",
    "sorgo",
    "colza",
    "avena",
    "centeno",
    "arroz",
    "mani",
    "maní",
    "algodon",
    "algodón",
    "lino",
    "cártamo",
    "cartamo",
]
_CULTIVO_RE = re.compile(r"\b(" + "|".join(_CULTIVOS) + r")\b", re.I)

# normalización de cultivo -> sin acento lower
_NORMALIZE = {"maíz": "maiz", "maní": "mani", "algodón": "algodon", "cártamo": "cartamo"}


def _norm_cultivo(c: str) -> str:
    low = c.lower()
    return _NORMALIZE.get(low, low)


def _parse_hectares(raw: str) -> str | None:
    """Devuelve hectares como string Decimal normalizado, o None si no es válido."""
    s = raw.replace(",", ".").strip()
    try:
        d = Decimal(s)
        if d <= 0 or d > 100000:
            return None
        # Normalizar: quitar trailing zeros pero mantener string
        # "80.0" -> "80", "80.5" -> "80.5"
        n = format(d.normalize(), "f") if d == d.to_integral_value() else format(d, "f")
        # Decimal normalize may produce exponent; fallback
        if "E" in n or "e" in n:
            n = format(d, "f").rstrip("0").rstrip(".")
        return n
    except (InvalidOperation, ValueError, AttributeError):
        return None


def extract_divisions(question: str, history: list[dict] | None = None) -> list[dict]:
    """Extrae divisions [{hectares: str, cultivo: str|None}] de question + history.

    - Cada match "120ha" es una división.
    - Si hay "ha" en el texto, también captura números sueltos cerca de cultivo
      (ej. "150 ha ... 100 soja y 50 trigo" → 3 divisions). Baby-step sin NER complejo.
    - Cultivo se busca en ventana ±50 chars alrededor del match.
    """
    if not question:
        return []
    text = question
    # Incluir último mensaje del historial para contexto ("y de soja?" tras "tengo 80ha")
    if history:
        last = " ".join(str(m.get("content", "")) for m in history[-2:])
        text = f"{last} {question}"

    divisions: list[dict] = []
    # cultivos globales en el texto (para fallback)
    global_cultivos = [_norm_cultivo(m.group(1)) for m in _CULTIVO_RE.finditer(text)]
    has_ha = bool(_HA_RE.search(text))
    seen_spans: set[tuple[int, int]] = set()

    def _nearest_cultivo(pos: int, end_pos: int) -> str | None:
        # buscar primero hacia adelante 50 chars, luego hacia atrás 50
        fwd = text[end_pos : end_pos + 50]
        m1 = _CULTIVO_RE.search(fwd)
        if m1:
            return _norm_cultivo(m1.group(1))
        bwd = text[max(0, pos - 50) : pos]
        m2 = _CULTIVO_RE.search(bwd)
        # elegir el más cercano atrás (último match)
        if m2:
            # tomar el último ocurrencia en bwd
            last = None
            for mm in _CULTIVO_RE.finditer(bwd):
                last = mm
            if last:
                return _norm_cultivo(last.group(1))
        return None

    for m in _HA_RE.finditer(text):
        raw_num = m.group(1)
        hectares = _parse_hectares(raw_num)
        if not hectares:
            continue
        seen_spans.add((m.start(1), m.end(1)))
        cultivo = _nearest_cultivo(m.start(), m.end())
        # fallback: si solo hay un cultivo global y no encontramos, usarlo
        if cultivo is None and len(set(global_cultivos)) == 1:
            cultivo = global_cultivos[0]
        divisions.append({"hectares": hectares, "cultivo": cultivo})

    # Segundo pase: números sueltos cerca de cultivo cuando ya hay un "ha" en el texto
    # ej. "tengo 150ha quiero 100 soja y 50 maiz" → captura 100 y 50
    if has_ha:
        # números que no son año (evitar 2026/27) y que están cerca de cultivo
        _LOOSE_NUM_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\b")
        for m in _LOOSE_NUM_RE.finditer(text):
            # ya capturado como ha ?
            if (m.start(1), m.end(1)) in seen_spans:
                continue
            # evitar años 202x
            if m.group(1) in ("2026", "2027", "2025"):
                continue
            hectares = _parse_hectares(m.group(1))
            if not hectares:
                continue
            cultivo = _nearest_cultivo(m.start(), m.end())
            if not cultivo:
                continue
            # evitar duplicar el mismo ha+cultivo ya existente
            if any(d["hectares"] == hectares and d["cultivo"] == cultivo for d in divisions):
                continue
            divisions.append({"hectares": hectares, "cultivo": cultivo})

    # Si hay 3 divisions y la primera es el total (sum de las otras), sutil: mantener todas
    # pero el frontend puede mostrarlas como potreros. No filtramos por ahora (baby step deja todo).
    return divisions


def field_collector_node(state: AgentState) -> AgentState:
    """Nodo LangGraph: extrae divisions/location y los anota en state.

    - Si state ya trae divisions (de una investigada previa), mergea solo si la nueva
      extracción no es vacía (no pisa con []).
    - location por ahora siempre {} vacío (abierto para futuro DI-5).
    """
    question = state.get("question", "")
    history = state.get("history") if isinstance(state.get("history"), list) else None

    # Langfuse span (if enabled)
    try:
        from observability import get_langfuse

        lf = get_langfuse()
        if lf is not None:
            with lf.start_as_current_observation(
                name="field_collector",
                as_type="span",
                input={"question": question, "history": history},
            ) as _span:
                extracted = extract_divisions(question, history)  # type: ignore[arg-type]
                existing = state.get("divisions")
                if extracted:
                    state["divisions"] = extracted
                elif existing is None:
                    state["divisions"] = []
                if state.get("location") is None:
                    state["location"] = {}
                try:
                    lf.update_current_span(
                        output={"divisions": state.get("divisions"), "location": state.get("location")}
                    )
                except Exception:
                    pass
                return state
    except Exception:
        pass

    extracted = extract_divisions(question, history)  # type: ignore[arg-type]
    existing = state.get("divisions")
    if extracted:
        state["divisions"] = extracted
    elif existing is None:
        state["divisions"] = []
    if state.get("location") is None:
        state["location"] = {}
    return state
