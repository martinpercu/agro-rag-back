"""Nodo clasificador: detecta la intencion de la pregunta para guiar el retrieval."""
from __future__ import annotations

from agent.state import AgentState, Intent


def _classify(question: str) -> Intent:
    """Clasificador rule-based. Liviano, sin LLM, para no gastar tokens en algo
    que podemos resolver con heuristica.

    Pensado para preguntas cortas y coloquiales de un productor argentino:
    "costos de soja", "que se proyecta para trigo", "precio del girasol", etc.
    """
    q = question.lower()

    # Ganaderia: chequear ANTES de costos, porque "novillo cuesta X" o
    # "kilo de feedlot" pueden matchear "cuesta/kilo" pero la intencion
    # real es ganaderia.
    if any(
        w in q
        for w in (
            "vaca",
            "vacas",
            "novillo",
            "novillos",
            "vaquillona",
            "vaquillonas",
            "tambo",
            "feedlot",
            "invernada",
            "cria",
            "cría",
            "rodeo",
            "pasto",
            "pastura",
            "rollos",
            "silo",
            "silaje",
            "kgvivo",
            "kg vivo",
            "kilo vivo",
            "ganaderia",
            "ganadería",
            "carne",
            "kg de carne",
        )
    ):
        return "ganaderia"

    # Tecnologia / insumos / malezas: tambien antes de costos porque
    # "glifosato cuesta X" es tecnologia, no costos.
    if any(
        w in q
        for w in (
            "tecnologia",
            "tecnología",
            "insumo",
            "insumos",
            "agroquimico",
            "agroquímico",
            "herbicida",
            "fungicida",
            "insecticida",
            "glifosato",
            "maleza",
            "malezas",
            "resistente",
            "resistencia",
            "semilla",
            "semillas",
            "fertilizante",
            "fertilizantes",
            "pulverizacion",
            "pulverización",
        )
    ):
        return "tecnologia"

    # Costos / margenes / arrendamientos / estructura
    if any(
        w in q
        for w in (
            "costo",
            "costos",
            "cuesta",
            "cuestan",
            "sale",
            "salir",
            "vale",
            "salida",
            "inversion",
            "inversión",
            "plata",
            "guita",
            "pesos",
            "dolares",
            "dólares",
            "margen",
            "margenes",
            "márgen",
            "márgenes",
            "rentabilidad",
            "resultado neto",
            "gastos",
            "capital",
            "arrendamiento",
            "arrendamientos",
            "qq/ha",
            "qqs",
            "qq por",
            "us$/ha",
            "u$s/ha",
            "estructura",
            "labranza",
            "siembra directa",
            "fertilizacion",
            "fertilización",
            "cosecha",
            "tarifa",
            "tarifas",
            "trilla",
        )
    ):
        return "costos"

    # Proyecciones / campana proxima
    if any(
        w in q
        for w in (
            "proyeccion",
            "proyecciones",
            "campana que viene",
            "campaña que viene",
            "el año que viene",
            "proxima campana",
            "próxima campaña",
            "2026/27",
            "2026 / 27",
            "2027",
        )
    ):
        return "proyecciones"

    # Mercado / precios / fas / exportacion
    if any(
        w in q
        for w in (
            "precio",
            "precios",
            "mercado",
            "fas",
            "fas teorico",
            "fas teórico",
            "exportacion",
            "exportaciones",
            "futuro",
            "disponible",
            "pizarra",
            "cotizacion",
            "cotización",
            "retenciones",
        )
    ):
        return "mercado"

    # Siembras / porcentajes
    if any(
        w in q
        for w in (
            "siembra",
            "siembras",
            "porcentaje",
            "a porcentaje",
            "campo propio",
            "campo arrendado",
        )
    ):
        return "siembras"

    return "general"


def classifier_node(state: AgentState) -> AgentState:
    question = state["question"]
    state["intent"] = _classify(question)
    return state
