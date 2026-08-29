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
            # English
            "cattle",
            "cow",
            "cows",
            "bull",
            "heifer",
            "steer",
            "dairy",
            "beef",
            "meat",
            "calf",
            "calves",
            "ranch",
            "rancher",
            "livestock",
            "pasture",
            "grazing",
            "stocker",
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
            # English
            "herbicide",
            "fungicide",
            "insecticide",
            "pesticide",
            "glyphosate",
            "weed",
            "weeds",
            "resistant",
            "resistance",
            "seed",
            "seeds",
            "fertilizer",
            "fertilizers",
            "spray",
            "spraying",
            "technology",
            "input",
            "inputs",
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
            # English
            "cost",
            "costs",
            "margin",
            "margins",
            "rent",
            "rental",
            "expense",
            "expenses",
            "income",
            "budget",
            "investment",
            "profit",
            "profitability",
            "dollars",
            "usd",
            "tillage",
            "harvest",
            "rate",
            "rates",
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
            # English
            "projection",
            "projections",
            "forecast",
            "outlook",
            "next season",
            "upcoming season",
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
            # English
            "market",
            "price",
            "prices",
            "export",
            "exports",
            "future",
            "futures",
            "board",
            "quote",
            "quotes",
            "fas",
            "withholding",
        )
    ):
        return "mercado"

    # Siembras / porcentajes / actividades de cultivo
    if any(
        w in q
        for w in (
            "siembra",
            "siembras",
            "sembrar",
            "sembrado",
            "sembrada",
            "sembrando",
            "siembro",
            "cultivar",
            "cultivando",
            "cultivado",
            "cultivo",
            "cultivos",
            "plantar",
            "plantado",
            "plantada",
            "plantando",
            "plantacion",
            "plantación",
            "cosechar",
            "cosechando",
            "cosechadora",
            "porcentaje",
            "a porcentaje",
            "campo propio",
            "campo arrendado",
            "lote",
            "lotes",
            "chacra",
            "chacras",
            "huerta",
            "huertas",
            # English
            "sow",
            "sowing",
            "plant",
            "planting",
            "planted",
            "grow",
            "growing",
            "cultivate",
            "crop",
            "crops",
            "planting",
            "acreage",
            "percentage",
            "own land",
            "rented land",
            "crop area",
            "field",
            "fields",
        )
    ):
        return "siembras"

    return "general"


def classifier_node(state: AgentState) -> AgentState:
    question = state["question"]
    state["intent"] = _classify(question)
    return state


def is_off_topic(question: str) -> bool:
    """Determina si la pregunta NO es sobre temas agropecuarios.

    Corre el clasificador existente primero (gratis, rule-based).
    Si devuelve algo distinto de "general" -> es agro.
    Si es "general" -> check extra con palabras clave agro.
    """
    intent = _classify(question)
    if intent != "general":
        return False

    q = question.lower()
    for kw in (
        "campo",
        "cultivo",
        "clima",
        "lluvia",
        "sequia",
        "sequía",
        "revista",
        "edicion",
        "edición",
        "margenes",
        "márgenes",
        "chacra",
        "productor",
        "kg",
        "kilo",
        "tonelada",
        "hectarea",
        "hectárea",
        "ha ",
        "qq ",
        "usd",
        "dolar",
        "dólar",
        "plaga",
        "enfermedad",
        # Cultivos
        "soja",
        "trigo",
        "maiz",
        "maíz",
        "girasol",
        "cebada",
        "sorgo",
        "colza",
        "alpiste",
        "avena",
        "centeno",
        "arroz",
        "poroto",
        "porotos",
        "mani",
        "maní",
        "algodon",
        "algodón",
        "lino",
        "lenteja",
        "garbanzo",
        "cartamo",
        "cártamo",
        # English
        "field",
        "farm",
        "farmer",
        "crop",
        "weather",
        "rain",
        "drought",
        "hail",
        "magazine",
        "ton",
        "tonne",
        "hectare",
        "soybean",
        "wheat",
        "corn",
        "maize",
        "sunflower",
        "barley",
        "sorghum",
        "canola",
        "rapeseed",
        "oats",
        "rye",
        "rice",
        "cotton",
        "flax",
        "peanut",
        "chickpea",
        "lentil",
        "pea",
        "beans",
        "potato",
        "sugarcane",
    ):
        if kw in q:
            return False

    return True
