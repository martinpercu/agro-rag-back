"""Nodo answerer: genera la respuesta final con gpt-4.1-nano + system prompt rioplatense."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from agent.state import AgentState
from agent.strategies.base import RetrievedItem

load_dotenv()

MODEL = "gpt-4.1-nano"
MAX_TOKENS = 900
TEMPERATURE = 0.2

SYSTEM_PROMPT = """\
Sos Agroposta, un consejero agropecuario de confianza para productores \
argentinos. Tu unica fuente de informacion es la edicion de la revista \
Margenes Agropecuarios que te paso abajo en "Contexto de la revista". \
\
COMO HABLAS \
- Rioplatense campechano y natural. Usas "che", "mira", "dale", "fijate", \
  "el muchacho" con moderacion y solo cuando viene bien al caso. \
- Tuteo siempre con el productor. \
- Nada de pedanteria ni latinajos. Si un termino tecnico es inevitable, \
  lo explicas al pasar. \
- Brevedad ante todo. Dos o tres parrafos cortos. Mejor pocos numeros \
  bien explicados que un chorizo. \
- Si el productor pregunta por "la campana que viene" o "el año que viene", \
  interpreta eso como la campana 2026/27 que es sobre la que habla esta \
  edicion. \
\
REGLAS DURAS (jamas las rompas) \
1. NO INVENTES NUNCA. Si un dato puntual no aparece en el contexto, decis \
   textualmente "en esta edicion no encontre ese dato" y offerse \
   alternativas que SI aparezcan. \
2. CITAS siempre la pagina entre parentesis, ej "(pag. 38)" o "(pag. 26-31)". \
   Si la seccion ayuda, mencionala (ej "seccion Costos y margenes"). \
3. NUMEROS: cuando una tabla tiene varios planteos o escenarios (por ej \
   "basico / feedlot / ciclo completo"), da el RANGO completo, no te quedes \
   con un solo valor. Ej: "el novillo esta entre 3,16 y 3,72 US$/kg segun \
   el planteo". \
4. Si la consulta es ambigua, pregunta amablemente que aclara antes de \
   tirar numeros. \
5. No hagas recomendaciones que vayan mas alla de lo que dice la revista. \
   No sos asesor financiero ni agronomo: sos un interprete de Margenes. \
\
ESTRUCTURA DE LA RESPUESTA \
- Arrancas con la conclusion principal en una o dos frases. \
- Despues los numeros clave con su unidad (US$/ha, qq/ha, US$/kg, etc). \
- Cerras con una recomendacion practica o el siguiente paso para el productor. \
- Si la respuesta sale larga, corta. Mejor corto y util que largo y completo. \
\
EJEMPLO DE RESPUESTA BUENA \
Pregunta: "Cuanto cuesta un kilo de novillo en feedlot?" \
Respuesta: \
"Depende del planteo, pero el kilo de novillo en esta edicion va de 3,16 \
a 3,72 US$/kg (pag. 76, seccion Ganaderia). En ciclo completo basico \
andas cerca de 3,16 US$/kg, y si sumas feedlot sobre la recr. te vas \
a 3,72 US$/kg. Los kilos vendidos por ha van de 92 a casi 148 kg netos \
segun que tan intensivo sea el planteo. \
Antes de decidir, mira bien los costos directos: van de 131 a 235 US$/ha \
segun el modelo. Fijate en la pagina 76 que esta toda la matriz." \
\
EJEMPLO DE RESPUESTA MALA (no hacer esto) \
"Yo creo que el kilo de novillo esta en 3,50 US$/kg mas o menos." \
-> MAL: inventar un numero que no estaba en el contexto, no citar pagina. \
"""


def _format_context(retrieved: list[tuple[dict, float]]) -> str:
    """Formatea los chunks recuperados para mandarlos al LLM."""
    if not retrieved:
        return "(Sin contexto: el vector store no devolvio resultados.)"
    parts: list[str] = []
    for i, (chunk, score) in enumerate(retrieved, 1):
        meta = chunk["metadata"]
        seccion = meta.get("seccion", "?")
        tipo = meta.get("tipo", "?")
        cultivo = meta.get("cultivo") or "sin cultivo especifico"
        campana = meta.get("campana") or "sin campana especifica"
        pagina = meta.get("pagina", "?")
        text = chunk["text"]
        parts.append(
            f"[Fragmento {i} | seccion={seccion} | tipo={tipo} | "
            f"cultivo={cultivo} | campana={campana} | pag. {pagina} | "
            f"relevancia={score:.2f}]\n{text}"
        )
    return "\n\n---\n\n".join(parts)


def _format_sources(retrieved: list[tuple[dict, float]]) -> list[dict]:
    """Devuelve la lista de fuentes citadas para mostrarla y para el PDF."""
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for chunk, score in retrieved:
        meta = chunk["metadata"]
        key = (meta.get("seccion", "?"), meta.get("pagina", 0))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "seccion": meta.get("seccion", "?"),
                "pagina": meta.get("pagina", 0),
                "cultivo": meta.get("cultivo"),
                "campana": meta.get("campana"),
                "tipo": meta.get("tipo"),
                "score": round(score, 3),
            }
        )
    return out


def _format_context_from_items(items: list[RetrievedItem]) -> str:
    """Idem _format_context pero trabaja sobre RetrievedItem (no tuples del state)."""
    if not items:
        return "(Sin contexto: el vector store no devolvio resultados.)"
    parts: list[str] = []
    for i, item in enumerate(items, 1):
        seccion = item.seccion or "?"
        tipo = item.tipo or "?"
        cultivo = item.cultivo or "sin cultivo especifico"
        campana = item.campana or "sin campana especifica"
        pagina = item.pagina if item.pagina is not None else "?"
        text = item.text
        parts.append(
            f"[Fragmento {i} | seccion={seccion} | tipo={tipo} | "
            f"cultivo={cultivo} | campana={campana} | pag. {pagina} | "
            f"relevancia={item.score:.2f}]\n{text}"
        )
    return "\n\n---\n\n".join(parts)


def _format_sources_from_items(items: list[RetrievedItem]) -> list[dict]:
    """Idem _format_sources pero sobre RetrievedItem."""
    seen: set[tuple[str | None, int | None]] = set()
    out: list[dict] = []
    for item in items:
        key = (item.seccion, item.pagina)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "seccion": item.seccion,
                "pagina": item.pagina,
                "cultivo": item.cultivo,
                "campana": item.campana,
                "tipo": item.tipo,
                "score": round(item.score, 3),
            }
        )
    return out


def _call_openai(question: str, context: str) -> tuple[str, int, int]:
    """Llamada comun al LLM. Devuelve (answer, in_tok, out_tok)."""
    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Pregunta del productor: {question}\n\n"
                    f"Contexto de la revista:\n{context}\n\n"
                    "Responde siguiendo tu personalidad y las reglas duras."
                ),
            },
        ],
    )
    answer = response.choices[0].message.content or ""
    in_tok = response.usage.prompt_tokens if response.usage else 0
    out_tok = response.usage.completion_tokens if response.usage else 0
    return answer, in_tok, out_tok


def answer(question: str, items: list[RetrievedItem]) -> dict:
    """API limpia para el runner del comparador: toma items y devuelve answer + sources + tokens.

    Devuelve: {answer, sources, input_tokens, output_tokens}
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY no encontrada. Define el .env en la raiz.")
    if not items:
        return {
            "answer": "En esta edicion no encontre informacion relevante para responder esto.",
            "sources": [],
            "input_tokens": 0,
            "output_tokens": 0,
        }
    context = _format_context_from_items(items)
    sources = _format_sources_from_items(items)
    answer_text, in_tok, out_tok = _call_openai(question, context)
    return {
        "answer": answer_text,
        "sources": sources,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }


def answerer_node(state: AgentState) -> AgentState:
    """Wrapper para mantener compatibilidad con el LangGraph del chat principal."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY no encontrada. Define el .env en la raiz.")

    question = state["question"]
    retrieved = state.get("retrieved", [])
    # Convertir tuples (chunk_dict, score) a RetrievedItem para reusar
    # la misma logica que answer()
    items: list[RetrievedItem] = []
    for chunk_dict, score in retrieved:
        meta = chunk_dict.get("metadata", {})
        items.append(
            RetrievedItem(
                chunk_id=chunk_dict.get("id", ""),
                text=chunk_dict.get("text", ""),
                seccion=meta.get("seccion"),
                pagina=meta.get("pagina"),
                cultivo=meta.get("cultivo"),
                campana=meta.get("campana"),
                tipo=meta.get("tipo"),
                score=float(score),
                rank=0,
            )
        )
    result = answer(question, items)
    state["answer"] = result["answer"]
    state["sources"] = result["sources"]
    return state
