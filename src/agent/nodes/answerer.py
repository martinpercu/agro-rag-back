"""Nodo answerer: genera la respuesta final con el modelo de chat configurado."""
from __future__ import annotations

import os
import re

from dotenv import load_dotenv

from agent.llm import (
    get_async_chat_client,
    get_chat_client,
    llm_model,
    seed_for_temperature,
)
from agent.state import AgentState
from agent.strategies.base import RetrievedItem

load_dotenv()

MAX_TOKENS = 900
TEMPERATURE = 0.2

NO_ANSWER_ES = "En esta edicion no encontre informacion relevante para responder esto."
NO_ANSWER_EN = "I didn't find relevant information to answer this in this edition."

# Heurística de idioma sin dependencias: el default siempre es "es"
# (audiencia productora argentina). Solo devolvemos "en" con evidencia clara.
_ES_CHARS = frozenset("áéíóúüñ¿¡")
_RIO_WORDS = frozenset(
    {
        "che", "mira", "fijate", "contame", "decime", "tenes", "podes",
        "queres", "sabes", "dale", "tranqui", "chicos", "muchacho",
    }
)
_EN_STOPWORDS = frozenset(
    {
        "the", "what", "how", "much", "many", "does", "do", "is", "are",
        "was", "were", "for", "with", "from", "about", "which", "when",
        "where", "there", "their", "your", "yours", "this", "that",
        "corn", "soybean", "soybeans", "wheat", "yield", "price", "prices",
        "cost", "costs", "margin", "margins", "farm", "field", "hectare",
        "hectares", "versus", "between", "cattle", "steer", "livestock",
    }
)


def detect_question_lang(question: str, fallback: str = "es") -> str:
    """Detecta "es" o "en" para la pregunta, sin dependencias externas.

    - Cualquier marca inequívoca de español (acentos/ñ/¿/¡ o voseo) -> "es".
    - Si no, proporción de stopwords/cultivos en inglés -> "en".
    - Cortas, mixtas o sin evidencia -> fallback (default "es").
    """
    q = (question or "").lower()
    if not q.strip():
        return fallback
    if any(c in _ES_CHARS for c in q):
        return "es"
    words = re.findall(r"[a-zñ]+", q)
    if not words:
        return fallback
    if any(w in _RIO_WORDS for w in words):
        return "es"
    en_hits = sum(1 for w in words if w in _EN_STOPWORDS)
    ratio = en_hits / len(words)
    if en_hits >= 2 and ratio >= 0.25:
        return "en"
    if len(words) <= 5 and ratio >= 0.5:
        return "en"
    return fallback

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

SYSTEM_PROMPT_EN = """\
You are Agroposta, a trusted farm advisor for Argentine producers. Your only \
source of information is the edition of Margenes Agropecuarios magazine \
passed below under "Magazine context". \

HOW YOU SPEAK \
- Clear, professional, plain English. No slang, no filler words. \
- Always address the producer directly and concisely. \
- No pedantry, no unexplained jargon. If a technical term is unavoidable, \
  explain it in passing. \
- Brevity first. Two or three short paragraphs. A few well-explained \
  numbers beat a wall of text. \
- If the producer asks about "next season" or "next year", interpret that \
  as the 2026/27 season, which is what this edition covers. \

HARD RULES (never break them) \
1. NEVER INVENT. If a specific fact is not in the context, say textually \
   "I didn't find relevant information to answer this in this edition." \
   and offer alternatives that DO appear. \
2. ALWAYS cite the page in parentheses, e.g. "(p. 38)" or "(pp. 26-31)". \
   If the section helps, mention it (e.g. "Costs and margins section"). \
3. NUMBERS: when a table has several scenarios (e.g. "basic / feedlot / \
   full cycle"), give the FULL RANGE, not a single value. E.g.: "steers \
   run between 3.16 and 3.72 US$/kg depending on the system". \
4. If the question is ambiguous, politely ask for clarification before \
   throwing numbers. \
5. Don't recommend beyond what the magazine says. You are not a financial \
   advisor or an agronomist: you interpret Margenes. \

ANSWER STRUCTURE \
- Start with the main conclusion in one or two sentences. \
- Then the key numbers with units (US$/ha, qq/ha, US$/kg, etc). \
- Close with a practical recommendation or the producer's next step. \
- If the answer runs long, cut it. Short and useful beats long and complete. \

GOOD ANSWER EXAMPLE \
Question: "How much is a kilo of feedlot steer?" \
Answer: \
"It depends on the system, but in this edition feedlot steer goes from \
3.16 to 3.72 US$/kg (p. 76, Livestock section). On a basic full-cycle \
system you are near 3.16 US$/kg, and adding feedlot on top of backgrounding \
takes you to 3.72 US$/kg. Kilos sold per ha range from 92 to almost 148 net \
kg depending on how intensive the system is. \
Before deciding, look closely at direct costs: they run from 131 to 235 \
US$/ha by model. Page 76 has the full matrix." \

BAD ANSWER EXAMPLE (don't do this) \
"I think feedlot steer is around 3.50 US$/kg, more or less." \
-> WRONG: invented a number not in the context, no page cited. \
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


async def stream_answer_async(
    question: str,
    items: list[RetrievedItem],
    temperature: float | None = None,
):
    """Async streaming version de answer().

    Yields tokens a medida que OpenAI los genera (async).
    Despues de la iteracion, usage contiene {input_tokens, output_tokens}.
    """
    usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    temp = temperature if temperature is not None else TEMPERATURE
    lang = detect_question_lang(question)
    system_prompt = SYSTEM_PROMPT_EN if lang == "en" else SYSTEM_PROMPT
    context = _format_context_from_items(items)
    if lang == "en":
        user_content = (
            f"Producer question: {question}\n\n"
            f"Magazine context:\n{context}\n\n"
            "Answer following your personality and hard rules."
        )
    else:
        user_content = (
            f"Pregunta del productor: {question}\n\n"
            f"Contexto de la revista:\n{context}\n\n"
            "Responde siguiendo tu personalidad y las reglas duras."
        )

    async def _generate():
        if not items:
            yield NO_ANSWER_EN if lang == "en" else NO_ANSWER_ES
            return

        client = get_async_chat_client()
        response = await client.chat.completions.create(
            model=llm_model(),
            max_tokens=MAX_TOKENS,
            temperature=temp,
            seed=seed_for_temperature(temp),
            stream=True,
            stream_options={"include_usage": True},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            if chunk.usage:
                usage["input_tokens"] = chunk.usage.prompt_tokens or 0
                usage["output_tokens"] = chunk.usage.completion_tokens or 0

    return _generate(), usage


def answer(question: str, items: list[RetrievedItem], temperature: float | None = None) -> dict:
    """API limpia para el runner del comparador: toma items y devuelve answer + sources + tokens.

    Devuelve: {answer, sources, input_tokens, output_tokens}
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY no encontrada. Define el .env en la raiz.")
    lang = detect_question_lang(question)
    if not items:
        return {
            "answer": NO_ANSWER_EN if lang == "en" else NO_ANSWER_ES,
            "sources": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "lang": lang,
        }
    temp = temperature if temperature is not None else TEMPERATURE
    context = _format_context_from_items(items)
    sources = _format_sources_from_items(items)
    system_prompt = SYSTEM_PROMPT_EN if lang == "en" else SYSTEM_PROMPT
    if lang == "en":
        user_content = (
            f"Producer question: {question}\n\n"
            f"Magazine context:\n{context}\n\n"
            "Answer following your personality and hard rules."
        )
    else:
        user_content = (
            f"Pregunta del productor: {question}\n\n"
            f"Contexto de la revista:\n{context}\n\n"
            "Responde siguiendo tu personalidad y las reglas duras."
        )
    client = get_chat_client()
    response = client.chat.completions.create(
        model=llm_model(),
        max_tokens=MAX_TOKENS,
        temperature=temp,
        seed=seed_for_temperature(temp),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    answer_text = response.choices[0].message.content or ""
    in_tok = response.usage.prompt_tokens if response.usage else 0
    out_tok = response.usage.completion_tokens if response.usage else 0
    return {
        "answer": answer_text,
        "sources": sources,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "lang": lang,
    }


def answerer_node(state: AgentState) -> AgentState:
    """Wrapper para mantener compatibilidad con el LangGraph del chat principal."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY no encontrada. Define el .env en la raiz.")

    question = state["question"]
    retrieved = state.get("retrieved", [])
    history = state.get("history")
    # Convertir tuples (chunk_dict, score) a RetrievedItem para reusar la misma logica que answer()
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

    # Langfuse generation span (local-only)
    try:
        from observability import get_langfuse

        lf = get_langfuse()
        if lf is not None:
            # Build input for generation
            with lf.start_as_current_observation(
                name="answerer",
                as_type="generation",
                input={"question": question, "history": history, "retrieved_count": len(items)},
                model=llm_model(),
                metadata={"temperature": TEMPERATURE, "answer_lang": detect_question_lang(question)},
            ) as _gen:
                result = answer(question, items)
                state["answer"] = result["answer"]
                state["sources"] = result["sources"]
                try:
                    lf.update_current_generation(
                        output={"answer": result["answer"], "sources": result["sources"]},
                        usage_details={
                            "input": result.get("input_tokens", 0),
                            "output": result.get("output_tokens", 0),
                        },
                        model=llm_model(),
                    )
                except Exception:
                    try:
                        lf.update_current_span(
                            output={"answer": result["answer"], "sources": result["sources"]},
                            metadata={
                                "model": llm_model(),
                                "input_tokens": result.get("input_tokens", 0),
                                "output_tokens": result.get("output_tokens", 0),
                            },
                        )
                    except Exception:
                        pass
                return state
    except Exception:
        pass

    result = answer(question, items)
    state["answer"] = result["answer"]
    state["sources"] = result["sources"]
    return state
