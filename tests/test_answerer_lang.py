"""Tests de idioma del answerer: deteccion ES/EN y prompts bilingües (sin LLM)."""
from __future__ import annotations

import asyncio

from agent.nodes.answerer import (
    NO_ANSWER_EN,
    NO_ANSWER_ES,
    SYSTEM_PROMPT_EN,
    answer,
    detect_question_lang,
    stream_answer_async,
)


# ---------- Deteccion ----------

def test_spanish_with_accents():
    assert detect_question_lang("¿Cuánto cuesta sembrar maíz en Pergamino?") == "es"


def test_spanish_voseo_without_accents():
    assert detect_question_lang("che, contame cuanto sale el novillo") == "es"
    assert detect_question_lang("tenes los costos de la soja?") == "es"


def test_english_questions():
    assert detect_question_lang("What is the maize gross margin for 80 hectares?") == "en"
    assert detect_question_lang("How much does a kilo of feedlot steer cost?") == "en"
    assert detect_question_lang("What are soybean production costs per hectare?") == "en"


def test_ambiguous_defaults_to_spanish():
    # Cortas, mixtas o sin evidencia -> fallback "es" (audiencia argentina)
    assert detect_question_lang("hola") == "es"
    assert detect_question_lang("maiz 80ha?") == "es"
    assert detect_question_lang("") == "es"
    assert detect_question_lang("   ") == "es"


def test_custom_fallback():
    assert detect_question_lang("hola", fallback="en") == "en"


# ---------- Contrato con el front ----------

def test_no_answer_prefixes_match_frontend():
    # page.tsx matchea estos prefijos para no guardar "no encontre" como investigada.
    # Si cambian, actualizar el front en la misma rama.
    assert NO_ANSWER_ES.startswith("En esta edicion no encontre")
    assert NO_ANSWER_EN.startswith("I didn't find")


def test_english_prompt_has_same_hard_rules():
    assert "NEVER INVENT" in SYSTEM_PROMPT_EN
    assert "(p. 38)" in SYSTEM_PROMPT_EN
    assert "FULL RANGE" in SYSTEM_PROMPT_EN
    assert "che" not in SYSTEM_PROMPT_EN.lower()


# ---------- Early-returns sin LLM ----------

def test_answer_no_items_spanish(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    result = answer("cuanto sale sembrar soja?", [])
    assert result["answer"] == NO_ANSWER_ES
    assert result["lang"] == "es"


def test_answer_no_items_english(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    result = answer("What does it cost to plant soybeans?", [])
    assert result["answer"] == NO_ANSWER_EN
    assert result["lang"] == "en"


def test_stream_no_items_english():
    async def collect():
        gen, _usage = await stream_answer_async("How much is corn?", [])
        return [t async for t in gen]

    assert asyncio.run(collect()) == [NO_ANSWER_EN]


def test_stream_no_items_spanish():
    async def collect():
        gen, _usage = await stream_answer_async("Cuanto sale el maiz?", [])
        return [t async for t in gen]

    assert asyncio.run(collect()) == [NO_ANSWER_ES]
