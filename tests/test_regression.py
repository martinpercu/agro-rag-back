"""Test de regression sobre golden_questions.json.

Corre las preguntas doradas contra el agente real y valida:
- intent detectado coincide con expected_intent
- la respuesta contiene al menos uno de los expected_answer_contains
- la respuesta NO contiene ninguno de los expected_answer_not_contains
- al menos una fuente pertenece a expected_sections_any
- si expected_cultivo_any no es null, al menos una fuente tiene ese cultivo
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

GOLDEN_PATH = ROOT / "tests" / "golden_questions.json"


def _load_golden() -> list[dict]:
    with open(GOLDEN_PATH) as f:
        return json.load(f)


GOLDEN = _load_golden()


def _check_contains(answer: str, options: list[str]) -> tuple[bool, str | None]:
    """Devuelve (True, match) si al menos uno de los options aparece en answer (case insensitive)."""
    if not options:
        return True, None
    a = answer.lower()
    for opt in options:
        if opt.lower() in a:
            return True, opt
    return False, None


def _check_not_contains(answer: str, options: list[str]) -> tuple[bool, str | None]:
    """Devuelve (True, None) si NINGUNO de los options aparece en answer."""
    if not options:
        return True, None
    a = answer.lower()
    for opt in options:
        if opt.lower() in a:
            return False, opt
    return True, None


def _check_sections(sources: list[dict], allowed: list[str]) -> bool:
    if not allowed:
        return True
    for s in sources:
        if s.get("seccion") in allowed:
            return True
    return False


def _check_cultivo(sources: list[dict], expected: list | None) -> bool:
    if expected is None:
        return True
    if not expected:  # lista vacia = no se exige
        return True
    cultivos = {s.get("cultivo") for s in sources}
    for c in expected:
        if c in cultivos:
            return True
    return False


@pytest.mark.skipif(
    "not config.getoption('--integration')",
    reason="requiere --integration y OPENAI_API_KEY",
)
@pytest.mark.parametrize("case", GOLDEN, ids=[c["id"] for c in GOLDEN])
def test_golden_question(case, has_openai_key):
    if not has_openai_key:
        pytest.skip("OPENAI_API_KEY no configurada")

    from agent.graph import graph

    t0 = time.time()
    result = graph.invoke({"question": case["question"]})
    elapsed = time.time() - t0

    # 1. Intent
    assert result.get("intent") == case["expected_intent"], (
        f"intent esperado {case['expected_intent']}, "
        f"obtenido {result.get('intent')} "
        f"(pregunta: {case['question']!r})"
    )

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    # 2. Answer contiene al menos uno de los esperados
    ok, match = _check_contains(answer, case["expected_answer_contains"])
    if not ok:
        pytest.fail(
            f"answer no contiene ninguno de {case['expected_answer_contains']}\n"
            f"answer: {answer[:400]}\n"
            f"pregunta: {case['question']!r}"
        )

    # 3. Answer NO contiene ninguno de los prohibidos
    ok, bad = _check_not_contains(answer, case["expected_answer_not_contains"])
    if not ok:
        pytest.fail(
            f"answer contiene termino prohibido: {bad!r}\n"
            f"answer: {answer[:400]}\n"
            f"pregunta: {case['question']!r}"
        )

    # 4. Al menos una fuente de las secciones esperadas
    if not _check_sections(sources, case["expected_sections_any"]):
        secciones = [s.get("seccion") for s in sources]
        pytest.fail(
            f"ninguna fuente esta en {case['expected_sections_any']}, "
            f"secciones obtenidas: {secciones}\n"
            f"pregunta: {case['question']!r}"
        )

    # 5. Al menos una fuente con el cultivo esperado (si se especifico)
    if not _check_cultivo(sources, case.get("expected_cultivo_any")):
        cultivos = [s.get("cultivo") for s in sources]
        pytest.fail(
            f"ninguna fuente tiene cultivo en {case['expected_cultivo_any']}, "
            f"cultivos obtenidos: {cultivos}\n"
            f"pregunta: {case['question']!r}"
        )

    # Sanity: la respuesta no debe ser vacia ni muy corta
    assert len(answer) > 30, f"answer demasiado corto: {answer!r}"

    # Logueamos latencia para detectar regresiones de performance
    print(f"\n[{case['id']}] intent={result['intent']} | {elapsed:.1f}s | sources={len(sources)}")
