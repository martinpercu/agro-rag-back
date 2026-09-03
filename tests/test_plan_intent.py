"""Tests plan_intent — rule-based, sin LLM, costo 0."""
from __future__ import annotations

from agent.nodes.plan_intent import is_plan_intent

CASES = [
    ("tengo 120ha, 80 de maiz y 40 de soja, que me conviene sembrar?", True),
    ("tengo 200 hectáreas para soja", True),
    ("tengo un campo de 150 ha, quiero dividir 100 soja y 50 trigo", True),
    ("tengo 80ha maiz y 40ha soja", True),
    ("tengo 120ha que me conviene?", True),
    ("tengo 100ha", True),
    ("que me conviene sembrar en esta zona?", True),
    ("como divido mi campo de 200ha?", True),
    ("que precio tiene el maiz?", False),
    ("hola como andas?", False),
    ("", False),
    ("costo del glifosato", False),
]


def test_is_plan_intent_cases():
    for q, expected in CASES:
        got = is_plan_intent(q)
        assert got is expected, f"q={q!r} -> {got}, esperaba {expected}"


def test_plan_intent_with_history():
    # "tengo" suelto sin ha no es plan, pero con historial de ha sí
    assert is_plan_intent("y de soja?", history=[{"role": "user", "content": "tengo 80ha maiz"}]) is True
    assert is_plan_intent("y de soja?", history=[]) is False


def test_plan_intent_node_sets_state():
    from agent.nodes.plan_intent import plan_intent_node

    state = {"question": "tengo 80ha maiz y 40ha soja", "history": []}
    out = plan_intent_node(state)  # type: ignore[arg-type]
    assert out["plan_intent"] is True

    state2 = {"question": "que precio tiene el maiz?", "history": []}
    out2 = plan_intent_node(state2)  # type: ignore[arg-type]
    assert out2["plan_intent"] is False
