"""Tests field_collector — Decimal ha/cultivo, costo 0."""
from __future__ import annotations

from agent.nodes.field_collector import extract_divisions, field_collector_node


def test_single_ha_with_cultivo():
    assert extract_divisions("tengo 200 hectáreas para soja") == [
        {"hectares": "200", "cultivo": "soja"}
    ]


def test_two_ha_explicit():
    assert extract_divisions("tengo 80ha maiz y 40ha soja") == [
        {"hectares": "80", "cultivo": "maiz"},
        {"hectares": "40", "cultivo": "soja"},
    ]


def test_loose_numbers_with_ha_context():
    # 100 y 50 sin "ha" pero con cultivo y un ha global -> se capturan
    divs = extract_divisions("tengo un campo de 150 ha, quiero dividir 100 soja y 50 trigo")
    # Debe contener 150, 100, 50 con cultivos correspondientes
    assert {"hectares": "100", "cultivo": "soja"} in divs
    assert {"hectares": "50", "cultivo": "trigo"} in divs
    assert any(d["hectares"] == "150" for d in divs)


def test_decimal_comma():
    assert extract_divisions("tengo 80,5 ha soja") == [
        {"hectares": "80.5", "cultivo": "soja"}
    ]


def test_cultivo_with_accent_normalized():
    divs = extract_divisions("tengo 100ha maíz")
    assert divs[0]["cultivo"] == "maiz"


def test_no_ha_no_divisions():
    assert extract_divisions("que precio tiene el maiz?") == []
    assert extract_divisions("") == []


def test_history_context():
    divs = extract_divisions("y de soja?", history=[{"role": "user", "content": "tengo 80ha maiz"}])
    # history aporta el 80ha
    assert {"hectares": "80", "cultivo": "maiz"} in divs


def test_field_collector_node_merges():
    state = {"question": "tengo 80ha maiz y 40ha soja", "history": []}
    out = field_collector_node(state)  # type: ignore[arg-type]
    assert out["divisions"] == [
        {"hectares": "80", "cultivo": "maiz"},
        {"hectares": "40", "cultivo": "soja"},
    ]
    assert out["location"] == {}

    # No pisa divisions existentes con [] si ya había
    state2 = {"question": "hola", "history": [], "divisions": [{"hectares": "100", "cultivo": "soja"}]}
    out2 = field_collector_node(state2)  # type: ignore[arg-type]
    assert out2["divisions"] == [{"hectares": "100", "cultivo": "soja"}]

    # Si venía vacío, queda []
    state3 = {"question": "hola", "history": []}
    out3 = field_collector_node(state3)  # type: ignore[arg-type]
    assert out3["divisions"] == []
