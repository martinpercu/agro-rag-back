"""Tests del clasificador de intencion."""
from __future__ import annotations

from agent.nodes.classifier import _classify


CASES = [
    # (query, expected_intent)
    ("Cuanto me sale sembrar soja de primera en zona norte?", "costos"),
    ("Cual es el margen bruto del maiz tardio?", "costos"),
    ("Cuanto cuesta un kilo de novillo en feedlot?", "ganaderia"),
    ("Cuanto me sale un kilo de carne?", "ganaderia"),
    ("Que se proyecta para trigo en la campana 2026/27?", "proyecciones"),
    ("Cual es el FAS teorico del girasol?", "mercado"),
    ("Que precio tiene la soja disponible?", "mercado"),
    ("Cual es la relacion insumo producto del glifosato?", "tecnologia"),
    ("Que herbicida uso para rama negra?", "tecnologia"),
    ("Cuales son los arrendamientos en zona nucleo?", "costos"),
    ("Hicimos la siembra a porcentaje, como me fue?", "siembras"),
    ("Que pasa con el clima y la humedad del suelo?", "general"),
]


def test_intent_classification():
    for query, expected in CASES:
        got = _classify(query)
        assert got == expected, f"query={query!r} -> {got}, esperaba {expected}"


def test_empty_query():
    assert _classify("") == "general"


def test_case_insensitive():
    assert _classify("COSTOS DE SOJA") == "costos"
    assert _classify("MARGEN BRUTO") == "costos"
