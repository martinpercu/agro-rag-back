"""Tests del extractor de PDF."""
from __future__ import annotations

from ingestion.extractor import _clean_table, _clean_text, table_to_text


def test_clean_text_preserves_paragraphs():
    raw = "Linea 1\n\n\n\nLinea 2\n\n\nLinea 3"
    out = _clean_text(raw)
    assert "\n\n\n" not in out
    assert "Linea 1" in out
    assert "Linea 2" in out


def test_clean_text_strips_outer_whitespace():
    raw = "  Hola  \n  Mundo  "
    out = _clean_text(raw)
    # _clean_text rstrip() por linea + strip() al final del output.
    # Los espacios internos entre lineas se preservan.
    assert out == "Hola\n  Mundo"


def test_clean_table_handles_none():
    raw = [[None, "a", "  "], [None, None, None]]
    out = _clean_table(raw)
    assert out == [["", "a", ""], ["", "", ""]]


def test_table_to_text_aligns_columns():
    table = [
        ["Cultivo", "Rinde", "Costo"],
        ["Soja", "40", "538"],
        ["Maiz", "100", "716"],
    ]
    out = table_to_text(table)
    lines = out.splitlines()
    assert lines[0] == "Cultivo | Rinde | Costo"
    # Cada linea tiene el ancho maximo de su columna (padding a la derecha)
    assert lines[1] == "Soja    | 40    | 538"
    assert lines[2] == "Maiz    | 100   | 716"
    assert "538" in out
    assert "716" in out


def test_table_to_text_empty():
    assert table_to_text([]) == ""
