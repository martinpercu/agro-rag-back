"""Tests del chunker."""
from __future__ import annotations

from ingestion.chunker import (
    _detect_campana,
    _detect_cultivo,
    _detect_section,
    chunk_pages,
)


def test_detect_section_costos(sample_pages):
    p = sample_pages[0]
    assert _detect_section(p.text) == "costos_margenes"


def test_detect_section_tecnologia(sample_pages):
    p = sample_pages[2]
    assert _detect_section(p.text) == "tecnologia"


def test_detect_section_ganaderia(sample_pages):
    p = sample_pages[3]
    assert _detect_section(p.text) == "ganaderia_costos"


def test_detect_cultivo_trigo(sample_pages):
    p = sample_pages[0]
    assert _detect_cultivo(p.text, "costos_margenes") == "trigo"


def test_detect_cultivo_soja(sample_pages):
    p = sample_pages[1]
    assert _detect_cultivo(p.text, "costos_margenes") == "soja"


def test_detect_cultivo_returns_none_for_general(sample_pages):
    p = sample_pages[0]
    assert _detect_cultivo(p.text, "general") is None


def test_detect_campana(sample_pages):
    p = sample_pages[0]
    # "2027" aparece en "PRECIO A COSECHA 2027" pero el pattern exige el
    # formato "YYYY/YY". Para que matchee, el sample page debe tener
    # la campana explicita.
    assert _detect_campana(p.text) is None  # no hay 2026/27 explicito


def test_detect_campana_when_explicit():
    text = "TRIGO: COSTOS y MARGENES 2026/27\nZONA OESTE"
    assert _detect_campana(text) == "2026_27"


def test_detect_campana_2025_26():
    text = "PROYECCIONES 2025/26 exportaciones trigo"
    assert _detect_campana(text) == "2025_26"


def test_chunk_pages_generates_one_chunk_per_page(sample_pages):
    chunks = chunk_pages(sample_pages, edition="2026_05")
    assert len(chunks) == len(sample_pages)
    for c, p in zip(chunks, sample_pages):
        assert c.metadata.pagina == p.page_number


def test_chunk_pages_dedupes_identical_text():
    from schemas import PageContent

    text = "TRIGO: COSTOS y MARGENES\nRENDIMIENTOS QQ/ha 40"
    pages = [
        PageContent(page_number=1, text=text, tables=[[["A", "B"], ["1", "2"]]]),
        PageContent(page_number=2, text=text, tables=[[["A", "B"], ["1", "2"]]]),
    ]
    chunks = chunk_pages(pages, edition="2026_05")
    # Aunque las dos paginas tienen el mismo texto y tabla, deberian
    # producir 2 chunks (paginas distintas = chunks distintos).
    assert len(chunks) == 2


def test_chunk_with_table_has_tipo_tabla(sample_pages):
    chunks = chunk_pages(sample_pages, edition="2026_05")
    tabla_chunks = [c for c in chunks if c.metadata.tipo == "tabla"]
    assert len(tabla_chunks) >= 1
    for c in tabla_chunks:
        assert c.metadata.seccion != "general" or "Tabla" in c.text
