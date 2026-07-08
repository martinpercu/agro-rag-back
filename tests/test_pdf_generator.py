"""Tests del generador de PDF."""
from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader

from export.pdf_generator import _safe, render_conversation_pdf


def test_safe_replaces_smart_quotes():
    assert _safe("hola\u201cmundo\u201d") == 'hola"mundo"'


def test_safe_replaces_em_dash():
    assert _safe("a\u2014b") == "a-b"


def test_safe_replaces_unicode_ellipsis():
    assert _safe("esperando\u2026") == "esperando..."


def test_safe_handles_empty_string():
    assert _safe("") == ""


def test_safe_handles_none():
    assert _safe(None) == ""


def test_render_conversation_pdf_basic():
    messages = [
        {"role": "user", "content": "Cuanto sale la soja?"},
        {
            "role": "assistant",
            "content": "Sale 331 US$/tn (pag. 38).",
            "sources": [
                {
                    "pagina": 38,
                    "seccion": "costos_margenes",
                    "cultivo": "soja",
                    "campana": "2026_27",
                    "tipo": "tabla",
                    "score": 0.57,
                }
            ],
        },
    ]
    pdf_bytes = render_conversation_pdf(messages, edition="2026_05")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 100
    assert pdf_bytes[:4] == b"%PDF"

    reader = PdfReader(BytesIO(pdf_bytes))
    assert len(reader.pages) >= 1
    text = reader.pages[0].extract_text()
    assert "Agroposta" in text
    assert "Productor:" in text
    assert "Agroposta:" in text
    assert "331" in text
    assert "pag. 38" in text


def test_render_conversation_pdf_no_sources():
    messages = [
        {"role": "user", "content": "Hola?"},
        {"role": "assistant", "content": "Hola productor."},
    ]
    pdf_bytes = render_conversation_pdf(messages, edition="2026_05")
    reader = PdfReader(BytesIO(pdf_bytes))
    text = reader.pages[0].extract_text()
    assert "Hola productor" in text
    # Sin sources: no aparece la seccion "Fuentes citadas"
    assert "Fuentes citadas" not in text
