"""Tests del indexer: sanitize_metadata y busqueda."""
from __future__ import annotations

import pytest

from ingestion.indexer import _sanitize_metadata


def test_sanitize_strips_none():
    meta = {"a": "x", "b": None, "c": "y"}
    out = _sanitize_metadata(meta)
    assert out == {"a": "x", "c": "y"}


def test_sanitize_strips_empty_string():
    meta = {"a": "x", "b": "", "c": "y"}
    out = _sanitize_metadata(meta)
    assert out == {"a": "x", "c": "y"}


def test_sanitize_keeps_falsy_numbers():
    meta = {"a": 0, "b": False, "c": ""}
    out = _sanitize_metadata(meta)
    assert out == {"a": 0, "b": False}


# Tests de ChromaDB real: requieren OPENAI_API_KEY + vector store indexado
@pytest.mark.skipif(
    "not config.getoption('--integration')",
    reason="requiere --integration y OPENAI_API_KEY",
)
def test_search_returns_results(has_openai_key):
    if not has_openai_key:
        pytest.skip("OPENAI_API_KEY no configurada")
    pytest.importorskip("chromadb")
    from ingestion.indexer import search

    results = search("costos de soja en zona nucleo", k=3)
    assert len(results) > 0
    for chunk, score in results:
        assert 0 <= score <= 1
        assert chunk.text
        assert chunk.metadata.pagina > 0
