"""Tests de la strategy lexical (BM25 puro, sin embeddings)."""
from __future__ import annotations

from unittest.mock import patch

from agent.strategies.lexical import LexicalStrategy


def _fake_index(n: int = 10):
    ids = [f"chunk-{i}" for i in range(n)]
    metas = [{"seccion": "costos_margenes", "pagina": 40 + i} for i in range(n)]
    docs = [f"documento numero {i} con soja" for i in range(n)]
    return (None, ids, metas, docs)


def test_returns_exactly_k_items():
    s = LexicalStrategy()
    hits = [(i, float(10 - i)) for i in range(6)]
    with patch("agent.strategies.lexical._bm25_search", return_value=hits), patch(
        "agent.strategies.lexical._get_bm25_index", return_value=_fake_index()
    ):
        result = s.retrieve("soja", k=6)
    assert len(result.items) == 6
    assert result.items[0].chunk_id == "chunk-0"
    assert result.items[0].score == 10.0
    assert result.items[0].rank == 0
    assert result.items[5].rank == 5


def test_k_is_passed_to_bm25_search():
    s = LexicalStrategy()
    with patch(
        "agent.strategies.lexical._bm25_search", return_value=[]
    ) as mock_search, patch(
        "agent.strategies.lexical._get_bm25_index", return_value=_fake_index()
    ):
        s.retrieve("soja", k=16)
    mock_search.assert_called_once_with("soja", k=16, allowed_sections=None)


def test_applies_intent_section_filter():
    s = LexicalStrategy()
    expected = ["costos_margenes", "costos_operativos", "siembras"]
    with patch("agent.strategies.lexical._classify", return_value="costos"), patch(
        "agent.strategies.lexical._bm25_search", return_value=[]
    ) as mock_search, patch(
        "agent.strategies.lexical._get_bm25_index", return_value=_fake_index()
    ):
        s.retrieve("cuanto cuesta sembrar soja")
    mock_search.assert_called_once_with(
        "cuanto cuesta sembrar soja", k=6, allowed_sections=expected
    )


def test_no_hits_returns_empty():
    s = LexicalStrategy()
    with patch("agent.strategies.lexical._bm25_search", return_value=[]), patch(
        "agent.strategies.lexical._get_bm25_index", return_value=_fake_index()
    ):
        result = s.retrieve("zxqzxq no existe")
    assert result.items == []


def test_trace_has_classify_and_bm25():
    s = LexicalStrategy()
    with patch(
        "agent.strategies.lexical._bm25_search", return_value=[(0, 1.0)]
    ), patch(
        "agent.strategies.lexical._get_bm25_index", return_value=_fake_index()
    ):
        result = s.retrieve("soja")
    steps = [t.step for t in result.trace]
    assert steps == ["classify", "bm25_search"]
    assert all(t.acc_ms >= 0 for t in result.trace)
    assert result.extra["embedding_used"] is False


def test_metadata_is_mapped_to_items():
    s = LexicalStrategy()
    with patch(
        "agent.strategies.lexical._bm25_search", return_value=[(3, 0.9)]
    ), patch(
        "agent.strategies.lexical._get_bm25_index", return_value=_fake_index()
    ):
        result = s.retrieve("soja")
    item = result.items[0]
    assert item.chunk_id == "chunk-3"
    assert item.seccion == "costos_margenes"
    assert item.pagina == 43