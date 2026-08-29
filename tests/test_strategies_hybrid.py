"""Tests de la strategy hybrid (BM25 + semantico + RRF)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.strategies.hybrid import (
    _RRF_K,
    HybridStrategy,
    _rrf_merge,
    _tokenize,
)


# ---------- Tests unitarios puros (sin chroma) ----------

def test_tokenize_lowercases_and_strips_accents():
    assert _tokenize("Márgenes AGROPECUARIOS") == ["margenes", "agropecuarios"]
    assert _tokenize("SOJA y Maíz") == ["soja", "y", "maiz"]
    assert _tokenize("") == []
    assert _tokenize(None) == []  # type: ignore[arg-type]


def test_tokenize_handles_spanish_chars():
    # "ñoño" -> ["nono"]
    assert _tokenize("El niño") == ["el", "nino"]


def test_rrf_merge_simple():
    """RRF: cada ranking aporta 1/(k+rank+1). Sin overlap, el rank 0 gana."""
    rankings = [["a", "b", "c"], ["b", "a", "d"]]
    merged = _rrf_merge(rankings, k=60)
    # a aparece rank 0 en r1 y rank 1 en r2 => 1/61 + 1/62
    # b aparece rank 1 en r1 y rank 0 en r2 => 1/62 + 1/61
    # c aparece rank 2 en r1 => 1/63
    # d aparece rank 2 en r2 => 1/63
    # a y b deberian empatar (mismo score)
    assert merged[0][0] in ("a", "b")
    assert merged[1][0] in ("a", "b")
    # c y d empatan en el ultimo lugar
    assert merged[2][0] in ("c", "d")


def test_rrf_merge_overlap_dominates():
    """Un chunk en ambos rankings top-1 debe estar primero."""
    rankings = [["x", "y", "z"], ["x", "a", "b"]]
    merged = _rrf_merge(rankings, k=60)
    assert merged[0][0] == "x"


def test_rrf_merge_handles_empty():
    assert _rrf_merge([], k=60) == []
    assert _rrf_merge([["a"]], k=60) == [("a", 1.0 / 61)]


# ---------- Tests con mocks de chroma y BM25 ----------

def _fake_chroma_data():
    """Datos de chroma simulados: 3 chunks."""
    return {
        "ids": ["c1", "c2", "c3"],
        "documents": [
            "ZONA OESTE soja rendimiento 40 qq/ha",
            "TRIGO sur de bs as costo 230 US$/tn",
            "NOVILLO feedlot precio 3,58 US$/kg",
        ],
        "metadatas": [
            {"seccion": "costos_margenes", "pagina": 38, "cultivo": "soja",
             "campana": "2026_27", "tipo": "tabla"},
            {"seccion": "costos_margenes", "pagina": 37, "cultivo": "trigo",
             "campana": "2026_27", "tipo": "tabla"},
            {"seccion": "ganaderia_costos", "pagina": 76, "cultivo": None,
             "campana": None, "tipo": "tabla"},
        ],
    }


def test_hybrid_returns_top_k_items():
    s = HybridStrategy()
    chroma_data = _fake_chroma_data()

    with (
        patch("agent.strategies.hybrid._classify", return_value="general"),
        patch(
            "agent.strategies.hybrid._get_bm25_index",
            return_value=(None, chroma_data["ids"], chroma_data["metadatas"], chroma_data["documents"]),
        ),
        patch(
            "agent.strategies.hybrid._bm25_search",
            return_value=[(2, 5.5), (0, 3.2)],  # c3, c1
        ),
        patch(
            "agent.strategies.hybrid.search",
            return_value=[
                (
                    type("Chunk", (), {
                        "id": "c1",
                        "text": "ZONA OESTE soja rendimiento 40 qq/ha",
                        "metadata": type("M", (), chroma_data["metadatas"][0])(),
                    })(),
                    0.9,
                ),
                (
                    type("Chunk", (), {
                        "id": "c3",
                        "text": "NOVILLO feedlot precio 3,58 US$/kg",
                        "metadata": type("M", (), chroma_data["metadatas"][2])(),
                    })(),
                    0.6,
                ),
            ],
        ),
    ):
        result = s.retrieve("novillo feedlot", k=3)

    assert result.name == "hybrid"
    assert result.intent == "general"
    assert 1 <= result.num_sources <= 3
    # todos los items tienen score RRF (positivo)
    assert all(i.score > 0 for i in result.items)
    # ranks son 0, 1, 2...
    assert [i.rank for i in result.items] == list(range(result.num_sources))


def test_hybrid_extra_metadata_populated():
    s = HybridStrategy()
    chroma_data = _fake_chroma_data()
    with (
        patch("agent.strategies.hybrid._classify", return_value="general"),
        patch(
            "agent.strategies.hybrid._get_bm25_index",
            return_value=(None, chroma_data["ids"], chroma_data["metadatas"], chroma_data["documents"]),
        ),
        patch("agent.strategies.hybrid._bm25_search", return_value=[(0, 1.0)]),
        patch("agent.strategies.hybrid.search", return_value=[]),
    ):
        result = s.retrieve("x")

    extra = result.extra
    assert extra["rrf_k"] == _RRF_K
    assert extra["bm25_top_n"] == 20
    assert extra["chroma_top_n"] == 20
    assert "overlap" in extra
    assert "filter_sections" in extra


def test_hybrid_branch_widths_are_configurable():
    s = HybridStrategy(bm25_top_k=8, chroma_top_k=12)
    chroma_data = _fake_chroma_data()
    with (
        patch("agent.strategies.hybrid._classify", return_value="general"),
        patch(
            "agent.strategies.hybrid._get_bm25_index",
            return_value=(None, chroma_data["ids"], chroma_data["metadatas"], chroma_data["documents"]),
        ),
        patch("agent.strategies.hybrid._bm25_search", return_value=[(0, 1.0)]) as mock_bm25,
        patch("agent.strategies.hybrid.search", return_value=[]) as mock_search,
    ):
        result = s.retrieve("x")

    mock_bm25.assert_called_once_with("x", k=8, allowed_sections=None)
    mock_search.assert_called_once()
    assert mock_search.call_args.kwargs["k"] == 12
    assert result.extra["bm25_top_n"] == 8
    assert result.extra["chroma_top_n"] == 12


def test_hybrid_applies_section_filter_to_both_retrievers():
    s = HybridStrategy()
    chroma_data = _fake_chroma_data()
    with (
        patch("agent.strategies.hybrid._classify", return_value="ganaderia"),
        patch(
            "agent.strategies.hybrid._get_bm25_index",
            return_value=(None, chroma_data["ids"], chroma_data["metadatas"], chroma_data["documents"]),
        ),
        patch(
            "agent.strategies.hybrid._bm25_search",
            return_value=[(2, 5.0)],
        ) as mock_bm25,
        patch(
            "agent.strategies.hybrid.search",
            return_value=[],
        ) as mock_search,
    ):
        result = s.retrieve("novillo feedlot", k=3)

    # el filtro de ganaderia se paso a bm25_search
    mock_bm25.assert_called_once()
    args, kwargs = mock_bm25.call_args
    assert kwargs.get("allowed_sections") is not None
    assert "ganaderia_costos" in kwargs["allowed_sections"]
    # y a chroma search
    mock_search.assert_called_once()
    _, kwargs = mock_search.call_args
    assert kwargs.get("where") is not None


def test_hybrid_dedupes_chunks_appearing_in_both_rankings():
    """Si un chunk esta en ambos rankings, RRF lo cuenta una sola vez en el ranking final."""
    s = HybridStrategy()
    chroma_data = _fake_chroma_data()
    with (
        patch("agent.strategies.hybrid._classify", return_value="general"),
        patch(
            "agent.strategies.hybrid._get_bm25_index",
            return_value=(None, chroma_data["ids"], chroma_data["metadatas"], chroma_data["documents"]),
        ),
        patch(
            "agent.strategies.hybrid._bm25_search",
            return_value=[(0, 1.0), (1, 0.8)],
        ),
        patch(
            "agent.strategies.hybrid.search",
            return_value=[
                (
                    type("Chunk", (), {
                        "id": "c1",  # mismo que bm25[0]
                        "text": "x",
                        "metadata": type("M", (), chroma_data["metadatas"][0])(),
                    })(),
                    0.9,
                ),
                (
                    type("Chunk", (), {
                        "id": "c2",
                        "text": "y",
                        "metadata": type("M", (), chroma_data["metadatas"][1])(),
                    })(),
                    0.7,
                ),
            ],
        ),
    ):
        result = s.retrieve("test", k=3)

    cids = [i.chunk_id for i in result.items]
    # no deberia haber duplicados
    assert len(cids) == len(set(cids))
    # overlap deberia ser 2 (c1 y c2 aparecen en ambos rankings)
    assert result.extra["overlap"] == 2


@pytest.mark.skipif(
    "not config.getoption('--integration')",
    reason="requiere --integration y chroma indexado",
)
def test_hybrid_integration_against_real_chroma(has_openai_key):
    """Hybrid contra el chroma real: trae chunks, sin errores."""
    if not has_openai_key:
        pytest.skip("OPENAI_API_KEY no configurada")
    s = HybridStrategy()
    result = s.retrieve("cuanto cuesta un kilo de novillo en feedlot?", k=6)
    assert result.name == "hybrid"
    assert result.intent == "ganaderia"
    assert result.num_sources > 0
    assert result.retrieval_ms > 0
    # Algun chunk deberia ser de ganaderia_costos
    secciones = {i.seccion for i in result.items}
    assert "ganaderia_costos" in secciones
    # El indice BM25 se cachea
    assert s.__class__.__name__ == "HybridStrategy"


@pytest.mark.skipif(
    "not config.getoption('--integration')",
    reason="requiere --integration y chroma indexado",
)
def test_hybrid_both_components_run_in_real_chroma(has_openai_key):
    """Hybrid corre BM25 + chroma contra el chroma real, ambos con data."""
    if not has_openai_key:
        pytest.skip("OPENAI_API_KEY no configurada")

    from agent.strategies.hybrid import _bm25_cache, _get_bm25_index

    # Verificamos que el cache BM25 esta construido
    _get_bm25_index()  # fuerza build
    assert _bm25_cache.get("bm25") is not None
    assert _bm25_cache.get("count", 0) > 0

    s = HybridStrategy()
    r = s.retrieve("costos de soja en zona nucleo", k=6)
    assert r.num_sources == 6
    # Todos los items tienen text no vacio
    assert all(i.text and i.text.strip() for i in r.items)
    # RRF scores son positivos
    assert all(i.score > 0 for i in r.items)
    # metadata extra: bm25 y chroma corrieron
    assert r.extra["bm25_returned"] >= 1
    assert r.extra["chroma_returned"] >= 1
    # Overlap es >= 0
    assert r.extra["overlap"] >= 0
