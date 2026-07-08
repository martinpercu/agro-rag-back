"""Tests de la strategy baseline (semantico + filtro por intent)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.strategies.baseline import BaselineStrategy


@pytest.fixture
def fake_chunk():
    """Un chunk minimalista para los mocks."""
    from schemas import Chunk, ChunkMetadata

    return Chunk(
        id="chunk-1",
        text="ZONA OESTE ... RENDIMIENTOS 40",
        metadata=ChunkMetadata(
            edicion="2026_05",
            seccion="costos_margenes",
            cultivo="soja",
            zona=None,
            campana="2026_27",
            tipo="tabla",
            pagina=38,
        ),
    )


def test_baseline_calls_search_with_no_filter_for_general_intent(fake_chunk):
    s = BaselineStrategy()
    with (
        patch("agent.strategies.baseline._classify", return_value="general"),
        patch(
            "agent.strategies.baseline.search",
            return_value=[(fake_chunk, 0.85)],
        ) as mock_search,
    ):
        result = s.retrieve("que pasa con el clima?")

    assert result.name == "baseline"
    assert result.intent == "general"
    assert result.num_sources == 1
    # general no filtra por seccion
    assert result.extra["filter_sections"] is None
    mock_search.assert_called_once()
    args, kwargs = mock_search.call_args
    assert kwargs.get("where") is None or "where" not in kwargs


def test_baseline_filters_by_section_for_costos_intent(fake_chunk):
    s = BaselineStrategy()
    with (
        patch("agent.strategies.baseline._classify", return_value="costos"),
        patch(
            "agent.strategies.baseline.search",
            return_value=[(fake_chunk, 0.72)],
        ) as mock_search,
    ):
        result = s.retrieve("cuanto sale la soja?")

    assert result.intent == "costos"
    assert result.extra["filter_sections"] is not None
    assert "costos_margenes" in result.extra["filter_sections"]
    mock_search.assert_called_once()
    _, kwargs = mock_search.call_args
    assert "where" in kwargs
    assert kwargs["where"]["seccion"]["$in"] == result.extra["filter_sections"]


def test_baseline_maps_search_results_to_retrieved_items(fake_chunk):
    s = BaselineStrategy()
    chunks = [
        (fake_chunk, 0.9),
        (fake_chunk.model_copy(update={"id": "chunk-2"}), 0.7),
    ]
    with (
        patch("agent.strategies.baseline._classify", return_value="general"),
        patch("agent.strategies.baseline.search", return_value=chunks),
    ):
        result = s.retrieve("test query")

    assert result.num_sources == 2
    assert result.items[0].rank == 0
    assert result.items[1].rank == 1
    assert result.items[0].score == 0.9
    assert result.items[0].seccion == "costos_margenes"
    assert result.items[0].pagina == 38
    assert result.items[0].cultivo == "soja"


def test_baseline_timing_is_recorded(fake_chunk):
    s = BaselineStrategy()
    with (
        patch("agent.strategies.baseline._classify", return_value="general"),
        patch("agent.strategies.baseline.search", return_value=[(fake_chunk, 0.5)]),
    ):
        result = s.retrieve("test")
    assert result.retrieval_ms >= 0
    assert result.retrieval_ms < 1000  # deberia ser sub-milisegundo con mocks


def test_baseline_returns_empty_when_no_hits():
    s = BaselineStrategy()
    with (
        patch("agent.strategies.baseline._classify", return_value="general"),
        patch("agent.strategies.baseline.search", return_value=[]),
    ):
        result = s.retrieve("nada en el store")
    assert result.num_sources == 0
    assert result.items == []


@pytest.mark.skipif(
    "not config.getoption('--integration')",
    reason="requiere --integration y chroma indexado",
)
def test_baseline_integration_against_real_chroma(has_openai_key):
    if not has_openai_key:
        pytest.skip("OPENAI_API_KEY no configurada")
    s = BaselineStrategy()
    result = s.retrieve("cuanto cuesta un kilo de novillo en feedlot?", k=6)
    assert result.name == "baseline"
    # Novillo en feedlot -> ganaderia (no costos)
    assert result.intent == "ganaderia"
    assert result.num_sources > 0
    # La primera fuente deberia ser de ganaderia_costos
    assert result.items[0].seccion == "ganaderia_costos"
