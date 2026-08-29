"""Tests de la strategy rerank_ce (cross-encoder via servicio local)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.rerank_client import RerankClient, RerankResult
from agent.strategies.rerank_ce import RerankCEStrategy

N_CHUNKS = 6
MOCK_ITEMS = [
    (MagicMock(id=f"c{i}"), float(1 - i * 0.05))
    for i in range(N_CHUNKS)
]
MOCK_DOCS = [f"doc {i}" for i in range(N_CHUNKS)]


@pytest.fixture()
def strategy(monkeypatch):
    """RerankCEStrategy con chroma y cliente mockeados."""
    fake_hits = []
    for i, (chunk, score) in enumerate(MOCK_ITEMS):
        chunk.metadata.seccion = "costos_margenes"
        chunk.metadata.pagina = 40 + i
        chunk.metadata.cultivo = "maiz"
        chunk.metadata.campana = None
        chunk.metadata.tipo = "tabla"
        chunk.text = MOCK_DOCS[i]
        fake_hits.append((chunk, score))
    monkeypatch.setattr("agent.strategies.rerank_ce.search", lambda *a, **k: fake_hits)

    client = MagicMock(spec=RerankClient)
    client.base_url = "http://mock:8001/v1/rerank"
    client.rerank.return_value = RerankResult(
        indices=[3, 0, 5, 1, 2, 4],
        scores=[0.98, 0.95, 0.90, 0.85, 0.80, 0.70],
        model="mock-cross-encoder",
        service_ms=130.0,
    )
    return RerankCEStrategy(client=client), client


def test_rerank_ce_orders_by_cross_encoder(strategy):
    s, client = strategy
    res = s.retrieve("Cuanto cuesta sembrar maiz?", k=6)
    assert [i.chunk_id for i in res.items] == ["c3", "c0", "c5", "c1", "c2", "c4"]
    assert client.rerank.called
    assert client.rerank.call_args.args[0] == "Cuanto cuesta sembrar maiz?"
    assert res.retrieval_ms >= 0
    assert res.extra["fallback"] is None
    assert res.extra["rerank_model"] == "mock-cross-encoder"
    assert res.extra["rerank_service_ms"] == 130.0
    # score final = relevance_score del cross-encoder
    assert round(res.items[0].score, 2) == 0.98


def test_rerank_ce_fallback_on_service_error(strategy):
    s, client = strategy
    client.rerank.side_effect = RuntimeError("connection refused")
    res = s.retrieve("Cuanto cuesta sembrar maiz?", k=6)
    assert [i.chunk_id for i in res.items] == ["c0", "c1", "c2", "c3", "c4", "c5"]
    assert res.extra["fallback"] == "service_error"
    assert "rerank_service_failed" in res.extra.get("error", "")


def test_rerank_ce_fills_up_to_k(strategy):
    s, client = strategy
    client.rerank.return_value = RerankResult(
        indices=[0, 1],
        scores=[0.9, 0.8],
        model="mock",
        service_ms=10.0,
    )
    res = s.retrieve("Cuanto cuesta sembrar maiz?", k=6)
    # 2 del cross-encoder + completar con el orden original hasta k
    assert len(res.items) == 6
    assert res.items[0].chunk_id == "c0"
    assert res.items[1].chunk_id == "c1"
    assert res.items[2].chunk_id == "c2"


def test_rerank_ce_empty_candidates(monkeypatch):
    monkeypatch.setattr("agent.strategies.rerank_ce.search", lambda *a, **k: [])
    client = MagicMock(spec=RerankClient)
    client.base_url = "http://mock:8001/v1/rerank"
    s = RerankCEStrategy(client=client)
    res = s.retrieve("Pregunta sin resultados", k=6)
    assert res.items == []
    assert not client.rerank.called