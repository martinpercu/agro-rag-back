"""Tests del runner: get_all_strategies, run_compare, metricas."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from agent.strategies.base import RetrievedItem, Strategy, StrategyResult
from agent.strategies.runner import (
    ComparisonResponse,
    StrategyComparisonResult,
    _run_one,
    get_all_strategies,
    run_compare,
)


# ---------- Tests del registry ----------

def test_get_all_strategies_returns_six():
    s = get_all_strategies()
    assert len(s) == 6
    names = [x.name for x in s]
    assert names == ["baseline", "hybrid", "rerank", "query_rewrite", "multi_query", "hyde"]


def test_get_all_strategies_is_singleton():
    s1 = get_all_strategies()
    s2 = get_all_strategies()
    assert s1 is s2  # misma lista


# ---------- Tests del runner con strategies mock ----------

class FakeStrategy(Strategy):
    """Strategy fake que devuelve items fijos y mide cuanto tarda."""

    def __init__(self, name: str, items: list[RetrievedItem], delay: float = 0.0):
        self.name = name
        self._items = items
        self._delay = delay

    def retrieve(self, question, history=None, k=6):
        time.sleep(self._delay)  # simula latencia
        return StrategyResult(
            name=self.name,
            items=self._items,
            intent="costos",
            retrieval_ms=self._delay * 1000,
            llm_input_tokens=10,
            llm_output_tokens=5,
        )


def _fake_items() -> list[RetrievedItem]:
    return [
        RetrievedItem(
            chunk_id=f"c{i}",
            text=f"texto {i}",
            seccion="costos_margenes",
            pagina=38 + i,
            cultivo="soja",
            campana="2026_27",
            tipo="tabla",
            score=0.5 - i * 0.05,
            rank=i,
        )
        for i in range(3)
    ]


def test_run_one_returns_all_metrics():
    items = _fake_items()
    s = FakeStrategy("test", items, delay=0.01)

    with patch("agent.strategies.runner.answer") as mock_answer:
        mock_answer.return_value = {
            "answer": "Respuesta fake",
            "sources": [{"seccion": "costos", "pagina": 38, "score": 0.5}],
            "input_tokens": 100,
            "output_tokens": 50,
        }
        result = asyncio.run(_run_one(s, "test question", None, 3))

    assert result.name == "test"
    assert result.intent == "costos"
    assert result.answer == "Respuesta fake"
    assert result.sources[0]["pagina"] == 38
    assert len(result.items) == 3
    m = result.metrics
    assert m["retrieval_ms"] > 0
    assert m["answerer_ms"] >= 0
    assert m["total_ms"] > 0
    assert m["answerer_input_tokens"] == 100
    assert m["answerer_output_tokens"] == 50
    assert m["aux_llm_input_tokens"] == 10
    assert m["aux_llm_output_tokens"] == 5
    assert m["num_sources"] == 3
    assert m["distinct_sources"] == 3
    assert m["intent"] == "costos"


def test_run_one_handles_answerer_failure():
    items = _fake_items()
    s = FakeStrategy("test", items)

    with patch(
        "agent.strategies.runner.answer",
        side_effect=RuntimeError("openai caida"),
    ):
        result = asyncio.run(_run_one(s, "test", None, 3))

    # Tenemos los items pero el answerer fallo
    assert result.name == "test"
    assert "answer_failed" in result.metrics["error"]
    assert len(result.items) == 3  # los items se conservan
    assert result.answer == ""  # sin answer


def test_run_one_handles_retrieve_failure():
    class Broken(Strategy):
        name = "broken"

        def retrieve(self, question, history=None, k=6):
            raise RuntimeError("retrieve explota")

    result = asyncio.run(_run_one(Broken(), "test", None, 3))
    assert "retrieve_failed" in result.metrics["error"]
    assert result.items == []


def test_run_compare_runs_in_parallel():
    """Si las strategies tardan 100ms cada una, run_compare deberia tardar ~100ms (no 600)."""
    items = _fake_items()

    def make(name, delay):
        return FakeStrategy(name, items, delay=delay)

    strategies = [make(f"s{i}", 0.1) for i in range(6)]

    with patch("agent.strategies.runner.answer") as mock_answer:
        mock_answer.return_value = {
            "answer": "x",
            "sources": [],
            "input_tokens": 10,
            "output_tokens": 5,
        }
        t0 = time.time()
        result = asyncio.run(run_compare("test", strategies=strategies))
        elapsed = time.time() - t0

    # En paralelo: deberia ser ~100ms (un poco mas por overhead)
    assert elapsed < 0.5, f"corrio en serie: {elapsed:.2f}s"
    assert isinstance(result, ComparisonResponse)
    assert len(result.strategies) == 6


def test_run_compare_returns_6_strategies_by_default():
    items = _fake_items()
    with patch("agent.strategies.runner.answer") as mock_answer:
        mock_answer.return_value = {
            "answer": "x",
            "sources": [],
            "input_tokens": 10,
            "output_tokens": 5,
        }
        result = asyncio.run(run_compare("test question"))

    # Default: las 6 strategies
    assert len(result.strategies) == 6
    assert all(name in result.strategies for name in [
        "baseline", "hybrid", "rerank", "query_rewrite", "multi_query", "hyde"
    ])
    # Cada strategy tiene su answer y metrics
    for name, s in result.strategies.items():
        assert s.answer == "x"
        assert s.metrics["total_ms"] > 0


def test_run_compare_passes_history_to_strategy():
    """El history del request llega a la strategy (lo usa query_rewrite)."""
    items = _fake_items()

    received_history = {}

    class HistorySpy(Strategy):
        name = "spy"

        def retrieve(self, question, history=None, k=6):
            received_history.update({"history": history, "question": question})
            return StrategyResult(name=self.name, items=items)

    history = [{"role": "user", "content": "msg1"}]
    with patch("agent.strategies.runner.answer") as mock_answer:
        mock_answer.return_value = {
            "answer": "x",
            "sources": [],
            "input_tokens": 0,
            "output_tokens": 0,
        }
        asyncio.run(run_compare("test", history=history, strategies=[HistorySpy()]))

    assert received_history["history"] == history
    assert received_history["question"] == "test"


def test_comparison_response_to_dict_shape():
    items = _fake_items()
    with patch("agent.strategies.runner.answer") as mock_answer:
        mock_answer.return_value = {
            "answer": "x",
            "sources": [],
            "input_tokens": 10,
            "output_tokens": 5,
        }
        result = asyncio.run(run_compare("test"))
    d = result.to_dict()
    assert d["question"] == "test"
    assert "strategies" in d
    for name, s in d["strategies"].items():
        assert "answer" in s
        assert "sources" in s
        assert "items" in s
        assert "metrics" in s
        # metricas minimas
        for k in ["retrieval_ms", "answerer_ms", "total_ms",
                  "answerer_input_tokens", "answerer_output_tokens",
                  "num_sources", "distinct_sources", "intent"]:
            assert k in s["metrics"]


def test_run_compare_with_empty_items():
    """Una strategy que no trae items: el answerer devuelve un fallback message."""

    class Empty(Strategy):
        name = "empty"

        def retrieve(self, question, history=None, k=6):
            return StrategyResult(name=self.name, items=[])

    with patch("agent.strategies.runner.answer") as mock_answer:
        mock_answer.return_value = {
            "answer": "fallback",
            "sources": [],
            "input_tokens": 0,
            "output_tokens": 0,
        }
        result = asyncio.run(run_compare("test", strategies=[Empty()]))

    assert result.strategies["empty"].answer == "fallback"
    assert result.strategies["empty"].items == []
