"""Tests del interface base de las estrategias."""
from __future__ import annotations

from agent.strategies.base import (
    RetrievedItem,
    Strategy,
    StrategyResult,
    time_call,
)


def test_retrieved_item_basic():
    item = RetrievedItem(
        chunk_id="abc",
        text="algo",
        seccion="costos_margenes",
        pagina=38,
        cultivo="soja",
        campana="2026_27",
        tipo="tabla",
        score=0.87,
        rank=0,
    )
    assert item.score == 0.87
    assert item.rank == 0


def test_strategy_result_num_sources():
    r = StrategyResult(name="x")
    assert r.num_sources == 0

    r.items = [
        RetrievedItem(
            chunk_id=f"c{i}",
            text="t",
            seccion="costos_margenes",
            pagina=38,
            cultivo=None,
            campana=None,
            tipo="tabla",
            score=0.5,
            rank=i,
        )
        for i in range(3)
    ]
    assert r.num_sources == 3


def test_strategy_result_distinct_sources():
    """distinct_sources cuenta unicos por (seccion, pagina)."""
    r = StrategyResult(name="x")
    r.items = [
        RetrievedItem("a", "t1", "costos", 38, None, None, "tabla", 0.5, 0),
        RetrievedItem("b", "t2", "costos", 38, None, None, "tabla", 0.4, 1),  # dup
        RetrievedItem("c", "t3", "mercado", 50, None, None, "tabla", 0.3, 2),
    ]
    assert r.num_sources == 3
    assert r.distinct_sources == 2  # (costos,38) y (mercado,50)


def test_strategy_result_to_dict_shape():
    r = StrategyResult(
        name="baseline",
        intent="costos",
        retrieval_ms=123.4,
        llm_input_tokens=10,
        llm_output_tokens=20,
        extra={"foo": "bar"},
    )
    r.items = [
        RetrievedItem("a", "t", "costos", 38, "soja", "2026_27", "tabla", 0.5, 0),
    ]
    d = r.to_dict()
    assert d["name"] == "baseline"
    assert d["intent"] == "costos"
    assert d["retrieval_ms"] == 123.4
    assert d["num_sources"] == 1
    assert d["distinct_sources"] == 1
    assert d["llm_input_tokens"] == 10
    assert d["llm_output_tokens"] == 20
    assert d["extra"] == {"foo": "bar"}
    assert d["items"][0]["chunk_id"] == "a"
    assert d["items"][0]["seccion"] == "costos"
    assert d["items"][0]["score"] == 0.5


def test_strategy_abstract_cannot_be_instantiated():
    """Strategy es abstracta: no se puede instanciar directamente."""
    try:
        Strategy()
        assert False, "debio fallar"
    except TypeError:
        pass


def test_custom_strategy_minimal():
    """Una estrategia concreta minima: devuelve 1 chunk fijo."""

    class Dummy(Strategy):
        name = "dummy"

        def retrieve(self, question, history=None, k=6):
            return StrategyResult(
                name=self.name,
                items=[
                    RetrievedItem(
                        chunk_id="x",
                        text=question,
                        seccion="general",
                        pagina=1,
                        cultivo=None,
                        campana=None,
                        tipo="narrativa",
                        score=1.0,
                        rank=0,
                    )
                ],
                intent="general",
            )

    s = Dummy()
    r = s.retrieve("hola")
    assert r.name == "dummy"
    assert r.num_sources == 1
    assert r.items[0].text == "hola"


def test_time_call_returns_elapsed_ms():
    def f():
        return 42

    result, ms = time_call(f)
    assert result == 42
    assert ms >= 0
    assert ms < 1000  # deberia ser instantaneo
