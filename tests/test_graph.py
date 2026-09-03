"""Tests graph Fase 2 — plan_intent + field_collector, sin OpenAI."""
from __future__ import annotations

from unittest.mock import patch

from agent.graph import make_graph


def test_graph_plan_intent_and_divisions():
    g = make_graph()
    # Mock el retriever.search para no tocar Pinecone/Chroma
    with patch("agent.nodes.retriever.search", return_value=[]):
        res = g.invoke({"question": "tengo 80ha maiz y 40ha soja", "history": []})
        assert res["plan_intent"] is True
        assert res["divisions"] == [
            {"hectares": "80", "cultivo": "maiz"},
            {"hectares": "40", "cultivo": "soja"},
        ]
        # classifier sigue funcionando
        assert res["intent"] in ("costos", "siembras", "general")


def test_graph_no_plan():
    g = make_graph()
    with patch("agent.nodes.retriever.search", return_value=[]):
        res = g.invoke({"question": "que precio tiene el maiz?", "history": []})
        assert res["plan_intent"] is False
        assert res["divisions"] == []
