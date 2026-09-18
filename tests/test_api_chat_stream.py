"""Tests de POST /chat/stream (grafo con SSE, sin OpenAI real)."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import api.main as main_module
from api.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # Fuerza el path sin Langfuse (determinista, sin red a :3003)
    monkeypatch.setattr(main_module, "_get_langfuse", lambda: None)
    return TestClient(app)


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parsea bloques SSE 'event: X\\ndata: {...}' en lista de (event, payload)."""
    events: list[tuple[str, dict]] = []
    for block in text.strip().split("\n\n"):
        name = None
        payload: dict = {}
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[len("data:"):].strip())
        if name is not None:
            events.append((name, payload))
    return events


def test_chat_stream_validation(client):
    r = client.post("/chat/stream", json={})
    assert r.status_code == 422
    r = client.post("/chat/stream", json={"question": ""})
    assert r.status_code == 422


def test_chat_stream_off_topic_rejects_without_spend(client):
    """Off-topic → un solo chat_error, sin retrieval ni LLM."""
    with (
        patch("agent.nodes.retriever.search") as mock_search,
        patch("agent.nodes.answerer.stream_answer_async") as mock_stream,
    ):
        r = client.post("/chat/stream", json={"question": "quien gano el mundial?"})
        assert r.status_code == 200
        events = _parse_sse(r.text)
        assert len(events) == 1
        assert events[0][0] == "chat_error"
        assert "Márgenes" in events[0][1]["error"] or "margins" in events[0][1]["error"]
        mock_search.assert_not_called()
        mock_stream.assert_not_called()


def test_chat_stream_empty_retrieval_streams_fallback(client):
    """Sin chunks (search mockeado vacío) → meta + token(s) + done, sin OpenAI.

    stream_answer_async con items=[] devuelve el fallback sin llamar al LLM.
    """
    with patch("agent.nodes.retriever.search", return_value=[]):
        r = client.post("/chat/stream", json={"question": "costo maiz"})
        assert r.status_code == 200
        events = _parse_sse(r.text)
        names = [n for n, _ in events]
        assert names[0] == "chat_meta"
        assert names[-1] == "chat_done"
        assert "chat_token" in names

        meta = events[0][1]
        assert meta["intent"] in ("costos", "mercado", "general", "siembras")
        assert meta["sources"] == []
        assert meta["num_sources"] == 0
        assert meta["retrieval_ms"] >= 0

        done = events[-1][1]
        assert "no encontre" in done["answer"]
        assert done["sources"] == []
        assert done["input_tokens"] == 0
        assert done["output_tokens"] == 0


def test_chat_stream_extracts_plan_and_divisions(client):
    """El prefijo del grafo anota plan_intent/divisions y viajan en meta+done."""
    with patch("agent.nodes.retriever.search", return_value=[]):
        r = client.post(
            "/chat/stream",
            json={"question": "tengo 80ha maiz y 40ha soja, que me conviene?"},
        )
        assert r.status_code == 200
        events = _parse_sse(r.text)
        meta = events[0][1]
        assert meta["plan_intent"] is True
        assert meta["divisions"] == [
            {"hectares": "80", "cultivo": "maiz"},
            {"hectares": "40", "cultivo": "soja"},
        ]
        done = events[-1][1]
        assert done["plan_intent"] is True
        assert done["divisions"] == meta["divisions"]


def test_chat_stream_with_history(client):
    """history se acepta y no rompe el SSE."""
    with patch("agent.nodes.retriever.search", return_value=[]):
        r = client.post(
            "/chat/stream",
            json={
                "question": "y de soja?",
                "history": [
                    {"role": "user", "content": "tengo 80ha"},
                    {"role": "assistant", "content": "contame mas"},
                ],
            },
        )
        assert r.status_code == 200
        events = _parse_sse(r.text)
        assert events[0][0] == "chat_meta"
        assert events[-1][0] == "chat_done"
