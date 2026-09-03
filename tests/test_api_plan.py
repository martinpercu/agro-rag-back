"""Tests API plan/parse y stream headers — sin OpenAI ni DB."""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_plan_parse_with_divisions():
    c = TestClient(app)
    r = c.post("/plan/parse", json={"question": "tengo 80ha maiz y 40ha soja"})
    assert r.status_code == 200
    j = r.json()
    assert j["plan_intent"] is True
    assert j["divisions"] == [
        {"hectares": "80", "cultivo": "maiz"},
        {"hectares": "40", "cultivo": "soja"},
    ]
    assert j["location"] == {}


def test_plan_parse_no_intent():
    c = TestClient(app)
    r = c.post("/plan/parse", json={"question": "que precio tiene el maiz?"})
    assert r.status_code == 200
    j = r.json()
    assert j["plan_intent"] is False
    assert j["divisions"] == []


def test_plan_parse_with_history():
    c = TestClient(app)
    r = c.post(
        "/plan/parse",
        json={
            "question": "y de soja?",
            "history": [{"role": "user", "content": "tengo 80ha maiz"}],
        },
    )
    assert r.status_code == 200
    assert any(d["hectares"] == "80" for d in r.json()["divisions"])


def test_compare_stream_headers_no_buffer():
    c = TestClient(app)
    # con is_off_topic -> no gasta OpenAI, devuelve strategy_error pero con headers
    r = c.post(
        "/compare/stream",
        json={"question": "hola que tal como andas?", "enabled": ["baseline"]},
    )
    # TestClient junta todo, pero podemos chequear headers
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers.get("cache-control") == "no-cache, no-transform"
    assert r.headers.get("x-accel-buffering") == "no"


def test_health_has_vector():
    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200
    assert r.json()["vector_store"] in ("pinecone", "chroma")
