"""Tests del API con TestClient (no requieren OpenAI para health/sessions/pdf)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "agroposta"


def test_stats(client):
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total" in data
    assert "by_section" in data
    assert "by_tipo" in data
    assert "by_cultivo" in data
    assert data["total"] > 0


def test_chat_missing_field(client):
    r = client.post("/chat", json={})
    assert r.status_code == 422  # pydantic validation


def test_chat_empty_question(client):
    r = client.post("/chat", json={"question": ""})
    assert r.status_code == 422


def test_session_not_found(client):
    r = client.get("/sessions/inexistente")
    assert r.status_code == 404


def test_export_pdf_unknown_session(client):
    r = client.post("/export-pdf", json={"session_id": "no-existe"})
    assert r.status_code == 404


def test_delete_unknown_session(client):
    # DELETE es idempotente: tiene que devolver 200 aunque no exista
    r = client.delete("/sessions/no-existe")
    assert r.status_code == 200


# Test que SI requiere OpenAI (skipeado salvo --integration)
@pytest.mark.skipif(
    "not config.getoption('--integration')",
    reason="requiere --integration y OPENAI_API_KEY",
)
def test_chat_integration(client, has_openai_key):
    if not has_openai_key:
        pytest.skip("OPENAI_API_KEY no configurada")
    r = client.post(
        "/chat",
        json={"question": "Cuanto me sale sembrar soja de primera en zona norte?"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "costos"
    assert "session_id" in data
    assert data["answer"]
    assert len(data["sources"]) > 0
    # El primer source tiene que tener metadata valida
    src = data["sources"][0]
    assert src["pagina"] > 0
    assert src["seccion"]


@pytest.mark.skipif(
    "not config.getoption('--integration')",
    reason="requiere --integration y OPENAI_API_KEY",
)
def test_export_pdf_integration(client, has_openai_key):
    if not has_openai_key:
        pytest.skip("OPENAI_API_KEY no configurada")
    r = client.post(
        "/chat",
        json={"question": "Que se proyecta para trigo en 2026/27?"},
    )
    assert r.status_code == 200
    session_id = r.json()["session_id"]

    r2 = client.post("/export-pdf", json={"session_id": session_id})
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "application/pdf"
    assert r2.content[:4] == b"%PDF"


# Tests del endpoint /compare

def test_compare_missing_field(client):
    """Validacion Pydantic: question es obligatorio."""
    r = client.post("/compare", json={})
    assert r.status_code == 422


def test_compare_empty_question(client):
    r = client.post("/compare", json={"question": ""})
    assert r.status_code == 422


@pytest.mark.skipif(
    "not config.getoption('--integration')",
    reason="requiere --integration y OPENAI_API_KEY",
)
def test_compare_integration(client, has_openai_key):
    """/compare corre las 6 strategies y devuelve todas las metricas."""
    if not has_openai_key:
        pytest.skip("OPENAI_API_KEY no configurada")

    r = client.post(
        "/compare",
        json={"question": "Cuanto cuesta un kilo de novillo en feedlot?"},
    )
    assert r.status_code == 200
    data = r.json()

    # Estructura
    assert data["question"] == "Cuanto cuesta un kilo de novillo en feedlot?"
    assert "strategies" in data
    assert len(data["strategies"]) == 6
    expected_names = {"baseline", "hybrid", "rerank", "query_rewrite", "multi_query", "hyde"}
    assert set(data["strategies"].keys()) == expected_names

    # Cada strategy tiene answer, sources, items, metrics
    for name, s in data["strategies"].items():
        assert "answer" in s
        assert "sources" in s
        assert "items" in s
        assert "metrics" in s
        m = s["metrics"]
        # Todas las metricas criticas presentes
        for k in [
            "retrieval_ms", "answerer_ms", "total_ms",
            "answerer_input_tokens", "answerer_output_tokens",
            "aux_llm_input_tokens", "aux_llm_output_tokens",
            "num_sources", "distinct_sources", "intent",
        ]:
            assert k in m, f"{name}.metrics[{k}] missing"
        # Intent correcto
        assert m["intent"] == "ganaderia"
        # Sources
        assert m["num_sources"] > 0
        assert m["distinct_sources"] > 0
        # Latencias razonables (< 60s para una query)
        assert m["total_ms"] < 60000


@pytest.mark.skipif(
    "not config.getoption('--integration')",
    reason="requiere --integration y OPENAI_API_KEY",
)
def test_compare_with_history(client, has_openai_key):
    """/compare acepta history y lo propaga a las strategies."""
    if not has_openai_key:
        pytest.skip("OPENAI_API_KEY no configurada")

    history = [
        {"role": "user", "content": "Hola, estoy pensando en armar un feedlot"},
        {"role": "assistant", "content": "Buena idea, te ayudo."},
        {"role": "user", "content": "Cuanto me sale?"},
    ]
    r = client.post(
        "/compare",
        json={
            "question": "Y eso en zona nucleo?",
            "history": history,
            "k": 4,
        },
    )
    assert r.status_code == 200
    data = r.json()
    # El query_rewrite tendria que haber reescrito "Y eso en zona nucleo?"
    # y haber gastado tokens auxiliares
    qr = data["strategies"]["query_rewrite"]
    assert qr["metrics"]["aux_llm_input_tokens"] > 0
    # k=4 deberia respetarse
    assert qr["metrics"]["num_sources"] <= 4
