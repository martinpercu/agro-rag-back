"""Tests Paso 2: POST /satellite/ndvi + GET /satellite/health (cliente mockeado)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from api.main import app
from satellite.sentinel import SentinelAuthError, SentinelError, SentinelQuotaError

SERIES = [
    {
        "date_from": "2026-08-16T00:00:00Z",
        "date_to": "2026-08-21T00:00:00Z",
        "ndvi_mean": 0.676,
        "ndvi_min": 0.162,
        "ndvi_max": 0.865,
        "ndvi_stdev": 0.1,
        "sample_count": 775,
        "no_data_count": 0,
        "empty": False,
    }
]


class FakeClient:
    calls = 0

    def __init__(self, *a, **k):
        pass

    def get_token(self):
        return "tok"

    def ndvi_timeseries(self, polygon, date_from, date_to, aggregation="P5D"):
        FakeClient.calls += 1
        FakeClient.last = {"polygon": polygon, "from": date_from, "to": date_to, "agg": aggregation}
        return SERIES


@pytest.fixture(autouse=True)
def _mock_satellite(monkeypatch):
    monkeypatch.setattr("satellite.sentinel.SentinelHubClient", FakeClient)
    FakeClient.calls = 0
    api_main._SAT_SERIES_CACHE.clear()


def test_ndvi_point_radius_and_cache():
    c = TestClient(app)
    body = {"location": {"lat": -34.5, "lng": -62.0, "ha": 5}, "date_from": "2026-08-01", "date_to": "2026-08-31"}
    r1 = c.post("/satellite/ndvi", json=body)
    assert r1.status_code == 200, r1.text
    j1 = r1.json()
    assert j1["series"] == SERIES
    assert j1["cached"] is False
    assert j1["area_ha"] == pytest.approx(5.0, rel=0.03)
    assert j1["origin"]["mode"] == "point_radius"
    assert FakeClient.last["agg"] == "P5D"
    r2 = c.post("/satellite/ndvi", json=body)
    assert r2.json()["cached"] is True
    assert FakeClient.calls == 1


def test_ndvi_polygon_passthrough_and_p1d():
    c = TestClient(app)
    poly = {
        "type": "Polygon",
        "coordinates": [[[-62.0, -34.0], [-61.9, -34.0], [-61.9, -33.9], [-62.0, -33.9], [-62.0, -34.0]]],
    }
    r = c.post(
        "/satellite/ndvi",
        json={"polygon": poly, "date_from": "2026-08-01", "date_to": "2026-08-10", "aggregation": "p1d"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["aggregation"] == "P1D"
    assert FakeClient.last["agg"] == "P1D"


def test_ndvi_validation_errors():
    c = TestClient(app)
    assert c.post("/satellite/ndvi", json={"location": {}}).status_code == 422
    assert c.post("/satellite/ndvi", json={}).status_code == 422
    assert c.post("/satellite/ndvi", json={"location": {"provincia": "BA"}}).status_code == 422
    base = {"location": {"lat": -34.5, "lng": -62.0, "ha": 5}}
    assert c.post("/satellite/ndvi", json={**base, "date_from": "no-fecha"}).status_code == 422
    assert c.post("/satellite/ndvi", json={**base, "date_from": "2026-08-10", "date_to": "2026-08-01"}).status_code == 422
    assert c.post("/satellite/ndvi", json={**base, "date_from": "2025-01-01", "date_to": "2026-08-01"}).status_code == 422
    assert c.post("/satellite/ndvi", json={**base, "aggregation": "P30D"}).status_code == 422
    assert FakeClient.calls == 0


def test_ndvi_quota_and_auth_mapping(monkeypatch):
    c = TestClient(app)
    body = {"location": {"lat": -34.5, "lng": -62.0, "ha": 5}}

    class QuotaClient(FakeClient):
        def ndvi_timeseries(self, *a, **k):
            raise SentinelQuotaError("429 slow down")

    monkeypatch.setattr("satellite.sentinel.SentinelHubClient", QuotaClient)
    assert c.post("/satellite/ndvi", json=body).status_code == 429

    class AuthClient(FakeClient):
        def ndvi_timeseries(self, *a, **k):
            raise SentinelAuthError("no creds")

    monkeypatch.setattr("satellite.sentinel.SentinelHubClient", AuthClient)
    assert c.post("/satellite/ndvi", json=body).status_code == 500

    class BoomClient(FakeClient):
        def ndvi_timeseries(self, *a, **k):
            raise SentinelError("sh exploded")

    monkeypatch.setattr("satellite.sentinel.SentinelHubClient", BoomClient)
    assert c.post("/satellite/ndvi", json=body).status_code == 502


def test_health_ok_and_not_configured(monkeypatch):
    c = TestClient(app)
    r = c.get("/satellite/health")
    assert r.status_code == 200
    assert r.json() == {"configured": True, "token_ok": True}

    class NoCreds(FakeClient):
        def get_token(self):
            raise SentinelAuthError("missing SENTINEL_CLIENT_ID")

    monkeypatch.setattr("satellite.sentinel.SentinelHubClient", NoCreds)
    j = c.get("/satellite/health").json()
    assert j["configured"] is False and j["token_ok"] is False
