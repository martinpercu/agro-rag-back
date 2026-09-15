"""Tests Paso 1: Sentinel Hub client with mocked HTTP (no quota spend)."""
from __future__ import annotations

import json

import httpx
import pytest

from satellite.geo import to_polygon_point_radius
from satellite.sentinel import (
    SentinelAuthError,
    SentinelError,
    SentinelHubClient,
    SentinelQuotaError,
)

POLY = to_polygon_point_radius(-34.5, -62.0, 5.0)
TOKEN_BODY = {"access_token": "tok123", "expires_in": 600}

STATS_BODY = {
    "data": [
        {
            "interval": {"from": "2026-06-01T00:00:00Z", "to": "2026-06-06T00:00:00Z"},
            "outputs": {
                "ndvi": {"bands": {"B0": {"stats": {"min": 0.2, "max": 0.8, "mean": 0.55, "stDev": 0.1, "sampleCount": 120, "noDataCount": 5}}}},
                "dataMask": {"bands": {"B0": {"stats": {"mean": 1.0}}}},
            },
        },
        {
            "interval": {"from": "2026-06-06T00:00:00Z", "to": "2026-06-11T00:00:00Z"},
            "outputs": {
                "ndvi": {"bands": {"B0": {"stats": {"min": 0.0, "max": 0.0, "mean": 0.0, "stDev": 0.0, "sampleCount": 0, "noDataCount": 130}}}},
            },
        },
    ]
}


def _client(handler, **kw):
    return SentinelHubClient(
        client_id="id", client_secret="sec", http=httpx.Client(transport=httpx.MockTransport(handler)), **kw
    )


def test_token_cached_between_calls():
    calls = {"token": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if "openid-connect/token" in str(req.url):
            calls["token"] += 1
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(200, json={"data": []})

    c = _client(handler)
    assert c.get_token() == "tok123"
    assert c.get_token() == "tok123"
    assert calls["token"] == 1


def test_missing_creds_without_http():
    c = SentinelHubClient(client_id="", client_secret="")
    with pytest.raises(SentinelAuthError):
        c.get_token()


def test_token_rejected():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "unauthorized_client"})

    with pytest.raises(SentinelAuthError):
        _client(handler).get_token()


def test_timeseries_parses_buckets_and_keeps_empty():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if "openid-connect/token" in str(req.url):
            return httpx.Response(200, json=TOKEN_BODY)
        seen["auth"] = req.headers.get("authorization")
        body = json.loads(req.content.decode())
        seen["payload"] = body
        return httpx.Response(200, json=STATS_BODY)

    c = _client(handler)
    series = c.ndvi_timeseries(POLY, "2026-06-01", "2026-06-11")
    assert seen["auth"] == "Bearer tok123"
    assert len(series) == 2
    full, cloudy = series
    assert full["ndvi_mean"] == pytest.approx(0.55)
    assert full["empty"] is False
    assert full["sample_count"] == 120
    assert cloudy["ndvi_mean"] is None and cloudy["empty"] is True
    agg = seen["payload"]["aggregation"]
    assert agg["aggregationInterval"] == {"of": "P5D"}
    assert agg["resx"] == 10 and agg["resy"] == 10
    assert "B04" in agg["evalscript"] and "B08" in agg["evalscript"]
    data_filter = seen["payload"]["input"]["data"][0]["dataFilter"]
    assert data_filter["maxCloudCoverage"] == 10
    assert seen["payload"]["input"]["data"][0]["type"] == "sentinel-2-l2a"


def test_401_refreshes_token_once_and_retries():
    calls = {"token": 0, "stats": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if "openid-connect/token" in str(req.url):
            calls["token"] += 1
            return httpx.Response(200, json={"access_token": f"tok{calls['token']}", "expires_in": 600})
        calls["stats"] += 1
        if calls["stats"] == 1:
            return httpx.Response(401, text="expired")
        return httpx.Response(200, json={"data": []})

    series = _client(handler).ndvi_timeseries(POLY, "2026-06-01", "2026-06-11")
    assert series == []
    assert calls == {"token": 2, "stats": 2}


def test_429_maps_to_quota_error():
    def handler(req: httpx.Request) -> httpx.Response:
        if "openid-connect/token" in str(req.url):
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(429, text="too many requests")

    with pytest.raises(SentinelQuotaError):
        _client(handler).ndvi_timeseries(POLY, "2026-06-01", "2026-06-11")


def test_500_maps_to_sentinel_error():
    def handler(req: httpx.Request) -> httpx.Response:
        if "openid-connect/token" in str(req.url):
            return httpx.Response(200, json=TOKEN_BODY)
        return httpx.Response(500, text="backend exploded")

    with pytest.raises(SentinelError):
        _client(handler).ndvi_timeseries(POLY, "2026-06-01", "2026-06-11")


def test_invalid_polygon_fails_before_http():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"data": []})

    with pytest.raises(ValueError):
        _client(handler).ndvi_timeseries({"type": "Polygon", "coordinates": []}, "2026-06-01", "2026-06-11")
    assert calls["n"] == 0
