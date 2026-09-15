"""Sentinel Hub client (Copernicus Data Space) — NDVI time series.

Separate from RAG: no LLM, no vector store. Sync httpx (same style as
agent/rerank_client.py). Credentials via env (never committed):

    SENTINEL_CLIENT_ID / SENTINEL_CLIENT_SECRET   (OAuth client, user dashboard)
    SENTINEL_BASE_URL  (default https://sh.dataspace.copernicus.eu)
    SENTINEL_TOKEN_URL (default CDSE realm token endpoint)

Docs: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html
"""
from __future__ import annotations

import os
import time

import httpx

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
BASE_URL = "https://sh.dataspace.copernicus.eu"

# NDVI = (B08 NIR - B04 red) / (B08 + B04), Sentinel-2 L2A. dataMask output
# excludes no-data pixels from the stats (required by Statistical API).
NDVI_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["B04", "B08", "dataMask"]}],
    output: [{id: "ndvi", bands: 1}, {id: "dataMask", bands: 1}]
  };
}
function evaluatePixel(s) {
  let ndvi = (s.B08 - s.B04) / (s.B08 + s.B04);
  return {ndvi: [ndvi], dataMask: [s.dataMask]};
}
"""


class SentinelError(Exception):
    pass


class SentinelAuthError(SentinelError):
    pass


class SentinelQuotaError(SentinelError):
    pass


class SentinelHubClient:
    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        base_url: str | None = None,
        token_url: str | None = None,
        http: httpx.Client | None = None,
        timeout: float = 30.0,
    ):
        self.client_id = client_id or os.getenv("SENTINEL_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("SENTINEL_CLIENT_SECRET", "")
        self.base_url = (base_url or os.getenv("SENTINEL_BASE_URL", BASE_URL)).rstrip("/")
        self.token_url = token_url or os.getenv("SENTINEL_TOKEN_URL", TOKEN_URL)
        self._http = http or httpx.Client(timeout=timeout)
        self._token: str | None = None
        self._token_exp: float = 0.0

    def _ensure_creds(self) -> None:
        if not (self.client_id and self.client_secret):
            raise SentinelAuthError(
                "missing SENTINEL_CLIENT_ID / SENTINEL_CLIENT_SECRET — "
                "create an OAuth client in Copernicus Dashboard > User Settings > OAuth clients"
            )

    def get_token(self) -> str:
        """OAuth2 client_credentials token, cached until ~60s before expiry."""
        if self._token and time.time() < self._token_exp:
            return self._token
        self._ensure_creds()
        r = self._http.post(
            self.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
        if r.status_code != 200:
            raise SentinelAuthError(f"token request failed ({r.status_code}): {r.text[:300]}")
        body = r.json()
        token = body.get("access_token", "")
        if not token:
            raise SentinelAuthError(f"token response without access_token: {str(body)[:300]}")
        self._token = token
        self._token_exp = time.time() + float(body.get("expires_in", 600)) - 60.0
        return token

    def _post_stats(self, payload: dict, token: str) -> httpx.Response:
        return self._http.post(
            f"{self.base_url}/api/v1/statistics",
            json=payload,
            headers={"authorization": f"Bearer {token}", "content-type": "application/json"},
        )

    def ndvi_timeseries(
        self,
        polygon: dict,
        date_from: str,
        date_to: str,
        aggregation: str = "P5D",
        max_cloud: int = 10,
        resolution_m: int = 10,
    ) -> list[dict]:
        """Mean NDVI per aggregation bucket for a GeoJSON Polygon (EPSG:4326).

        Dates "YYYY-MM-DD". aggregation "P5D" (default, ~Sentinel-2 revisit,
        less PU) or "P1D". resolution_m is converted to degrees because the
        Statistical API interprets resx/resy in the geometry CRS units
        (EPSG:4326 -> degrees, ~111320 m/deg). Returns
        [{date_from, date_to, ndvi_mean|None, ...}] keeping empty (fully
        cloudy) buckets with ndvi_mean None.
        """
        from satellite.geo import validate_polygon

        validated = validate_polygon(polygon)["polygon"]
        res_deg = resolution_m / 111_320.0
        payload = {
            "input": {
                "bounds": {
                    "geometry": validated,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {"from": f"{date_from}T00:00:00Z", "to": f"{date_to}T23:59:59Z"},
                            "maxCloudCoverage": max_cloud,
                        },
                    }
                ],
            },
            "aggregation": {
                "timeRange": {"from": f"{date_from}T00:00:00Z", "to": f"{date_to}T23:59:59Z"},
                "aggregationInterval": {"of": aggregation},
                "evalscript": NDVI_EVALSCRIPT,
                "resx": res_deg,
                "resy": res_deg,
            },
            "calculations": {"default": {}},
        }
        r = self._post_stats(payload, self.get_token())
        if r.status_code == 401:  # token stale -> refresh once and retry
            self._token, self._token_exp = None, 0.0
            r = self._post_stats(payload, self.get_token())
        if r.status_code == 429:
            raise SentinelQuotaError(f"rate limit / quota exceeded (429): {r.text[:300]}")
        if r.status_code != 200:
            raise SentinelError(f"statistics failed ({r.status_code}): {r.text[:500]}")
        return [_parse_bucket(b) for b in r.json().get("data", [])]


def _parse_bucket(bucket: dict) -> dict:
    interval = bucket.get("interval", {})
    stats = (
        bucket.get("outputs", {}).get("ndvi", {}).get("bands", {}).get("B0", {}).get("stats", {})
    )
    sample_count = int(stats.get("sampleCount", 0) or 0)
    no_data = int(stats.get("noDataCount", 0) or 0)
    empty = sample_count == 0
    return {
        "date_from": interval.get("from"),
        "date_to": interval.get("to"),
        "ndvi_mean": None if empty else stats.get("mean"),
        "ndvi_min": None if empty else stats.get("min"),
        "ndvi_max": None if empty else stats.get("max"),
        "ndvi_stdev": None if empty else stats.get("stDev"),
        "sample_count": sample_count,
        "no_data_count": no_data,
        "empty": empty,
    }
