"""Live demo: NDVI P5D series for a fixed Pampa Humeda field (needs creds).

Setup (OAuth client, shown once — keep the secret in .env.local, never commit):
  1. Login at https://dataspace.copernicus.eu -> profile icon -> "Sentinel Hub"
  2. User Settings -> OAuth clients -> Create (name e.g. agroposta-dev, NOT a SPA)
  3. Copy client ID + secret, then add to .env.local:
       SENTINEL_CLIENT_ID=<id>
       SENTINEL_CLIENT_SECRET=<secret>

Run:
  cd agro-rag-back && set -a; source .env.local; set +a
  uv run python scripts/satellite_demo.py
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from satellite.geo import to_polygon_point_radius  # noqa: E402
from satellite.sentinel import SentinelHubClient  # noqa: E402


def main() -> int:
    if not (os.getenv("SENTINEL_CLIENT_ID") and os.getenv("SENTINEL_CLIENT_SECRET")):
        print("missing SENTINEL_CLIENT_ID / SENTINEL_CLIENT_SECRET (see docstring above)")
        return 2
    polygon = to_polygon_point_radius(-34.5, -62.0, 5.0)
    to_day = date.today()
    from_day = to_day - timedelta(days=90)
    client = SentinelHubClient()
    series = client.ndvi_timeseries(polygon, from_day.isoformat(), to_day.isoformat(), aggregation="P5D")
    print(f"NDVI P5D {from_day} -> {to_day} (5 ha @ -34.5,-62.0, maxCloud 10%)")
    print(f"{'from':<12}{'to':<12}{'mean':>8}{'min':>8}{'max':>8}  samples")
    for b in series:
        mean = f"{b['ndvi_mean']:.3f}" if b["ndvi_mean"] is not None else "  nubes"
        mn = f"{b['ndvi_min']:.3f}" if b["ndvi_min"] is not None else "   -"
        mx = f"{b['ndvi_max']:.3f}" if b["ndvi_max"] is not None else "   -"
        print(f"{b['date_from'][:10]:<12}{b['date_to'][:10]:<12}{mean:>8}{mn:>8}{mx:>8}  {b['sample_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
