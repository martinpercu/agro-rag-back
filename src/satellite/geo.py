"""Field geometry helpers for the satellite module (Sentinel-2, separate from RAG).

All locations normalize to a GeoJSON Polygon (EPSG:4326) plus centroid,
area and origin mode, so every input mode (point+radius, manual points,
drawn map polygon) flows through one contract:

    {"polygon": {"type": "Polygon", "coordinates": [[[lng, lat], ...]]},
     "centroid": {"lat": .., "lng": ..},
     "area_ha": ..,
     "origin": {"mode": "point_radius" | "manual_points" | "manual_bbox" | "polygon"},
     "label"?: ..}

Stdlib only (no geo deps).
"""
from __future__ import annotations

import math

# Product scope: Argentine producers. Lenient bounding box with margin.
AR_LNG_MIN, AR_LNG_MAX = -75.0, -53.0
AR_LAT_MIN, AR_LAT_MAX = -56.0, -21.0

# Point+radius mode: slider range agreed with product (1-20 ha).
POINT_MIN_HA, POINT_MAX_HA = 1.0, 20.0

# Manual/drawn polygons: sanity caps (a single field above this is an error).
MAX_VERTICES = 500
MAX_AREA_HA = 50_000.0

_METERS_PER_DEG_LAT = 111_320.0


def ha_to_radius_m(ha: float) -> float:
    """Circle radius in meters for a given area in hectares (r = sqrt(A/pi))."""
    return math.sqrt(ha * 10_000.0 / math.pi)


def _check_lat_lng(lat: float, lng: float) -> None:
    if not (isinstance(lat, (int, float)) and isinstance(lng, (int, float))):
        raise ValueError(f"lat/lng must be numbers, got lat={lat!r} lng={lng!r}")
    if math.isnan(lat) or math.isnan(lng) or math.isinf(lat) or math.isinf(lng):
        raise ValueError(f"lat/lng must be finite, got lat={lat!r} lng={lng!r}")
    if not (AR_LAT_MIN <= lat <= AR_LAT_MAX and AR_LNG_MIN <= lng <= AR_LNG_MAX):
        raise ValueError(
            f"point out of scope (Argentina approx): lat={lat} lng={lng} "
            f"expected lat [{AR_LAT_MIN},{AR_LAT_MAX}] lng [{AR_LNG_MIN},{AR_LNG_MAX}]"
        )


def to_polygon_point_radius(lat: float, lng: float, ha: float, vertices: int = 32) -> dict:
    """Build a circle-like GeoJSON Polygon from a center point + area in ha."""
    if not (POINT_MIN_HA <= ha <= POINT_MAX_HA):
        raise ValueError(f"ha must be within [{POINT_MIN_HA},{POINT_MAX_HA}] for point_radius mode, got {ha!r}")
    _check_lat_lng(lat, lng)
    if not (8 <= vertices <= MAX_VERTICES):
        raise ValueError(f"vertices must be within [8,{MAX_VERTICES}], got {vertices!r}")
    r_m = ha_to_radius_m(float(ha))
    lat0 = math.radians(float(lat))
    ring: list[list[float]] = []
    for i in range(vertices):
        theta = 2.0 * math.pi * i / vertices
        dx = r_m * math.cos(theta)  # meters east
        dy = r_m * math.sin(theta)  # meters north
        plat = float(lat) + dy / _METERS_PER_DEG_LAT
        plng = float(lng) + dx / (_METERS_PER_DEG_LAT * math.cos(lat0))
        ring.append([plng, plat])
    ring.append(ring[0][:])
    return {"type": "Polygon", "coordinates": [ring]}


def to_polygon_manual_bbox(lat_min: float, lat_max: float, lng_min: float, lng_max: float) -> dict:
    """Build a rectangle GeoJSON Polygon from a bounding box."""
    for v in (lat_min, lat_max, lng_min, lng_max):
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            raise ValueError(f"bbox values must be finite numbers, got {v!r}")
    if not (lat_min < lat_max and lng_min < lng_max):
        raise ValueError(f"bbox must satisfy lat_min<lat_max and lng_min<lng_max, got {lat_min, lat_max, lng_min, lng_max!r}")
    _check_lat_lng(lat_min, lng_min)
    _check_lat_lng(lat_max, lng_max)
    ring = [
        [lng_min, lat_min],
        [lng_max, lat_min],
        [lng_max, lat_max],
        [lng_min, lat_max],
        [lng_min, lat_min],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def _coerce_ring(points: list) -> list[list[float]]:
    """Accept [{lat,lng}, ...] or [[lng,lat], ...] (GeoJSON order) -> [[lng,lat], ...]."""
    ring: list[list[float]] = []
    for p in points:
        if isinstance(p, dict):
            lat, lng = p.get("lat"), p.get("lng")
        elif isinstance(p, (list, tuple)) and len(p) == 2:
            lng, lat = p[0], p[1]
        else:
            raise ValueError(f"vertex must be {{lat,lng}} or [lng,lat], got {p!r}")
        if not (isinstance(lat, (int, float)) and isinstance(lng, (int, float))):
            raise ValueError(f"vertex lat/lng must be numbers, got {p!r}")
        ring.append([float(lng), float(lat)])
    return ring


def to_polygon_manual_points(points: list) -> dict:
    """Build a GeoJSON Polygon from >=3 vertices (auto-closes the ring)."""
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError(f"manual_points needs at least 3 vertices, got {points!r}")
    if len(points) > MAX_VERTICES:
        raise ValueError(f"too many vertices ({len(points)} > {MAX_VERTICES})")
    ring = _coerce_ring(points)
    if ring[0] != ring[-1]:
        ring.append(ring[0][:])
    for lng, lat in ring:
        _check_lat_lng(lat, lng)
    return {"type": "Polygon", "coordinates": [ring]}


def polygon_area_ha(polygon: dict) -> float:
    """Planar area in ha (equirectangular projection around ring centroid)."""
    try:
        ring = polygon["coordinates"][0]
    except (KeyError, IndexError, TypeError):
        raise ValueError(f"invalid GeoJSON Polygon: {polygon!r}")
    if len(ring) < 4:
        raise ValueError(f"polygon ring needs at least 4 positions (closed), got {len(ring)}")
    lat0 = sum(p[1] for p in ring) / len(ring)
    cos0 = math.cos(math.radians(lat0))

    def proj(lng: float, lat: float) -> tuple[float, float]:
        return (lng * _METERS_PER_DEG_LAT * cos0, lat * _METERS_PER_DEG_LAT)

    area = 0.0
    for i in range(len(ring) - 1):
        x0, y0 = proj(ring[i][0], ring[i][1])
        x1, y1 = proj(ring[i + 1][0], ring[i + 1][1])
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0 / 10_000.0


def polygon_centroid(polygon: dict) -> dict:
    ring = polygon["coordinates"][0]
    pts = ring[:-1] if ring[0] == ring[-1] else ring
    return {
        "lat": sum(p[1] for p in pts) / len(pts),
        "lng": sum(p[0] for p in pts) / len(pts),
    }


def validate_polygon(polygon: dict) -> dict:
    """Validate a GeoJSON Polygon -> {polygon, centroid, area_ha} (raises ValueError)."""
    if not isinstance(polygon, dict) or polygon.get("type") != "Polygon":
        raise ValueError(f"polygon must be a GeoJSON Polygon, got {polygon!r}")
    coords = polygon.get("coordinates")
    if not isinstance(coords, list) or len(coords) != 1 or not isinstance(coords[0], list):
        raise ValueError("polygon must have exactly one linear ring")
    ring = coords[0]
    if len(ring) < 4:
        raise ValueError(f"polygon ring needs at least 4 positions (closed), got {len(ring)}")
    if len(ring) - 1 > MAX_VERTICES:
        raise ValueError(f"too many vertices ({len(ring) - 1} > {MAX_VERTICES})")
    if ring[0] != ring[-1]:
        raise ValueError("polygon ring must be closed (first == last position)")
    for p in ring:
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            raise ValueError(f"invalid position {p!r}, expected [lng,lat]")
        lng, lat = p
        if not (isinstance(lat, (int, float)) and isinstance(lng, (int, float))):
            raise ValueError(f"position lat/lng must be numbers, got {p!r}")
        _check_lat_lng(float(lat), float(lng))
    area_ha = polygon_area_ha(polygon)
    if area_ha > MAX_AREA_HA:
        raise ValueError(f"polygon area {area_ha:.1f} ha exceeds max {MAX_AREA_HA:.0f} ha")
    if area_ha <= 0:
        raise ValueError("polygon area must be positive")
    return {"polygon": polygon, "centroid": polygon_centroid(polygon), "area_ha": round(area_ha, 3)}


def normalize_location(location: dict | None) -> dict:
    """Normalize any accepted location input to the satellite contract ({} legacy ok).

    Accepted:
      {} | None                      -> {} (no field location)
      {"polygon": GeoJSON}           -> validated contract, origin mode "polygon"
      {"lat","lng","ha"}             -> point+radius circle, mode "point_radius"
      {"vertices": [...]}            -> manual points, mode "manual_points"
      {"bbox": {lat_min,...}}        -> manual bbox rectangle, mode "manual_bbox"
    Optional "label" is preserved. Raises ValueError on invalid input.
    """
    if location is None:
        return {}
    if not isinstance(location, dict):
        raise ValueError(f"location must be an object, got {location!r}")
    if not location:
        return {}
    label = location.get("label")

    def wrap(validated: dict, mode: str, extra: dict | None = None) -> dict:
        out = {
            "polygon": validated["polygon"],
            "centroid": validated["centroid"],
            "area_ha": validated["area_ha"],
            "origin": {"mode": mode, **(extra or {})},
        }
        if label is not None:
            out["label"] = label
        return out

    if "polygon" in location:
        return wrap(validate_polygon(location["polygon"]), "polygon")
    if "lat" in location and "lng" in location and "ha" in location:
        poly = to_polygon_point_radius(location["lat"], location["lng"], location["ha"])
        v = validate_polygon(poly)
        return wrap(v, "point_radius", {"ha": float(location["ha"])})
    if "vertices" in location:
        return wrap(validate_polygon(to_polygon_manual_points(location["vertices"])), "manual_points")
    if "bbox" in location:
        b = location["bbox"]
        try:
            poly = to_polygon_manual_bbox(b["lat_min"], b["lat_max"], b["lng_min"], b["lng_max"])
        except KeyError as e:
            raise ValueError(f"bbox needs lat_min/lat_max/lng_min/lng_max, missing {e}")
        return wrap(validate_polygon(poly), "manual_bbox")
    # Unknown shape: keep legacy tolerant behavior only for empty, else reject loudly.
    raise ValueError(
        "location must be {} or one of {polygon} | {lat,lng,ha} | {vertices} | {bbox}, "
        f"got keys {sorted(location.keys())}"
    )
