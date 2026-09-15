"""Tests Paso 0: contrato location satelital (sin red, sin DB)."""
from __future__ import annotations

import math

import pytest

from satellite.geo import (
    ha_to_radius_m,
    normalize_location,
    polygon_area_ha,
    to_polygon_manual_bbox,
    to_polygon_manual_points,
    to_polygon_point_radius,
    validate_polygon,
)

# Pampa Humeda (dentro del bbox AR)
LAT, LNG = -34.5, -62.0


def test_ha_to_radius_m_bounds():
    assert ha_to_radius_m(1.0) == pytest.approx(56.42, abs=0.05)
    assert ha_to_radius_m(20.0) == pytest.approx(252.31, abs=0.05)


def test_point_radius_circle_matches_area():
    for ha in (1.0, 5.0, 10.0, 20.0):
        poly = to_polygon_point_radius(LAT, LNG, ha)
        ring = poly["coordinates"][0]
        assert len(ring) == 33  # 32 vertices + cierre
        assert ring[0] == ring[-1]
        assert polygon_area_ha(poly) == pytest.approx(ha, rel=0.03)


def test_point_radius_ha_out_of_range():
    for bad in (0.0, 0.5, 20.5, 100.0):
        with pytest.raises(ValueError):
            to_polygon_point_radius(LAT, LNG, bad)


def test_point_radius_out_of_scope():
    with pytest.raises(ValueError):
        to_polygon_point_radius(0.0, 0.0, 5.0)


def test_manual_points_rectangle_and_autoclose():
    verts = [
        {"lat": -34.0, "lng": -62.0},
        {"lat": -34.0, "lng": -61.9},
        {"lat": -33.9, "lng": -61.9},
        {"lat": -33.9, "lng": -62.0},
    ]
    poly = to_polygon_manual_points(verts)  # sin cerrar -> auto-cierra
    ring = poly["coordinates"][0]
    assert ring[0] == ring[-1]
    assert len(ring) == 5
    area = polygon_area_ha(poly)
    assert area == pytest.approx(10270.0, rel=0.05)  # ~0.1deg x 0.1deg


def test_manual_points_geojson_order():
    poly = to_polygon_manual_points(
        [[-62.0, -34.0], [-61.9, -34.0], [-61.9, -33.9], [-62.0, -33.9]]
    )
    assert polygon_area_ha(poly) > 0


def test_manual_points_too_few():
    with pytest.raises(ValueError):
        to_polygon_manual_points([{"lat": -34.0, "lng": -62.0}])


def test_manual_bbox_rectangle():
    poly = to_polygon_manual_bbox(-34.0, -33.9, -62.0, -61.9)
    assert polygon_area_ha(poly) == pytest.approx(10270.0, rel=0.05)
    with pytest.raises(ValueError):
        to_polygon_manual_bbox(-33.9, -34.0, -62.0, -61.9)  # min>max


def test_validate_polygon_rejects_open_ring_and_huge():
    with pytest.raises(ValueError):
        validate_polygon({"type": "Polygon", "coordinates": [[[-62.0, -34.0], [-61.9, -34.0]]]})
    huge = to_polygon_manual_bbox(-55.0, -22.0, -74.0, -54.0)
    with pytest.raises(ValueError):
        validate_polygon(huge)


def test_normalize_legacy_empty():
    assert normalize_location(None) == {}
    assert normalize_location({}) == {}


def test_normalize_point_radius_mode():
    out = normalize_location({"lat": LAT, "lng": LNG, "ha": 5, "label": "lote 1"})
    assert out["origin"]["mode"] == "point_radius"
    assert out["origin"]["ha"] == 5.0
    assert out["centroid"]["lat"] == pytest.approx(LAT, abs=1e-6)
    assert out["area_ha"] == pytest.approx(5.0, rel=0.03)
    assert out["label"] == "lote 1"


def test_normalize_manual_modes():
    out = normalize_location(
        {"vertices": [{"lat": -34.0, "lng": -62.0}, {"lat": -34.0, "lng": -61.9}, {"lat": -33.9, "lng": -61.95}]}
    )
    assert out["origin"]["mode"] == "manual_points"
    out2 = normalize_location({"bbox": {"lat_min": -34.0, "lat_max": -33.9, "lng_min": -62.0, "lng_max": -61.9}})
    assert out2["origin"]["mode"] == "manual_bbox"
    assert out2["area_ha"] == pytest.approx(10270.0, rel=0.05)


def test_normalize_passthrough_polygon():
    poly = to_polygon_point_radius(LAT, LNG, 2.0)
    out = normalize_location({"polygon": poly})
    assert out["origin"]["mode"] == "polygon"
    assert out["area_ha"] == pytest.approx(2.0, rel=0.03)


def test_normalize_unknown_shape_rejects():
    with pytest.raises(ValueError):
        normalize_location({"provincia": "Buenos Aires"})
    with pytest.raises(ValueError):
        normalize_location({"lat": LAT})  # incompleto


def test_vertices_within_20ha_radius():
    poly = to_polygon_point_radius(LAT, LNG, 20.0)
    ring = poly["coordinates"][0][:-1]
    max_dist_m = max(
        math.hypot(
            (p[0] - LNG) * 111_320.0 * math.cos(math.radians(LAT)),
            (p[1] - LAT) * 111_320.0,
        )
        for p in ring
    )
    assert max_dist_m == pytest.approx(252.31, rel=0.02)
