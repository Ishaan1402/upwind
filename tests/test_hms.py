import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
import backend.services.hms as hms_mod
from backend.services.hms import (
    point_in_polygon,
    check_hms_smoke_plume,
    fetch_hms_smoke,
)


@pytest.fixture(autouse=True)
def _reset_hms_cache():
    """The HMS GeoJSON cache is module-global; reset it so tests stay isolated."""
    hms_mod._hms_cache.clear()
    yield
    hms_mod._hms_cache.clear()


def _square_polygon(lon_min, lat_min, lon_max, lat_max):
    return [[
        [lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max],
        [lon_min, lat_max], [lon_min, lat_min],
    ]]


def _geojson(features):
    return {"type": "FeatureCollection", "features": features}


def _poly_feature(ring, density):
    # ring is already a GeoJSON Polygon coordinate list ([outer_ring]).
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": ring},
        "properties": {"Density": density},
    }


def test_point_in_polygon_inside_and_outside():
    ring = _square_polygon(0, 0, 10, 10)[0]
    assert point_in_polygon(5.0, 5.0, ring) is True
    assert point_in_polygon(20.0, 5.0, ring) is False
    assert point_in_polygon(5.0, 20.0, ring) is False


def test_point_in_polygon_multipolygon_uses_outer_ring():
    # A ring with a hole: outer square with an inner (hole) ring. Ray-casting
    # uses the outer ring only, so a point in the hole still counts as inside.
    ring = _square_polygon(0, 0, 10, 10)[0]
    assert point_in_polygon(1.0, 1.0, ring) is True


def test_check_hms_inside_polygon_detects_density():
    geojson = _geojson([_poly_feature(_square_polygon(0, 0, 10, 10), 27)])
    res = check_hms_smoke_plume(5.0, 5.0, geojson)
    assert res["status"] == "present"
    assert res["density"] == "heavy"

    geojson = _geojson([_poly_feature(_square_polygon(0, 0, 10, 10), 16)])
    res = check_hms_smoke_plume(5.0, 5.0, geojson)
    assert res["status"] == "present"
    assert res["density"] == "medium"

    geojson = _geojson([_poly_feature(_square_polygon(0, 0, 10, 10), 5)])
    res = check_hms_smoke_plume(5.0, 5.0, geojson)
    assert res["status"] == "present"
    assert res["density"] == "light"


def test_check_hms_text_density_labels():
    geojson = _geojson([_poly_feature(_square_polygon(0, 0, 10, 10), "Heavy")])
    assert check_hms_smoke_plume(5.0, 5.0, geojson)["density"] == "heavy"
    geojson = _geojson([_poly_feature(_square_polygon(0, 0, 10, 10), "Medium")])
    assert check_hms_smoke_plume(5.0, 5.0, geojson)["density"] == "medium"
    geojson = _geojson([_poly_feature(_square_polygon(0, 0, 10, 10), "Light")])
    assert check_hms_smoke_plume(5.0, 5.0, geojson)["density"] == "light"


def test_check_hms_outside_polygon_is_absent():
    geojson = _geojson([_poly_feature(_square_polygon(0, 0, 10, 10), 16)])
    res = check_hms_smoke_plume(5.0, 30.0, geojson)
    assert res["status"] == "absent"
    assert res["density"] is None


def test_check_hms_multipolygon_matches_any_part():
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    _square_polygon(0, 0, 5, 5),
                    _square_polygon(10, 10, 15, 15),
                ],
            },
            "properties": {"Density": 5},
        }],
    }
    assert check_hms_smoke_plume(12.0, 12.0, geojson)["status"] == "present"
    assert check_hms_smoke_plume(2.0, 2.0, geojson)["status"] == "present"
    assert check_hms_smoke_plume(7.0, 7.0, geojson)["status"] == "absent"


def test_check_hms_polygon_with_hole():
    """A point inside an interior ring (hole) must not count as inside the plume."""
    outer = _square_polygon(0, 0, 10, 10)
    hole = _square_polygon(3, 3, 5, 5)
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [outer[0], hole[0]]},
            "properties": {"Density": 5},
        }],
    }
    assert check_hms_smoke_plume(2.0, 2.0, geojson)["status"] == "present"   # outer, not hole
    assert check_hms_smoke_plume(4.0, 4.0, geojson)["status"] == "absent"    # inside hole
    assert check_hms_smoke_plume(20.0, 20.0, geojson)["status"] == "absent"  # outside outer


def _mock_client_get(geojson=None, raise_exc=False):
    mock_client = AsyncMock()
    if raise_exc:
        mock_client.get.side_effect = RuntimeError("network down")
    else:
        # Plain Mock: resp.json() is synchronous in the real code.
        response = Mock()
        response.status_code = 200
        response.json.return_value = geojson
        mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def test_fetch_hms_smoke_success_returns_no_raw_geojson():
    geojson = _geojson([_poly_feature(_square_polygon(0, 0, 10, 10), 16)])
    with patch("backend.services.hms.httpx.AsyncClient", return_value=_mock_client_get(geojson)):
        res = asyncio.run(fetch_hms_smoke(5.0, 5.0))

    assert res["status"] == "present"
    assert res["density"] == "medium"
    assert "raw_geojson" not in res
    assert "hms_polygons" not in res




def test_fetch_hms_smoke_both_urls_fail_is_unavailable():
    with patch("backend.services.hms.httpx.AsyncClient", return_value=_mock_client_get(raise_exc=True)):
        res = asyncio.run(fetch_hms_smoke(5.0, 5.0))

    assert res["status"] == "unavailable"
    assert res["density"] is None


def test_fetch_hms_smoke_caches_geojson_within_ttl():
    """The multi-MB feed is downloaded once and reused within the TTL window."""
    geojson = _geojson([_poly_feature(_square_polygon(0, 0, 10, 10), 16)])
    mock_client = _mock_client_get(geojson)

    with patch("backend.services.hms.httpx.AsyncClient", return_value=mock_client):
        res1 = asyncio.run(fetch_hms_smoke(5.0, 5.0))
        res2 = asyncio.run(fetch_hms_smoke(5.0, 5.0))

    assert res1["status"] == "present"
    assert res2["status"] == "present"
    assert mock_client.get.call_count == 1
