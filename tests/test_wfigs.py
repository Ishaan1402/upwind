import asyncio
from unittest.mock import AsyncMock, Mock, patch
from backend.services.wfigs import (
    fetch_wfigs_incident,
    _select_nearest_wildfire,
    WFIGS_MAX_RADIUS_MILES,
)


def _wf_feature(name, lon, lat, size=1000, contained=None, state="US-OR",
                county="Clackamas", modified_ms=1785600000000):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "IncidentName": name,
            "IncidentSize": size,
            "PercentContained": contained,
            "POOState": state,
            "POOCounty": county,
            "FireDiscoveryDateTime": 1770000000000,
            "ModifiedOnDateTime_dt": modified_ms,
            "IrwinID": "{TEST-IRWIN}",
            "GlobalID": "{TEST-GLOBAL}",
        },
    }


def _geojson(features):
    return {"type": "FeatureCollection", "features": features}


def test_select_nearest_wildfire_distance_bearing_and_upwind():
    # Target at (45.0, -121.0); fire due east ~49 mi away.
    features = [
        _wf_feature("Distant Fire", -115.0, 45.0, contained=10),  # ~290 mi, within 300
        _wf_feature("Grasshopper Fire", -120.0, 45.0, contained=19, size=35000),
        _wf_feature("Contained Fire", -119.5, 45.0, contained=95),  # >90% contained -> skipped
        _wf_feature("", -120.2, 45.0),  # empty name -> skipped
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []}, "properties": {}},  # bad geom
    ]
    incidents = _select_nearest_wildfire(features, 45.0, -121.0, wind_dir_deg=90.0)

    assert [i["name"] for i in incidents] == ["Grasshopper Fire", "Distant Fire"]
    nearest = incidents[0]
    assert nearest["size_acres"] == 35000
    assert nearest["percent_contained"] == 19
    assert nearest["state"] == "OR"  # "US-" prefix stripped
    assert nearest["county"] == "Clackamas"
    assert nearest["bearing"] == "E"
    assert 48.0 < nearest["distance_miles"] < 50.0
    assert nearest["is_upwind"] is True  # wind FROM east, fire east of target

    # With wind FROM the west, the same eastern fire is downwind.
    incidents_west = _select_nearest_wildfire([_wf_feature("Grasshopper Fire", -120.0, 45.0)], 45.0, -121.0, wind_dir_deg=270.0)
    assert incidents_west[0]["is_upwind"] is False


def test_select_nearest_wildfire_excludes_beyond_radius():
    features = [_wf_feature("Far Fire", -110.0, 45.0, contained=5)]  # ~530 mi away
    incidents = _select_nearest_wildfire(features, 45.0, -121.0, wind_dir_deg=None)
    assert incidents == []


def _mock_client(geojson=None, raise_exc=False, status=200):
    mock_client = AsyncMock()
    if raise_exc:
        mock_client.get.side_effect = RuntimeError("network down")
    else:
        # Plain Mock: resp.json() is synchronous in the real code.
        response = Mock()
        response.status_code = status
        response.json.return_value = geojson
        mock_client.get.return_value = response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


def test_fetch_wfigs_present_selects_nearest_and_alignment():
    geojson = _geojson([_wf_feature("Grasshopper Fire", -120.0, 45.0, size=35000, contained=19)])
    with patch("backend.services.wfigs.httpx.AsyncClient", return_value=_mock_client(geojson)):
        res = asyncio.run(fetch_wfigs_incident(45.0, -121.0, wind_dir_deg=90.0))

    assert res["status"] == "present"
    assert res["incident"]["name"] == "Grasshopper Fire"
    assert res["alignment"] == "upwind"
    assert res["count"] == 1
    assert "Grasshopper Fire" in res["details"]
    assert "35,000 acres" in res["details"].replace("35000", "35,000") or "35000" in res["details"]


def test_fetch_wfigs_absent_when_no_qualifying_incident():
    geojson = _geojson([_wf_feature("Contained Fire", -120.0, 45.0, contained=95)])
    with patch("backend.services.wfigs.httpx.AsyncClient", return_value=_mock_client(geojson)):
        res = asyncio.run(fetch_wfigs_incident(45.0, -121.0))

    assert res["status"] == "absent"
    assert res["incident"] is None
    assert res["alignment"] is None


def test_fetch_wfigs_unavailable_when_feed_down():
    with patch("backend.services.wfigs.httpx.AsyncClient", return_value=_mock_client(raise_exc=True)):
        res = asyncio.run(fetch_wfigs_incident(45.0, -121.0))

    assert res["status"] == "unavailable"
    assert res["incident"] is None


def test_fetch_wfigs_error_payload_is_unavailable():
    """ArcGIS can return HTTP 200 with {"error": {...}}; that is an outage, not absence."""
    mock_client = _mock_client(geojson={"error": {"code": 400, "message": "Cannot perform query"}})
    with patch("backend.services.wfigs.httpx.AsyncClient", return_value=mock_client):
        res = asyncio.run(fetch_wfigs_incident(45.0, -121.0))

    assert res["status"] == "unavailable"
    assert res["incident"] is None


def test_fetch_wfigs_non_200_falls_through_to_unavailable():
    with patch("backend.services.wfigs.httpx.AsyncClient", return_value=_mock_client(status=500)):
        res = asyncio.run(fetch_wfigs_incident(45.0, -121.0))

    assert res["status"] == "unavailable"


def test_fetch_wfigs_sends_native_spatial_filter():
    """The query must use ArcGIS point-distance filtering so the top-200 result
    cap applies within 300 mi, not nationwide."""
    geojson = _geojson([_wf_feature("Grasshopper Fire", -120.0, 45.0, contained=19)])
    mock_client = _mock_client(geojson)
    with patch("backend.services.wfigs.httpx.AsyncClient", return_value=mock_client):
        asyncio.run(fetch_wfigs_incident(45.0, -121.0, wind_dir_deg=90.0))

    params = mock_client.get.call_args.kwargs["params"]
    assert params["geometry"] == "-121.0,45.0"
    assert params["geometryType"] == "esriGeometryPoint"
    assert params["spatialRel"] == "esriSpatialRelIntersects"
    assert float(params["distance"]) == 300.0
    assert params["units"] == "esriSRUnit_StatuteMile"
    assert params["inSR"] == 4326


def test_fetch_wfigs_parses_percent_and_size_strings():
    """ArcGIS may return numeric fields as strings or percentages; parsing must
    be defensive so '95%' incidents are filtered and sizes still parse."""
    geojson = _geojson([
        _wf_feature("Contained Pct String", -120.0, 45.0, contained="95%"),
        _wf_feature("Active Pct String", -119.5, 45.0, contained="19%", size="35000"),
    ])
    with patch("backend.services.wfigs.httpx.AsyncClient", return_value=_mock_client(geojson)):
        res = asyncio.run(fetch_wfigs_incident(45.0, -121.0))

    assert res["status"] == "present"
    assert res["incident"]["name"] == "Active Pct String"
    assert res["incident"]["percent_contained"] == 19.0
    assert res["incident"]["size_acres"] == 35000.0
    assert "35,000 acres" in res["details"]
