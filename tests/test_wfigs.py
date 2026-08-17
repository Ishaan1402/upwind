import asyncio
from unittest.mock import AsyncMock, Mock, patch
from backend.services.wfigs import (
    fetch_wfigs_incident,
    _select_relevant_wildfires,
)


def _wf_feature(name, lon, lat, size=1000, contained=None, state="US-OR",
                county="Clackamas", modified_ms=1785600000000,
                cpx_id=None, cpx_name=None, is_cpx_child=False):
    props = {
        "IncidentName": name,
        "IncidentSize": size,
        "PercentContained": contained,
        "POOState": state,
        "POOCounty": county,
        "FireDiscoveryDateTime": 1770000000000,
        "ModifiedOnDateTime_dt": modified_ms,
        "IrwinID": "{TEST-IRWIN}",
        "GlobalID": "{TEST-GLOBAL}",
    }
    if cpx_id is not None:
        props["CpxID"] = cpx_id
    if cpx_name is not None:
        props["CpxName"] = cpx_name
    if is_cpx_child is not False and is_cpx_child is not None:
        props["IsCpxChild"] = is_cpx_child
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": props,
    }


def _geojson(features):
    return {"type": "FeatureCollection", "features": features}


def test_select_relevant_wildfires_distance_bearing_and_upwind():
    # Target at (45.0, -121.0); fire due east ~49 mi away.
    features = [
        _wf_feature("Distant Fire", -115.0, 45.0, contained=10),  # ~290 mi, within 300
        _wf_feature("Grasshopper Fire", -120.0, 45.0, contained=19, size=35000),
        _wf_feature("Contained Fire", -119.5, 45.0, contained=95),  # >90% contained -> skipped
        _wf_feature("", -120.2, 45.0),  # empty name -> skipped
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []}, "properties": {}},  # bad geom
    ]
    incidents = _select_relevant_wildfires(features, 45.0, -121.0, wind_dir_deg=90.0)

    # Ranked by relevance (size x activity x upwind x distance decay), not
    # nearest-first: the 35k-acre fire outranks the farther smaller one.
    assert [i["name"] for i in incidents] == ["Grasshopper Fire", "Distant Fire"]
    assert [i["relevance"] for i in incidents] == sorted(
        (i["relevance"] for i in incidents), reverse=True
    )
    nearest = incidents[0]
    assert nearest["size_acres"] == 35000
    assert nearest["percent_contained"] == 19
    assert nearest["state"] == "OR"  # "US-" prefix stripped
    assert nearest["county"] == "Clackamas"
    assert nearest["bearing"] == "E"
    assert 48.0 < nearest["distance_miles"] < 50.0
    assert nearest["is_upwind"] is True  # wind FROM east, fire east of target
    assert nearest["relevance"] > incidents[1]["relevance"]

    # With wind FROM the west, the same eastern fire is downwind.
    incidents_west = _select_relevant_wildfires([_wf_feature("Grasshopper Fire", -120.0, 45.0)], 45.0, -121.0, wind_dir_deg=270.0)
    assert incidents_west[0]["is_upwind"] is False


def test_select_relevant_wildfires_excludes_beyond_radius():
    features = [_wf_feature("Far Fire", -110.0, 45.0, contained=5)]  # ~530 mi away
    incidents = _select_relevant_wildfires(features, 45.0, -121.0, wind_dir_deg=None)
    assert incidents == []


def test_select_relevant_wildfires_wenatchee_prefers_large_upwind_fire():
    """Wenatchee regression: 'Three Queens' (~3600 ac) must beat the much
    smaller 'Taneum Creek' (~16 ac) as the smoke source even though it sits
    ~15 mi farther west (both fires are upwind under a westerly wind)."""
    target_lat, target_lon = 47.42, -120.31
    features = [
        _wf_feature("Taneum Creek", -121.17, 47.42, size=16, contained=None),
        _wf_feature("Three Queens", -121.49, 47.42, size=3600, contained=None),
    ]
    incidents = _select_relevant_wildfires(features, target_lat, target_lon, wind_dir_deg=270.0)

    assert len(incidents) == 2
    assert incidents[0]["name"] == "Three Queens"
    assert incidents[0]["size_acres"] == 3600
    assert incidents[0]["is_upwind"] is True
    assert 54.0 < incidents[0]["distance_miles"] < 56.0
    assert incidents[0]["relevance"] > incidents[1]["relevance"]

    # Through the fetch path, the top-ranked incident is selected as upwind.
    with patch("backend.services.wfigs.httpx.AsyncClient", return_value=_mock_client(_geojson(features))):
        res = asyncio.run(fetch_wfigs_incident(target_lat, target_lon, wind_dir_deg=270.0))

    assert res["status"] == "present"
    assert res["incident"]["name"] == "Three Queens"
    assert res["alignment"] == "upwind"
    assert res["count"] == 2
    assert res["candidates"][0]["name"] == "Three Queens"


def test_select_relevant_wildfires_collapses_complexes():
    """Two children sharing a CpxID collapse into one representative incident
    with the summed size and the complex name, and the complex outranks a
    small nearby lone fire."""
    features = [
        _wf_feature("Child One", -121.50, 45.0, size=100, contained=0,
                    cpx_id="CPX-42", cpx_name="Big Complex", is_cpx_child=True),
        _wf_feature("Child Two", -121.80, 45.0, size=900, contained=20,
                    cpx_id="CPX-42", cpx_name="Big Complex", is_cpx_child=True),
        _wf_feature("Little Fire", -121.30, 45.0, size=50, contained=0),
    ]
    incidents = _select_relevant_wildfires(features, 45.0, -121.0, wind_dir_deg=270.0)

    assert len(incidents) == 2
    top = incidents[0]
    assert top["name"] == "Big Complex"
    assert top["size_acres"] == 1000.0  # SUM of the two children
    assert top["percent_contained"] == 0.0  # least-contained child governs
    assert top["is_cpx_child"] is True
    assert top["is_upwind"] is True
    # Distance/bearing come from the nearest child point.
    assert 23.0 < top["distance_miles"] < 26.0
    assert top["bearing"] == "W"
    assert top["relevance"] > incidents[1]["relevance"]
    assert incidents[1]["name"] == "Little Fire"


def test_select_relevant_wildfires_groups_children_by_name_without_id():
    """Children flagged as complex members but lacking a CpxID still collapse
    when they share the same complex name."""
    features = [
        _wf_feature("Prong A", -121.5, 45.0, size=400, contained=10,
                    cpx_name="North Complex", is_cpx_child=True),
        _wf_feature("Prong B", -121.8, 45.0, size=600, contained=40,
                    cpx_name="North Complex", is_cpx_child=True),
    ]
    incidents = _select_relevant_wildfires(features, 45.0, -121.0, wind_dir_deg=270.0)

    assert len(incidents) == 1
    assert incidents[0]["name"] == "North Complex"
    assert incidents[0]["size_acres"] == 1000.0
    assert incidents[0]["percent_contained"] == 10.0


def test_select_relevant_wildfires_prefers_upwind_over_nearer_downwind():
    """Second sub-bug: the nearest fire is DOWNWIND but a larger UPWIND fire
    is slightly farther away; relevance must select the upwind fire and the
    fetch result must report alignment='upwind'."""
    target_lat, target_lon = 45.0, -121.0
    features = [
        _wf_feature("Downwind Near", -120.4, 45.0, size=2000, contained=10),
        _wf_feature("Upwind Big", -121.7, 45.0, size=20000, contained=10),
    ]
    incidents = _select_relevant_wildfires(features, target_lat, target_lon, wind_dir_deg=270.0)

    assert incidents[0]["name"] == "Upwind Big"
    assert incidents[0]["is_upwind"] is True
    assert incidents[1]["name"] == "Downwind Near"
    assert incidents[1]["is_upwind"] is False

    with patch("backend.services.wfigs.httpx.AsyncClient", return_value=_mock_client(_geojson(features))):
        res = asyncio.run(fetch_wfigs_incident(target_lat, target_lon, wind_dir_deg=270.0))

    assert res["incident"]["name"] == "Upwind Big"
    assert res["alignment"] == "upwind"


def test_select_relevant_wildfires_parses_string_typed_complex_fields():
    """ArcGIS may serialize booleans as strings; 'false' must not collapse and
    'true' must group by complex name."""
    features = [
        _wf_feature("Lone Fire", -121.2, 45.0, size=100,
                    cpx_name="Lone Complex", is_cpx_child="false"),
        _wf_feature("Prong One", -121.5, 45.0, size=300,
                    cpx_name="String Complex", is_cpx_child="true"),
        _wf_feature("Prong Two", -121.8, 45.0, size=700,
                    cpx_name="String Complex", is_cpx_child="true"),
    ]
    incidents = _select_relevant_wildfires(features, 45.0, -121.0, wind_dir_deg=270.0)

    assert len(incidents) == 2
    assert incidents[0]["name"] == "String Complex"
    assert incidents[0]["size_acres"] == 1000.0
    assert incidents[1]["name"] == "Lone Fire"


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


def test_fetch_wfigs_present_selects_top_and_alignment():
    geojson = _geojson([_wf_feature("Grasshopper Fire", -120.0, 45.0, size=35000, contained=19)])
    with patch("backend.services.wfigs.httpx.AsyncClient", return_value=_mock_client(geojson)):
        res = asyncio.run(fetch_wfigs_incident(45.0, -121.0, wind_dir_deg=90.0))

    assert res["status"] == "present"
    assert res["incident"]["name"] == "Grasshopper Fire"
    assert res["alignment"] == "upwind"
    assert res["count"] == 1
    assert res["candidates"][0]["name"] == "Grasshopper Fire"
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
