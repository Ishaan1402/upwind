import math
import random

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from backend.engine.params import DEFAULT
from backend.services.firms import (
    calculate_bearing_degrees,
    calculate_haversine_distance,
    angular_difference,
    angular_upwind_factor,
    filter_upwind_hotspots,
    firms_search_radius_miles,
    cluster_firms_hotspots,
    parse_firms_csv_rows,
    fetch_firms_hotspots,
)

# Real VIIRS_SNPP_NRT header + sample row from NASA FIRMS (Burns, OR area, 2026-07-27)
FIRMS_VIIRS_CSV_SAMPLE = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
43.6719,-119.12631,297.1,0.6,0.4,2026-07-27,940,N,VIIRS,n,2.0URT,287.1,1.86,N
43.5800,-119.0500,300.0,0.6,0.4,2026-07-27,941,N,VIIRS,n,2.0URT,290.0,5.20,N
"""

FIRMS_CSV_HEADER = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight"

# Fixed reference time for deterministic recency math in tests (backtest-style).
REFERENCE_UTC = datetime(2026, 7, 27, 9, 40, tzinfo=timezone.utc)


def _source_from_url(url: str) -> str:
    """Extract the FIRMS source name from a ``/area/csv/{key}/{source}/...`` URL."""
    return url.split("csv/")[1].split("/")[1]


def _mock_fetch(csv_by_source, **fetch_kwargs):
    """Run fetch_firms_hotspots against canned per-source FIRMS CSV responses.

    ``csv_by_source`` maps FIRMS source name -> CSV text; a bare string is
    served for the first source (VIIRS_SNPP_NRT) only. Sources without an entry
    respond with a header-only CSV (zero rows). Returns ``(result, requested)``
    where ``requested`` is the list of URLs fetched.
    """
    if isinstance(csv_by_source, str):
        csv_by_source = {"VIIRS_SNPP_NRT": csv_by_source}
    requested: list = []

    def _handler(url):
        requested.append(str(url))
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = csv_by_source.get(
            _source_from_url(url), FIRMS_CSV_HEADER + "\n"
        )
        return mock_response

    mock_client = AsyncMock()
    mock_client.get.side_effect = _handler
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("backend.services.firms.FIRMS_MAP_KEY", "test-key"), \
         patch("backend.services.firms.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(fetch_firms_hotspots(**fetch_kwargs))
    return result, requested


def test_calculate_bearing_degrees_cardinal():
    origin_lat, origin_lon = 34.0, -118.0
    
    # Due North
    b_north = calculate_bearing_degrees(origin_lat, origin_lon, 35.0, -118.0)
    assert abs(b_north - 0.0) < 1.0 or abs(b_north - 360.0) < 1.0

    # Due East
    b_east = calculate_bearing_degrees(origin_lat, origin_lon, 34.0, -117.0)
    assert abs(b_east - 90.0) < 2.0

    # Due South
    b_south = calculate_bearing_degrees(origin_lat, origin_lon, 33.0, -118.0)
    assert abs(b_south - 180.0) < 1.0

    # Due West
    b_west = calculate_bearing_degrees(origin_lat, origin_lon, 34.0, -119.0)
    assert abs(b_west - 270.0) < 2.0

def test_angular_difference():
    assert angular_difference(350, 10) == 20.0
    assert angular_difference(10, 350) == 20.0
    assert angular_difference(0, 180) == 180.0
    assert angular_difference(90, 90) == 0.0
    assert angular_difference(45, 135) == 90.0

def test_filter_upwind_hotspots_excludes_downwind():
    # wind_dir_deg = 0 means wind comes FROM 0° (North), blowing South.
    # The upwind bearing from the target is therefore 0° (where smoke sources
    # sit): hotspots within +/-90° of 0° (i.e. 270-90°) are upwind; those near
    # 180° are downwind.
    hotspots = [
        {"id": 1, "bearing_deg": 350.0},  # Upwind (diff 10 deg from 0)
        {"id": 2, "bearing_deg": 10.0},   # Upwind (diff 10 deg from 0)
        {"id": 3, "bearing_deg": 100.0},  # Downwind side (diff 100 deg from 0)
        {"id": 4, "bearing_deg": 170.0},  # Downwind (diff 170 deg from 0)
    ]
    
    upwind = filter_upwind_hotspots(hotspots, wind_dir_deg=0.0)
    upwind_ids = [h["id"] for h in upwind]
    
    assert 1 in upwind_ids
    assert 2 in upwind_ids
    assert 3 not in upwind_ids
    assert 4 not in upwind_ids

def test_filter_upwind_hotspots_east_wind_matches_eastern_fire():
    # Wind FROM the east (90°) carries smoke westward onto the target, so a
    # fire due east of the target (bearing 90°) is upwind and one due west
    # (bearing 270°) is downwind.
    hotspots = [
        {"id": 1, "bearing_deg": 90.0},   # Due east -> upwind
        {"id": 2, "bearing_deg": 100.0},  # Still within the upwind sector
        {"id": 3, "bearing_deg": 260.0},  # Due-west side -> downwind
    ]
    upwind = filter_upwind_hotspots(hotspots, wind_dir_deg=90.0)
    upwind_ids = [h["id"] for h in upwind]
    assert 1 in upwind_ids
    assert 2 in upwind_ids
    assert 3 not in upwind_ids

def test_filter_upwind_hotspots_no_wind_returns_all():
    hotspots = [
        {"id": 1, "bearing_deg": 10.0},
        {"id": 2, "bearing_deg": 180.0}
    ]
    res = filter_upwind_hotspots(hotspots, wind_dir_deg=None)
    assert len(res) == 2

def test_firms_search_radius_floors_calm_winds():
    assert firms_search_radius_miles(0.0) == DEFAULT.firms_min_radius_miles
    assert firms_search_radius_miles(3.0) == DEFAULT.firms_min_radius_miles
    assert firms_search_radius_miles(20.0) == 100.0
    assert firms_search_radius_miles(50.0) == 150.0

def test_parse_firms_csv_rows_reads_frp_by_header_not_index():
    """Regression: acq_date sits at index 5; old code tried float(parts[5]) and dropped every row."""
    rows = parse_firms_csv_rows(FIRMS_VIIRS_CSV_SAMPLE)
    assert len(rows) == 2
    assert rows[0]["lat"] == pytest.approx(43.6719)
    assert rows[0]["lon"] == pytest.approx(-119.12631)
    assert rows[0]["frp"] == pytest.approx(1.86)
    assert rows[1]["frp"] == pytest.approx(5.20)

def test_parse_firms_csv_rows_reads_recency_confidence_metadata():
    """acq_date/acq_time/confidence/satellite/daynight are parsed by header name;
    abbreviated NRT confidence labels ('n') normalize to 'nominal'."""
    rows = parse_firms_csv_rows(FIRMS_VIIRS_CSV_SAMPLE)
    assert rows[0]["acq_date"] == "2026-07-27"
    assert rows[0]["acq_time"] == "940"
    assert rows[0]["confidence"] == "nominal"
    assert rows[0]["satellite"] == "N"
    assert rows[0]["daynight"] == "N"

def test_parse_firms_csv_rows_missing_metadata_columns_default_none():
    """Schema shifts that drop acq_date/acq_time/confidence degrade gracefully:
    fields default to None (fetch treats missing acq as fresh, age 0)."""
    rows = parse_firms_csv_rows("latitude,longitude,frp\n43.5,-119.1,2.5\n")
    assert len(rows) == 1
    assert rows[0]["acq_date"] is None
    assert rows[0]["acq_time"] is None
    assert rows[0]["confidence"] is None
    assert rows[0]["satellite"] is None
    assert rows[0]["daynight"] is None

def test_parse_firms_csv_rows_empty_and_header_only():
    assert parse_firms_csv_rows("") == []
    assert parse_firms_csv_rows("latitude,longitude\n") == []

def test_fetch_firms_error_body_is_unavailable_not_absent():
    """FIRMS returns HTTP 200 with a plain-text error for bad keys / rate limits;
    that must be 'unavailable', never 'absent' (verified absence)."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.text = "Invalid MAP_KEY"

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("backend.services.firms.FIRMS_MAP_KEY", "bad-key"), \
         patch("backend.services.firms.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(fetch_firms_hotspots(43.586, -119.054))

    assert result["status"] == "unavailable"
    assert "Invalid MAP_KEY" in result["details"]

def test_fetch_firms_header_only_is_absent():
    """A well-formed CSV with a header but zero rows is a genuine absence."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.text = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("backend.services.firms.FIRMS_MAP_KEY", "test-key"), \
         patch("backend.services.firms.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(fetch_firms_hotspots(43.586, -119.054))

    assert result["status"] == "absent"


def test_fetch_firms_hotspots_parses_viirs_csv_response():
    requested_urls = []

    def _handler(url):
        requested_urls.append(str(url))
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = (
            FIRMS_VIIRS_CSV_SAMPLE if "/VIIRS_SNPP_NRT/" in url
            else FIRMS_CSV_HEADER + "\n"
        )
        return mock_response

    mock_client = AsyncMock()
    mock_client.get.side_effect = _handler
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("backend.services.firms.FIRMS_MAP_KEY", "test-key"), \
         patch("backend.services.firms.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(fetch_firms_hotspots(
            43.586, -119.054, wind_dir_deg=270.0, wind_speed_mph=3.0, reference_utc=REFERENCE_UTC
        ))

    assert result["status"] == "present"
    assert result["total_count"] == 2
    assert result["count"] >= 1
    assert result["nearest"] is not None
    assert result["nearest"]["frp"] > 0
    assert result["nearest"]["relevance"] > 0
    assert "strongest cluster" in result["details"]

    # Cluster response shape: clusters ranked by relevance; 'nearest' is the
    # top cluster by relevance across ALL clusters and keeps every field
    # score.py reads.
    assert len(result["clusters"]) == result["total_count"]
    assert result["nearest"]["distance_miles"] > 0
    assert result["nearest"]["distance_km"] > 0
    assert result["nearest"]["bearing"] in ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    assert "detections" in result["nearest"]
    assert "age_hours" in result["nearest"]
    assert result["clusters"][0]["relevance"] >= result["clusters"][1]["relevance"]
    # The strong downwind cluster (5.2 MW) ranks first overall AND claims the
    # named 'nearest' slot (top relevance across all clusters); alignment and
    # count still report the upwind subset.
    assert result["nearest"]["is_upwind"] is False
    assert result["clusters"][0]["is_upwind"] is False
    assert result["clusters"][0]["frp"] == pytest.approx(5.2)
    assert result["nearest"]["frp"] == pytest.approx(5.2)
    # Clusters expose peak intensity and max confidence weight.
    assert "peak_frp" in result["clusters"][0]
    assert "confidence_weight" in result["clusters"][0]

    # Raw pixel list retained for the frontend map layer.
    assert all(k in result["hotspots"][0] for k in
               ("lat", "lon", "frp", "distance_miles", "bearing", "bearing_deg", "is_upwind", "age_hours", "confidence"))

    # One request per active VIIRS NRT instrument, each a single-source URL
    # with a 48h window (never a comma-joined source list, which FIRMS rejects).
    assert len(requested_urls) == 3
    assert {_source_from_url(u) for u in requested_urls} == {
        "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT",
    }
    for url in requested_urls:
        assert "," not in _source_from_url(url)
        assert url.rstrip("/").endswith("/2")


def test_fetch_firms_relevance_ranks_high_frp_far_over_zero_frp_near():
    """Regression: a zero-FRP hotspot right next to the target must not
    outrank a much more intense hotspot slightly farther away; relevance
    (FRP x upwind x distance decay) governs ranking, not proximity."""
    csv = (
        f"{FIRMS_CSV_HEADER}\n"
        "43.586,-119.20,297.1,0.6,0.4,2026-07-27,940,N,VIIRS,n,2.0URT,287.1,0,N\n"
        "43.586,-119.60,301.0,0.6,0.4,2026-07-27,941,N,VIIRS,n,2.0URT,291.0,60,N\n"
    )
    result, _ = _mock_fetch(csv, target_lat=43.586, target_lon=-119.054,
                         wind_dir_deg=270.0, wind_speed_mph=3.0, reference_utc=REFERENCE_UTC)

    assert result["status"] == "present"
    assert result["hotspots"][0]["frp"] == 60.0
    assert result["hotspots"][0]["relevance"] > result["hotspots"][1]["relevance"]
    assert result["hotspots"][0]["distance_miles"] > result["hotspots"][1]["distance_miles"]
    # 'nearest' keeps its key name but now points at the strongest cluster.
    assert result["nearest"]["frp"] == 60.0
    # Both west of the target under a westerly wind, so both are upwind.
    assert all(h["is_upwind"] for h in result["hotspots"])
    assert result["hotspots"] == sorted(result["hotspots"], key=lambda h: h["relevance"], reverse=True)
    # The zero-FRP pixel alone forms a cluster below the FRP floor -> dropped.
    assert result["total_count"] == 1


def test_fetch_firms_recency_downweights_stale_detections_and_drops_over_window():
    """A 47h-old detection survives but is heavily downweighted (recency floor),
    while a 50h-old detection is outside the 48h window and dropped entirely."""
    csv = (
        f"{FIRMS_CSV_HEADER}\n"
        "43.586,-119.20,297.1,0.6,0.4,2026-07-25,1040,N,VIIRS,n,2.0URT,287.1,60,N\n"  # 47h old
        "43.586,-119.60,301.0,0.6,0.4,2026-07-27,940,N,VIIRS,n,2.0URT,291.0,60,N\n"   # fresh
        "43.600,-119.90,301.0,0.6,0.4,2026-07-25,0730,N,VIIRS,n,2.0URT,291.0,60,N\n"  # 50h old -> dropped
    )
    result, _ = _mock_fetch(csv, target_lat=43.586, target_lon=-119.054,
                         wind_dir_deg=270.0, wind_speed_mph=3.0, reference_utc=REFERENCE_UTC)

    assert result["status"] == "present"
    assert result["total_count"] == 2  # 50h-old pixel dropped; clusters = fresh + stale
    assert result["clusters"][0]["age_hours"] == 0.0  # fresh detection drives its cluster
    assert result["clusters"][1]["age_hours"] == pytest.approx(47.0)
    assert result["clusters"][0]["frp"] == result["clusters"][1]["frp"]  # identical FRP
    assert result["clusters"][0]["relevance"] > result["clusters"][1]["relevance"]
    assert result["nearest"]["age_hours"] == 0.0
    assert result["nearest"]["relevance"] > result["clusters"][1]["relevance"]
    # Dropped pixel's coordinate never appears in the raw map layer.
    assert all(h["lon"] != -119.90 for h in result["hotspots"])


def test_fetch_firms_drops_low_confidence_detections():
    """"low" confidence detections (sun glint / false positives) are dropped;
    nominal/high survive and the cluster label reflects the best member."""
    csv = (
        f"{FIRMS_CSV_HEADER}\n"
        "43.586,-119.20,297.1,0.6,0.4,2026-07-27,940,N,VIIRS,low,2.0URT,287.1,10,N\n"
        "43.586,-119.60,301.0,0.6,0.4,2026-07-27,940,N,VIIRS,high,2.0URT,291.0,5,N\n"
        "43.586,-119.80,301.0,0.6,0.4,2026-07-27,940,N,VIIRS,nominal,2.0URT,291.0,3,N\n"
    )
    result, _ = _mock_fetch(csv, target_lat=43.586, target_lon=-119.054,
                         wind_dir_deg=270.0, wind_speed_mph=3.0, reference_utc=REFERENCE_UTC)

    assert result["status"] == "present"
    assert len(result["hotspots"]) == 2
    assert result["total_count"] == 2
    assert all(h["confidence"] != "low" for h in result["hotspots"])
    assert all(h["lon"] != -119.20 for h in result["hotspots"])
    # Highest-FRP surviving cluster is the high-confidence one; the confidence
    # weight (1.0 vs 0.7) is carried onto each cluster.
    assert result["clusters"][0]["confidence"] == "high"
    assert result["clusters"][0]["confidence_weight"] == 1.0
    assert result["clusters"][1]["confidence"] == "nominal"
    assert result["clusters"][1]["confidence_weight"] == pytest.approx(0.7)


def test_fetch_firms_clusters_merge_pixels_with_summed_frp():
    """Two pixels within DEFAULT.firms_cluster_radius_km merge into one cluster with
    summed FRP and detections == 2; that cluster outranks a lone zero-FRP pixel."""
    csv = (
        f"{FIRMS_CSV_HEADER}\n"
        "43.586,-119.10,297.1,0.6,0.4,2026-07-27,940,N,VIIRS,n,2.0URT,287.1,3,N\n"
        "43.586,-119.102,297.1,0.6,0.4,2026-07-27,941,N,VIIRS,n,2.0URT,287.1,3,N\n"
        "43.586,-119.30,297.1,0.6,0.4,2026-07-27,942,N,VIIRS,n,2.0URT,287.1,0,N\n"
    )
    result, _ = _mock_fetch(csv, target_lat=43.586, target_lon=-119.054,
                         wind_dir_deg=270.0, wind_speed_mph=3.0, reference_utc=REFERENCE_UTC)

    assert result["status"] == "present"
    assert result["total_count"] == 1  # zero-FRP cluster dropped by FRP floor
    cluster = result["clusters"][0]
    assert cluster["detections"] == 2
    assert cluster["frp"] == 6.0       # summed FRP kept informational
    assert cluster["peak_frp"] == 3.0  # intensity = most intense single detection
    assert len(cluster["pixels"]) == 2
    assert cluster["lat"] == pytest.approx(43.586)
    assert cluster["lon"] == pytest.approx(-119.101, abs=0.001)
    # Merged cluster (relevance) outranks the lone zero-FRP pixel (relevance 0).
    assert result["hotspots"][0]["frp"] == 3.0
    assert result["hotspots"][-1]["frp"] == 0.0
    assert result["nearest"]["frp"] == 6.0
    assert result["nearest"]["detections"] == 2


def test_fetch_firms_wenatchee_high_frp_cluster_outranks_tiny_near_cluster():
    """Wenatchee-style regression: a tiny-FRP cluster nearer the target must NOT
    outrank a high-FRP cluster slightly farther away (relevance, not nearest)."""
    csv = (
        f"{FIRMS_CSV_HEADER}\n"
        "43.586,-119.09,297.1,0.6,0.4,2026-07-27,940,N,VIIRS,n,2.0URT,287.1,2,N\n"
        "43.586,-119.60,301.0,0.6,0.4,2026-07-27,940,N,VIIRS,n,2.0URT,291.0,60,N\n"
    )
    result, _ = _mock_fetch(csv, target_lat=43.586, target_lon=-119.054,
                         wind_dir_deg=270.0, wind_speed_mph=3.0, reference_utc=REFERENCE_UTC)

    assert result["status"] == "present"
    assert len(result["clusters"]) == 2
    assert result["clusters"][0]["frp"] == 60.0
    assert result["clusters"][1]["frp"] == 2.0
    # The high-FRP cluster is farther from the target but outranks the nearer one.
    assert result["clusters"][0]["distance_miles"] > result["clusters"][1]["distance_miles"]
    assert result["clusters"][0]["relevance"] > result["clusters"][1]["relevance"]
    assert result["nearest"]["frp"] == 60.0
    assert result["nearest"]["distance_miles"] > result["clusters"][1]["distance_miles"]


def test_fetch_firms_missing_acq_columns_treats_as_fresh():
    """Schema shifts that drop acq_date/acq_time degrade gracefully: age 0.0."""
    csv = "latitude,longitude,frp\n43.586,-119.20,5\n"
    result, _ = _mock_fetch(csv, target_lat=43.586, target_lon=-119.054,
                         wind_dir_deg=270.0, wind_speed_mph=3.0, reference_utc=REFERENCE_UTC)
    assert result["status"] == "present"
    assert result["hotspots"][0]["age_hours"] == 0.0
    assert result["total_count"] == 1


def test_fetch_firms_high_confidence_outranks_nominal():
    """At otherwise equal inputs a 'high'-confidence hotspot (weight 1.0) must
    outrank a 'nominal' one (weight 0.7) via the confidence term, even when the
    high-confidence detection sits slightly farther from the target."""
    csv = (
        f"{FIRMS_CSV_HEADER}\n"
        "43.586,-119.20,297.1,0.6,0.4,2026-07-27,940,N,VIIRS,nominal,2.0URT,287.1,10,N\n"
        "43.586,-119.25,297.1,0.6,0.4,2026-07-27,940,N,VIIRS,high,2.0URT,287.1,10,N\n"
    )
    result, _ = _mock_fetch(csv, target_lat=43.586, target_lon=-119.054,
                         wind_dir_deg=270.0, wind_speed_mph=3.0, reference_utc=REFERENCE_UTC)

    assert result["status"] == "present"
    assert result["total_count"] == 2  # separate clusters, same FRP/age, both upwind
    assert result["clusters"][0]["confidence"] == "high"
    assert result["clusters"][0]["confidence_weight"] == 1.0
    assert result["clusters"][1]["confidence"] == "nominal"
    assert result["clusters"][1]["confidence_weight"] == pytest.approx(0.7)
    assert result["clusters"][0]["relevance"] > result["clusters"][1]["relevance"]
    assert result["nearest"]["confidence"] == "high"


def test_fetch_firms_nearest_is_top_relevance_even_with_upwind_cluster():
    """Track C Part 3: 'nearest' is the top cluster by relevance across ALL
    clusters, so a far larger downwind fire claims the named slot; alignment
    stays 'upwind' and count still reflects the upwind subset (corroboration
    gating is untouched)."""
    csv = (
        f"{FIRMS_CSV_HEADER}\n"
        "43.586,-119.20,297.1,0.6,0.4,2026-07-27,940,N,VIIRS,n,2.0URT,287.1,3,N\n"    # upwind, weak
        "43.586,-119.03,301.0,0.6,0.4,2026-07-27,940,N,VIIRS,n,2.0URT,291.0,60,N\n"   # downwind, strong
    )
    result, _ = _mock_fetch(csv, target_lat=43.586, target_lon=-119.054,
                         wind_dir_deg=270.0, wind_speed_mph=3.0, reference_utc=REFERENCE_UTC)

    assert result["status"] == "present"
    assert result["alignment"] == "upwind"
    assert result["count"] == 1
    # The strong downwind cluster outranks overall and now wins 'nearest'.
    assert result["clusters"][0]["frp"] == 60.0
    assert result["clusters"][0]["is_upwind"] is False
    assert result["clusters"][0]["relevance"] > result["clusters"][1]["relevance"]
    # 'nearest' is the top-relevance cluster, with the full score.py contract.
    assert result["nearest"]["is_upwind"] is False
    assert result["nearest"]["frp"] == 60.0
    assert result["nearest"]["frp"] > result["clusters"][1]["frp"]
    assert result["nearest"]["relevance"] == result["clusters"][0]["relevance"]
    for field in ("distance_miles", "bearing", "distance_km", "bearing_deg",
                  "frp", "is_upwind", "relevance", "age_hours", "detections"):
        assert field in result["nearest"]


def test_cluster_firms_peak_frp_and_persistence():
    """cluster_firms_hotspots separates intensity (peak member FRP) from
    persistence (bounded detection-count multiplier); summed FRP is kept as an
    informational field only."""
    pixels = [
        {"lat": 43.586, "lon": -119.100, "frp": 3.0, "age_hours": 0.0,
         "confidence": "nominal", "confidence_weight": 0.7, "is_upwind": True},
        {"lat": 43.586, "lon": -119.102, "frp": 3.0, "age_hours": 0.0,
         "confidence": "nominal", "confidence_weight": 0.7, "is_upwind": True},
    ]
    cluster = cluster_firms_hotspots(pixels, 43.586, -119.054)[0]
    assert cluster["detections"] == 2
    assert cluster["frp"] == 6.0       # summed FRP stays informational
    assert cluster["peak_frp"] == 3.0  # intensity = most intense single detection
    assert cluster["confidence"] == "nominal"
    assert cluster["confidence_weight"] == pytest.approx(0.7)
    # persistence = 1.0 + 0.2 * (2-1) = 1.2, uncapped:
    # relevance = peak_frp * confidence_weight * persistence * upwind * decay * recency
    _, dist_mi = calculate_haversine_distance(43.586, -119.054, 43.586, -119.101)
    expected = round(3.0 * 0.7 * 1.2 * 4.0 * (1.0 / (dist_mi + 1.0)) * 1.0, 1)
    assert cluster["relevance"] == expected
    # Without the persistence boost (1.0 instead of 1.2) the cluster ranks lower.
    without_persistence = round(3.0 * 0.7 * 1.0 * 4.0 * (1.0 / (dist_mi + 1.0)), 1)
    assert expected > without_persistence


def test_cluster_firms_persistence_capped():
    """Five co-located detections would give 1.0 + 0.2*4 = 1.8, but the
    persistence factor is capped at DEFAULT.firms_persistence_cap (1.6)."""
    pixels = [
        {"lat": 43.586, "lon": -119.100, "frp": 2.0, "age_hours": 0.0,
         "confidence": "nominal", "confidence_weight": 0.7, "is_upwind": True},
        {"lat": 43.586, "lon": -119.101, "frp": 2.0, "age_hours": 0.0,
         "confidence": "nominal", "confidence_weight": 0.7, "is_upwind": True},
        {"lat": 43.586, "lon": -119.102, "frp": 2.0, "age_hours": 0.0,
         "confidence": "nominal", "confidence_weight": 0.7, "is_upwind": True},
        {"lat": 43.586, "lon": -119.103, "frp": 2.0, "age_hours": 0.0,
         "confidence": "nominal", "confidence_weight": 0.7, "is_upwind": True},
        {"lat": 43.586, "lon": -119.104, "frp": 2.0, "age_hours": 0.0,
         "confidence": "nominal", "confidence_weight": 0.7, "is_upwind": True},
    ]
    cluster = cluster_firms_hotspots(pixels, 43.586, -119.054)[0]
    assert cluster["detections"] == 5
    assert cluster["peak_frp"] == 2.0
    assert cluster["frp"] == 10.0
    _, dist_mi = calculate_haversine_distance(43.586, -119.054, 43.586, -119.102)
    capped = round(2.0 * 0.7 * 1.6 * 4.0 * (1.0 / (dist_mi + 1.0)) * 1.0, 1)
    uncapped = round(2.0 * 0.7 * 1.8 * 4.0 * (1.0 / (dist_mi + 1.0)) * 1.0, 1)
    assert cluster["relevance"] == capped
    assert capped != uncapped  # the cap genuinely binds at 5 detections


def test_fetch_firms_merges_hotspots_from_multiple_sources():
    """Hotspots from different NRT instruments are merged into one result, and
    each instrument is queried with its own single-source URL (FIRMS rejects
    comma-joined source lists with HTTP 400)."""
    noaa20_csv = (
        f"{FIRMS_CSV_HEADER}\n"
        "43.60,-119.10,299.0,0.6,0.4,2026-07-27,945,N,VIIRS,h,2.0URT,289.0,8.0,N\n"
    )
    result, requested = _mock_fetch(
        {"VIIRS_SNPP_NRT": FIRMS_VIIRS_CSV_SAMPLE, "VIIRS_NOAA20_NRT": noaa20_csv},
        target_lat=43.586, target_lon=-119.054,
        wind_dir_deg=270.0, wind_speed_mph=3.0, reference_utc=REFERENCE_UTC,
    )

    # Three single-source requests, never a comma-joined source list.
    assert len(requested) == 3
    assert {_source_from_url(u) for u in requested} == {
        "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT",
    }
    for url in requested:
        assert "," not in _source_from_url(url)
        assert url.rstrip("/").endswith("/2")

    # SNPP's two rows plus NOAA-20's one row are merged into one result.
    assert result["status"] == "present"
    lons = {round(h["lon"], 4) for h in result["hotspots"]}
    assert -119.1263 in lons  # SNPP row
    assert -119.1 in lons     # NOAA-20 row
    assert result["total_count"] == 3


def test_fetch_firms_partial_source_failure_keeps_healthy_sources():
    """A source that 400s must not poison the whole response: the healthy
    source's hotspots are still returned (never 'unavailable')."""
    requested_urls = []

    def _handler(url):
        requested_urls.append(str(url))
        mock_response = AsyncMock()
        if "/VIIRS_SNPP_NRT/" in url:
            mock_response.status_code = 400
            mock_response.text = "Invalid source"
        elif "/VIIRS_NOAA20_NRT/" in url:
            mock_response.status_code = 200
            mock_response.text = FIRMS_VIIRS_CSV_SAMPLE
        else:
            mock_response.status_code = 200
            mock_response.text = FIRMS_CSV_HEADER + "\n"
        return mock_response

    mock_client = AsyncMock()
    mock_client.get.side_effect = _handler
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("backend.services.firms.FIRMS_MAP_KEY", "test-key"), \
         patch("backend.services.firms.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(fetch_firms_hotspots(
            43.586, -119.054, wind_dir_deg=270.0, wind_speed_mph=3.0, reference_utc=REFERENCE_UTC
        ))

    assert len(requested_urls) == 3
    assert result["status"] == "present"
    assert result["total_count"] == 2  # NOAA-20's rows survive
    assert len(result["hotspots"]) == 2


def test_fetch_firms_all_sources_failed_is_unavailable():
    """Every instrument returning an HTTP error is an outage ('unavailable'),
    never verified absence."""
    mock_response = AsyncMock()
    mock_response.status_code = 400
    mock_response.text = "Invalid source"

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("backend.services.firms.FIRMS_MAP_KEY", "test-key"), \
         patch("backend.services.firms.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(fetch_firms_hotspots(43.586, -119.054))

    assert result["status"] == "unavailable"
    assert result["hotspots"] == []
    assert "all sources failed" in result["details"]


def _synthetic_hotspot(lat: float, lon: float, frp: float) -> dict:
    """Minimal pixel record in the shape fetch_firms_hotspots hands to clustering."""
    return {
        "lat": lat,
        "lon": lon,
        "frp": frp,
        "age_hours": 0.0,
        "is_upwind": True,
        "confidence": "nominal",
        "confidence_weight": 0.7,
        "distance_km": 0.0,
        "distance_miles": 0.0,
        "bearing": "N",
        "bearing_deg": 0.0,
        "relevance": 0.0,
    }


def _naive_cluster_signature(hotspots, cluster_radius_km=DEFAULT.firms_cluster_radius_km):
    """O(hotspots x clusters) reference: replicate the pre-grid greedy centroid
    clustering exactly and return the surviving clusters' signature as
    (ordered member (lat, lon) tuples, summed FRP rounded like the output, detections)."""
    ordered = sorted(hotspots, key=lambda h: h["frp"], reverse=True)
    member_groups = []
    centroids = []
    for h in ordered:
        joined = False
        for idx, (members, (clat, clon)) in enumerate(zip(member_groups, centroids)):
            dist_km, _ = calculate_haversine_distance(clat, clon, h["lat"], h["lon"])
            if dist_km <= cluster_radius_km:
                members.append(h)
                n = len(members)
                centroids[idx] = (
                    (clat * (n - 1) + h["lat"]) / n,
                    (clon * (n - 1) + h["lon"]) / n,
                )
                joined = True
                break
        if not joined:
            member_groups.append([h])
            centroids.append((h["lat"], h["lon"]))

    return [
        (tuple((m["lat"], m["lon"]) for m in g), round(sum(m["frp"] for m in g), 1), len(g))
        for g in member_groups
        if sum(m["frp"] for m in g) >= DEFAULT.firms_min_cluster_frp
    ]


def _grid_cluster_signature(clusters):
    return [
        (tuple((p["lat"], p["lon"]) for p in c["pixels"]), c["frp"], c["detections"])
        for c in clusters
    ]


def test_angular_upwind_factor_decays_with_angular_difference():
    """Track C Part 2 (pure function): the upwind multiplier is graded by the
    cosine of the angular difference from the upwind bearing - full 4x on-axis,
    ~3.1x at 45 deg off, and exactly 1x at/above 90 deg off (never below 1.0)."""
    factor = angular_upwind_factor
    assert factor(0.0, 4.0) == pytest.approx(4.0)
    assert factor(45.0, 4.0) == pytest.approx(1.0 + 3.0 * math.cos(math.radians(45.0)))
    assert factor(90.0, 4.0) == pytest.approx(1.0)
    assert factor(135.0, 4.0) == pytest.approx(1.0)
    assert factor(180.0, 4.0) == pytest.approx(1.0)
    assert factor(0.0, 4.0) > factor(45.0, 4.0) > factor(90.0, 4.0)
    # Bonus-neutral wind (bonus 1.0) always stays 1.0.
    assert factor(0.0, 1.0) == 1.0
    assert factor(90.0, 1.0) == 1.0


def test_fetch_firms_angular_decay_on_axis_outranks_off_axis():
    """Track C Part 2 integration: three identical-FRP, equidistant clusters at
    0, 45, and 90 degrees off the upwind bearing rank strictly by angular
    alignment (4x, ~3.1x, 1x) instead of every upwind-sector pixel counting as
    a full 'upwind' boost."""
    target_lat, target_lon = 45.0, -120.0

    def _pixel(bearing_deg: float, frp: float) -> str:
        # Place a pixel exactly 10 mi from the target at the given bearing.
        lat = target_lat + 10.0 * math.cos(math.radians(bearing_deg)) / 69.0
        lon = target_lon + 10.0 * math.sin(math.radians(bearing_deg)) / (69.0 * math.cos(math.radians(target_lat)))
        return f"{lat:.5f},{lon:.5f},297.1,0.6,0.4,2026-07-27,940,N,VIIRS,n,2.0URT,287.1,{frp},N"

    csv = f"{FIRMS_CSV_HEADER}\n" + "\n".join([
        _pixel(270.0, 20.0),  # on-axis (wind FROM west)
        _pixel(225.0, 20.0),  # 45 deg off
        _pixel(180.0, 20.0),  # 90 deg off (sector edge)
    ])
    result, _ = _mock_fetch(csv, target_lat=target_lat, target_lon=target_lon,
                         wind_dir_deg=270.0, wind_speed_mph=3.0, reference_utc=REFERENCE_UTC)

    assert result["status"] == "present"
    rel = {round(h["bearing_deg"]): h["relevance"] for h in result["hotspots"]}
    assert rel[270] > rel[225] > rel[180]
    # The 45-deg-off source keeps a real (but reduced) upwind boost.
    assert rel[225] > rel[180]
    # On-axis keeps the full 4x bonus; 90-deg-off drops to neutral.
    assert rel[270] == pytest.approx(rel[180] * 4.0, rel=0.05)


def test_cluster_firms_angular_decay_grades_by_cluster_bearing():
    """Track C Part 2 (cluster level): the cluster relevance multiplier uses the
    same graded angular factor when an upwind target bearing is supplied."""
    pixels = [
        {"lat": 43.586, "lon": -119.200, "frp": 3.0, "age_hours": 0.0,
         "confidence": "nominal", "confidence_weight": 0.7, "is_upwind": True},
    ]
    # On-axis cluster (due west of the target under a westerly wind).
    on_axis = cluster_firms_hotspots(pixels, 43.586, -119.054, upwind_target_deg=270.0)[0]
    assert on_axis["is_upwind"] is True
    _, dist_mi = calculate_haversine_distance(43.586, -119.054, 43.586, -119.200)
    full_bonus = round(3.0 * 0.7 * 1.0 * angular_upwind_factor(0.0, 4.0)
                       * (1.0 / (dist_mi + 1.0)) * 1.0, 1)
    assert on_axis["relevance"] == full_bonus
    # Same pixel, but upwind bearing 90 deg away -> neutral 1x.
    edge = cluster_firms_hotspots(pixels, 43.586, -119.054, upwind_target_deg=180.0)[0]
    assert edge["relevance"] < on_axis["relevance"]


def test_cluster_firms_grid_matches_naive_on_dense_synthetic_cloud():
    """Determinism/perf smoke test: a dense 300-hotspot patch (3 km across) plus
    300 scattered hotspots must cluster identically to the naive O(n x k)
    reference -- same cluster count, same summed FRP, same membership and
    member order -- and the grid result must be deterministic across calls.
    This exercises centroid drift across grid cells inside the dense patch."""
    rng = random.Random(42)
    patch_lat, patch_lon = 38.5, -122.5
    lat_deg = 3.0 / 111.0
    lon_deg = 3.0 / (111.0 * math.cos(math.radians(patch_lat)))
    hotspots = []
    for _ in range(300):
        hotspots.append(_synthetic_hotspot(
            patch_lat + rng.uniform(-lat_deg, lat_deg),
            patch_lon + rng.uniform(-lon_deg, lon_deg),
            rng.uniform(0.5, 60.0),
        ))
    for _ in range(300):
        hotspots.append(_synthetic_hotspot(
            38.0 + rng.uniform(-2.0, 2.0),
            -122.5 + rng.uniform(-2.0, 2.0),
            rng.uniform(0.1, 40.0),
        ))

    result = cluster_firms_hotspots(hotspots, 38.5, -122.5)
    # Deterministic: a second call returns a byte-identical payload.
    assert result == cluster_firms_hotspots(hotspots, 38.5, -122.5)

    naive_sig = _naive_cluster_signature(hotspots)
    # The grid result is relevance-sorted while the naive signature preserves
    # creation order, so compare canonical (sorted) multisets: identical clusters.
    assert sorted(_grid_cluster_signature(result)) == sorted(naive_sig)
    assert len(result) == len(naive_sig)
    assert sum(c["frp"] for c in result) == sum(frp for _, frp, _ in naive_sig)

