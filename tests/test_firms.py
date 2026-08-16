import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from backend.services.firms import (
    calculate_bearing_degrees,
    calculate_haversine_distance,
    angular_difference,
    filter_upwind_hotspots,
    firms_search_radius_miles,
    cluster_firms_hotspots,
    FIRMS_MIN_RADIUS_MILES,
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


def _mock_fetch(csv_text: str, **fetch_kwargs):
    """Run fetch_firms_hotspots against a canned FIRMS CSV response."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.text = csv_text

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("backend.services.firms.FIRMS_MAP_KEY", "test-key"), \
         patch("backend.services.firms.httpx.AsyncClient", return_value=mock_client):
        return asyncio.run(fetch_firms_hotspots(**fetch_kwargs))


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
    assert firms_search_radius_miles(0.0) == FIRMS_MIN_RADIUS_MILES
    assert firms_search_radius_miles(3.0) == FIRMS_MIN_RADIUS_MILES
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
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.text = FIRMS_VIIRS_CSV_SAMPLE

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
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

    # Cluster response shape: clusters ranked by relevance, 'nearest' is the
    # strongest UPWIND cluster (not the strongest overall) and keeps every
    # field score.py reads.
    assert len(result["clusters"]) == result["total_count"]
    assert result["nearest"]["distance_miles"] > 0
    assert result["nearest"]["distance_km"] > 0
    assert result["nearest"]["bearing"] in ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    assert "detections" in result["nearest"]
    assert "age_hours" in result["nearest"]
    assert result["nearest"]["is_upwind"] is True
    assert result["clusters"][0]["relevance"] >= result["clusters"][1]["relevance"]
    # The strong downwind cluster (5.2 MW) may rank first overall without being
    # the named 'nearest' source; the weaker upwind cluster is the one reported.
    assert result["clusters"][0]["is_upwind"] is False
    assert result["clusters"][0]["frp"] == pytest.approx(5.2)
    assert result["nearest"]["frp"] == pytest.approx(1.9)  # cluster frp rounded to 0.1
    # Clusters expose peak intensity and max confidence weight.
    assert "peak_frp" in result["clusters"][0]
    assert "confidence_weight" in result["clusters"][0]

    # Raw pixel list retained for the frontend map layer.
    assert all(k in result["hotspots"][0] for k in
               ("lat", "lon", "frp", "distance_miles", "bearing", "bearing_deg", "is_upwind", "age_hours", "confidence"))

    # Query must cover all three active VIIRS NRT satellites with a 48h window.
    requested_url = mock_client.get.call_args.args[0]
    assert "VIIRS_SNPP_NRT,VIIRS_NOAA20_NRT,VIIRS_NOAA21_NRT" in requested_url
    assert requested_url.rstrip("/").endswith("/2")


def test_fetch_firms_relevance_ranks_high_frp_far_over_zero_frp_near():
    """Regression: a zero-FRP hotspot right next to the target must not
    outrank a much more intense hotspot slightly farther away; relevance
    (FRP x upwind x distance decay) governs ranking, not proximity."""
    csv = (
        f"{FIRMS_CSV_HEADER}\n"
        "43.586,-119.20,297.1,0.6,0.4,2026-07-27,940,N,VIIRS,n,2.0URT,287.1,0,N\n"
        "43.586,-119.60,301.0,0.6,0.4,2026-07-27,941,N,VIIRS,n,2.0URT,291.0,60,N\n"
    )
    result = _mock_fetch(csv, target_lat=43.586, target_lon=-119.054,
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
    result = _mock_fetch(csv, target_lat=43.586, target_lon=-119.054,
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
    result = _mock_fetch(csv, target_lat=43.586, target_lon=-119.054,
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
    """Two pixels within FIRMS_CLUSTER_RADIUS_KM merge into one cluster with
    summed FRP and detections == 2; that cluster outranks a lone zero-FRP pixel."""
    csv = (
        f"{FIRMS_CSV_HEADER}\n"
        "43.586,-119.10,297.1,0.6,0.4,2026-07-27,940,N,VIIRS,n,2.0URT,287.1,3,N\n"
        "43.586,-119.102,297.1,0.6,0.4,2026-07-27,941,N,VIIRS,n,2.0URT,287.1,3,N\n"
        "43.586,-119.30,297.1,0.6,0.4,2026-07-27,942,N,VIIRS,n,2.0URT,287.1,0,N\n"
    )
    result = _mock_fetch(csv, target_lat=43.586, target_lon=-119.054,
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
    result = _mock_fetch(csv, target_lat=43.586, target_lon=-119.054,
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
    result = _mock_fetch(csv, target_lat=43.586, target_lon=-119.054,
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
    result = _mock_fetch(csv, target_lat=43.586, target_lon=-119.054,
                         wind_dir_deg=270.0, wind_speed_mph=3.0, reference_utc=REFERENCE_UTC)

    assert result["status"] == "present"
    assert result["total_count"] == 2  # separate clusters, same FRP/age, both upwind
    assert result["clusters"][0]["confidence"] == "high"
    assert result["clusters"][0]["confidence_weight"] == 1.0
    assert result["clusters"][1]["confidence"] == "nominal"
    assert result["clusters"][1]["confidence_weight"] == pytest.approx(0.7)
    assert result["clusters"][0]["relevance"] > result["clusters"][1]["relevance"]
    assert result["nearest"]["confidence"] == "high"


def test_fetch_firms_nearest_is_upwind_when_stronger_downwind_exists():
    """A strong downwind cluster must not be reported as the upwind source:
    when any upwind cluster exists, 'nearest' is the strongest upwind cluster
    and alignment is 'upwind', even if a downwind cluster outranks it."""
    csv = (
        f"{FIRMS_CSV_HEADER}\n"
        "43.586,-119.20,297.1,0.6,0.4,2026-07-27,940,N,VIIRS,n,2.0URT,287.1,3,N\n"    # upwind, weak
        "43.586,-119.03,301.0,0.6,0.4,2026-07-27,940,N,VIIRS,n,2.0URT,291.0,60,N\n"   # downwind, strong
    )
    result = _mock_fetch(csv, target_lat=43.586, target_lon=-119.054,
                         wind_dir_deg=270.0, wind_speed_mph=3.0, reference_utc=REFERENCE_UTC)

    assert result["status"] == "present"
    assert result["alignment"] == "upwind"
    assert result["count"] == 1
    # The strong downwind cluster outranks overall but must not win 'nearest'.
    assert result["clusters"][0]["frp"] == 60.0
    assert result["clusters"][0]["is_upwind"] is False
    assert result["clusters"][0]["relevance"] > result["clusters"][1]["relevance"]
    # 'nearest' is the strongest upwind cluster, with the full score.py contract.
    assert result["nearest"]["is_upwind"] is True
    assert result["nearest"]["frp"] == 3.0
    assert result["nearest"]["frp"] < result["clusters"][0]["frp"]
    assert result["nearest"]["relevance"] == result["clusters"][1]["relevance"]
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
    persistence factor is capped at FIRMS_PERSISTENCE_CAP (1.6)."""
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
