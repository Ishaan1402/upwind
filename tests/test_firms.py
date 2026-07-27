import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from backend.services.firms import (
    calculate_bearing_degrees,
    angular_difference,
    filter_upwind_hotspots,
    firms_search_radius_miles,
    FIRMS_MIN_RADIUS_MILES,
    parse_firms_csv_rows,
    fetch_firms_hotspots,
)

# Real VIIRS_SNPP_NRT header + sample row from NASA FIRMS (Burns, OR area, 2026-07-27)
FIRMS_VIIRS_CSV_SAMPLE = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
43.6719,-119.12631,297.1,0.6,0.4,2026-07-27,940,N,VIIRS,n,2.0URT,287.1,1.86,N
43.5800,-119.0500,300.0,0.6,0.4,2026-07-27,941,N,VIIRS,n,2.0URT,290.0,5.20,N
"""

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
    # Therefore upwind bearing is (0 + 180) % 360 = 180° (North of user).
    hotspots = [
        {"id": 1, "bearing_deg": 170.0},  # Upwind (diff 10 deg from 180)
        {"id": 2, "bearing_deg": 100.0},  # Upwind (diff 80 deg from 180)
        {"id": 3, "bearing_deg": 260.0},  # Upwind (diff 80 deg from 180)
        {"id": 4, "bearing_deg": 10.0},   # Downwind (diff 170 deg from 180)
    ]
    
    upwind = filter_upwind_hotspots(hotspots, wind_dir_deg=0.0)
    upwind_ids = [h["id"] for h in upwind]
    
    assert 1 in upwind_ids
    assert 2 in upwind_ids
    assert 3 in upwind_ids
    assert 4 not in upwind_ids

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

def test_parse_firms_csv_rows_empty_and_header_only():
    assert parse_firms_csv_rows("") == []
    assert parse_firms_csv_rows("latitude,longitude\n") == []

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
        result = asyncio.run(fetch_firms_hotspots(43.586, -119.054, wind_dir_deg=270.0, wind_speed_mph=3.0))

    assert result["status"] == "present"
    assert result["total_count"] == 2
    assert result["count"] >= 1
    assert result["nearest"] is not None
    assert result["nearest"]["frp"] > 0
