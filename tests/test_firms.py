import pytest
from backend.services.firms import (
    calculate_bearing_degrees,
    angular_difference,
    filter_upwind_hotspots
)

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
