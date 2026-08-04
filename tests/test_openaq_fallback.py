"""Freshness-aware monitor fallback for the OpenAQ concentration signal."""

import asyncio
from unittest.mock import AsyncMock, patch

from backend.engine.signals import collect_openaq_signal


def _candidate(location_id, name="Monitor"):
    return {
        "location_id": location_id,
        "name": name,
        "distance_km": 5.0,
        "timezone": "America/Los_Angeles",
        "provider": "US EPA",
        "owner": "State Agency",
    }


def _readings(**overrides):
    readings = {
        "co": {"value": 0.4, "unit": "ppm", "as_of": "2026-08-04T19:00:00+00:00"},
        "no2": {"value": 10.0, "unit": "ppb", "as_of": "2026-08-04T19:00:00+00:00"},
    }
    readings.update(overrides)
    return readings


def _run(monitors, latest_side_effect):
    with patch("backend.engine.signals.discover_reference_monitors", new_callable=AsyncMock, return_value=monitors), \
         patch("backend.engine.signals.fetch_latest", new_callable=AsyncMock, side_effect=latest_side_effect), \
         patch("backend.engine.signals.fetch_location_sensors", new_callable=AsyncMock, return_value={
             101: {"name": "pm25", "units": "µg/m³"},
             102: {"name": "co", "units": "ppm"},
         }), \
         patch("backend.engine.signals.fetch_daily_baseline", new_callable=AsyncMock, return_value=None), \
         patch("backend.engine.signals.fetch_same_hour_baseline", new_callable=AsyncMock, return_value=None):
        return asyncio.run(collect_openaq_signal(34.0, -118.0))


def test_falls_back_to_live_monitor_when_nearest_is_stale():
    stale = _candidate(111, "Dead Feed")
    live = _candidate(222, "Live Monitor")
    signal = _run([stale, live], [{}, _readings(pm25={"value": 12.0, "unit": "µg/m³", "as_of": "2026-08-04T19:00:00+00:00"})])

    assert signal["status"] == "present"
    assert signal["monitor"]["location_id"] == 222
    assert signal["pm25"] == 12.0


def test_prefers_monitor_with_live_pm25_over_any_readings():
    no_pm25 = _candidate(111, "CO Only")
    with_pm25 = _candidate(222, "Full Site")
    signal = _run(
        [no_pm25, with_pm25],
        [_readings(), _readings(pm25={"value": 9.5, "unit": "µg/m³", "as_of": "2026-08-04T19:00:00+00:00"})],
    )

    assert signal["status"] == "present"
    assert signal["monitor"]["location_id"] == 222
    assert signal["pm25"] == 9.5


def test_unavailable_when_no_candidate_has_fresh_readings():
    signal = _run([_candidate(111), _candidate(222)], [{}, {}])

    assert signal["status"] == "unavailable"
    assert "No fresh readings" in signal["details"]
