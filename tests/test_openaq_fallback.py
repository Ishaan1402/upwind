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
         patch("backend.engine.signals.fetch_same_hour_baseline", new_callable=AsyncMock, return_value=None), \
         patch("backend.engine.signals.fetch_airnow_concentrations", new_callable=AsyncMock, return_value=None):
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


def test_prefers_monitor_with_both_pm_fractions_for_ratio():
    pm25_only = _candidate(111, "PM2.5 Only")
    full_site = _candidate(222, "Full Site")
    pm25 = {"value": 10.0, "unit": "µg/m³", "as_of": "2026-08-04T19:00:00+00:00"}
    signal = _run(
        [pm25_only, full_site],
        [
            _readings(pm25=pm25),
            _readings(pm25=pm25, pm10={"value": 25.0, "unit": "µg/m³", "as_of": "2026-08-04T19:00:00+00:00"}),
        ],
    )

    assert signal["status"] == "present"
    assert signal["monitor"]["location_id"] == 222
    assert signal["pm25_pm10_ratio"] == 0.4


def test_unavailable_when_no_candidate_has_fresh_readings():
    signal = _run([_candidate(111), _candidate(222)], [{}, {}])

    assert signal["status"] == "unavailable"
    assert "No fresh readings" in signal["details"]


def test_skips_baseline_calls_when_include_baselines_false():
    pm25 = {"value": 10.0, "unit": "µg/m³", "as_of": "2026-08-04T19:00:00+00:00"}
    with patch("backend.engine.signals.discover_reference_monitors", new_callable=AsyncMock, return_value=[_candidate(111)]), \
         patch("backend.engine.signals.fetch_latest", new_callable=AsyncMock, return_value=_readings(pm25=pm25)), \
         patch("backend.engine.signals.fetch_location_sensors", new_callable=AsyncMock, return_value={
             101: {"name": "pm25", "units": "µg/m³"},
         }), \
         patch("backend.engine.signals.fetch_daily_baseline", new_callable=AsyncMock) as daily, \
         patch("backend.engine.signals.fetch_same_hour_baseline", new_callable=AsyncMock) as same_hour, \
         patch("backend.engine.signals.fetch_airnow_concentrations", new_callable=AsyncMock, return_value=None):
        signal = asyncio.run(collect_openaq_signal(34.0, -118.0, include_baselines=False))

    assert signal["status"] == "present"
    assert signal["pm25"] == 10.0
    daily.assert_not_awaited()
    same_hour.assert_not_awaited()
