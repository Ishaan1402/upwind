"""Unit tests for the AirNow /aq/data/ PM2.5/PM10 ratio fallback and AQI observation."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.engine.signals import collect_openaq_signal
from backend.services.airnow import fetch_airnow_concentrations, fetch_airnow_observation


def _row(parameter, value, lat=34.01, lon=-118.0, utc="2026-08-04T20:00:00",
         site_name="Site A", aqsid="060370001", raw=None):
    row = {
        "Latitude": lat,
        "Longitude": lon,
        "UTC": utc,
        "Parameter": parameter,
        "Value": value,
        "Unit": "UG/M3",
        "SiteName": site_name,
        "AQSID": aqsid,
    }
    if raw is not None:
        row["RawConcentration"] = raw
    return row


def _obs_row(parameter, aqi, utc="2026-08-04T20:00:00", lat=34.01, lon=-118.0,
             site_name="Site A", aqsid="060370001"):
    """PascalCase /aq/data/ row with an AQI (not a concentration)."""
    return {
        "Latitude": lat,
        "Longitude": lon,
        "UTC": utc,
        "Parameter": parameter,
        "AQI": aqi,
        "Unit": "index",
        "SiteName": site_name,
        "AQSID": aqsid,
    }


def make_client(rows, status=200, json_raises=None):
    resp = Mock()
    resp.status_code = status
    if json_raises is not None:
        resp.json.side_effect = json_raises
    else:
        resp.json.return_value = rows

    client = Mock()

    async def fake_get(url, params=None, **kwargs):
        return resp

    client.get = AsyncMock(side_effect=fake_get)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _fetch(rows, lat=34.0, lon=-118.0, status=200, json_raises=None):
    client = make_client(rows, status=status, json_raises=json_raises)
    with patch("backend.services.airnow.httpx.AsyncClient", return_value=client), \
         patch("backend.services.airnow.AIRNOW_KEY", "test-key"):
        result = asyncio.run(fetch_airnow_concentrations(lat, lon))
    return result, client


def _fetch_observation(rows, lat=34.0, lon=-118.0, status=200, json_raises=None):
    client = make_client(rows, status=status, json_raises=json_raises)
    with patch("backend.services.airnow.httpx.AsyncClient", return_value=client), \
         patch("backend.services.airnow.AIRNOW_KEY", "test-key"):
        result = asyncio.run(fetch_airnow_observation(lat, lon))
    return result, client


def test_picks_nearest_site_with_both_pm_fractions():
    rows = [
        _row("PM2.5", 10.0, lat=34.01, site_name="Near", aqsid="1"),
        _row("PM10", 25.0, lat=34.01, site_name="Near", aqsid="1"),
        _row("PM2.5", 30.0, lat=34.3, site_name="Far", aqsid="2"),
        _row("PM10", 100.0, lat=34.3, site_name="Far", aqsid="2"),
    ]
    result, client = _fetch(rows)

    assert result["pm25"] == 10.0
    assert result["pm10"] == 25.0
    assert result["pm25_pm10_ratio"] == 0.4
    assert result["site"]["name"] == "Near"
    assert result["site"]["aqsid"] == "1"
    assert result["site"]["distance_km"] == pytest.approx(1.11, abs=0.01)
    assert result["site"]["as_of"] == "2026-08-04T20:00:00+00:00"

    # Request shape: 3h lookback, both PM parameters, +/-0.25 deg BBOX.
    url, params = client.get.await_args.args[0], client.get.await_args.kwargs["params"]
    assert url == "https://www.airnowapi.org/aq/data/"
    assert params["parameters"] == "PM25,PM10"
    assert params["dataType"] == "B"
    assert params["monitorType"] == 0
    assert params["includerawconcentrations"] == 1
    assert params["format"] == "application/json"
    assert params["verbose"] == 1
    assert params["API_KEY"] == "test-key"
    assert params["BBOX"] == "-118.25,33.75,-117.75,34.25"
    assert params["startDate"].endswith("T") is False  # YYYY-MM-DDTHH form
    assert "T" in params["startDate"] and ":" not in params["startDate"]


def test_raw_concentration_fallback_and_latest_hour_wins():
    rows = [
        # -999 Value falls back to the raw concentration
        _row("PM2.5", -999, utc="2026-08-04T19:00:00", raw=10.0),
        # Later hour, null Value, valid raw -> raw used and latest wins
        _row("PM2.5", None, utc="2026-08-04T20:00:00", raw=12.0),
        # Both missing -> row dropped entirely
        _row("PM2.5", -999, utc="2026-08-04T21:00:00", raw=-999),
        _row("PM10", -999, utc="2026-08-04T19:00:00", raw=40.0),
        _row("PM10", None, utc="2026-08-04T20:00:00", raw=48.0),
    ]
    result, _ = _fetch(rows)

    assert result["pm25"] == 12.0
    assert result["pm10"] == 48.0
    assert result["pm25_pm10_ratio"] == pytest.approx(0.25)


def test_negative_values_clamped_and_bad_rows_skipped():
    rows = [
        # Negative PM2.5 clamped to 0 and kept (latest valid hour)
        _row("PM2.5", -5.0, utc="2026-08-04T19:00:00"),
        # 21:00 null with no raw -> skipped, so the clamped 19:00 value wins
        _row("PM2.5", None, utc="2026-08-04T21:00:00"),
        # PM10 null with null raw -> skipped
        _row("PM10", None, utc="2026-08-04T20:00:00"),
        _row("PM10", 24.0, utc="2026-08-04T19:00:00"),
    ]
    result, _ = _fetch(rows)

    assert result["pm25"] == 0.0
    assert result["pm10"] == 24.0
    assert result["pm25_pm10_ratio"] == 0.0


def test_none_when_no_site_has_both_pm_fractions():
    rows = [
        _row("PM2.5", 10.0, site_name="PM25 Only", aqsid="1"),
        _row("PM10", 30.0, site_name="PM10 Only", aqsid="2", lat=34.2),
    ]
    result, _ = _fetch(rows)
    assert result is None

    # Non-PM parameters are ignored entirely.
    rows = [
        {"Latitude": 34.0, "Longitude": -118.0, "UTC": "2026-08-04T20:00:00",
         "Parameter": "O3", "Value": 50.0, "Unit": "PPB",
         "SiteName": "Site", "AQSID": "1"},
    ]
    result, _ = _fetch(rows)
    assert result is None


def test_degrades_on_http_error_and_malformed_payload():
    result, _ = _fetch([], status=500)
    assert result is None

    result, _ = _fetch({"not": "a list"})
    assert result is None

    result, _ = _fetch([])
    assert result is None

    result, _ = _fetch([_row("PM2.5", 10.0), _row("PM10", 20.0)], json_raises=ValueError("bad json"))
    assert result is None

    client = make_client([_row("PM2.5", 10.0), _row("PM10", 20.0)])
    with patch("backend.services.airnow.httpx.AsyncClient", return_value=client), \
         patch("backend.services.airnow.AIRNOW_KEY", ""):
        assert asyncio.run(fetch_airnow_concentrations(34.0, -118.0)) is None

    with patch("backend.services.airnow.httpx.AsyncClient", side_effect=RuntimeError("boom")), \
         patch("backend.services.airnow.AIRNOW_KEY", "test-key"):
        assert asyncio.run(fetch_airnow_concentrations(34.0, -118.0)) is None


# ---------------------------------------------------------------------------
# AQI observation on the /aq/data/ feed (dataType=A).
# ---------------------------------------------------------------------------

def test_observation_uses_latest_hour_max_aqi():
    """The max-AQI row in the latest UTC hour becomes the primary pollutant;
    earlier-hour rows are excluded from the observation snapshot."""
    rows = [
        _obs_row("PM2.5", 80, utc="2026-08-04T20:00:00", site_name="Site A", aqsid="1"),
        _obs_row("PM10", 120, utc="2026-08-04T20:00:00", site_name="Site A", aqsid="1"),
        _obs_row("O3", 150, utc="2026-08-04T20:00:00", site_name="Site B", aqsid="2"),
        _obs_row("CO", 30, utc="2026-08-04T19:00:00", site_name="Site C", aqsid="3"),
    ]
    obs, client = _fetch_observation(rows)

    assert obs["source"] == "AirNow"
    assert obs["aqi"] == 150
    assert obs["primary_pollutant"] == "O3"
    assert obs["category"] == "Unhealthy for Sensitive Groups"
    assert obs["category_color"] == "#ff7e00"
    assert obs["category_text_color"] == "#ffffff"
    assert obs["reporting_area"] == "Site B"
    assert obs["pollutants"] == {"PM2.5": 80, "PM10": 120, "O3": 150}
    assert obs["as_of"] == "2026-08-04T20:00:00+00:00"

    # Request shape: /aq/data/ A-data with the full parameter list and 3h lookback.
    url, params = client.get.await_args.args[0], client.get.await_args.kwargs["params"]
    assert url == "https://www.airnowapi.org/aq/data/"
    assert params["parameters"] == "PM25,PM10,OZONE,CO,NO2,SO2"
    assert params["dataType"] == "A"
    assert params["monitorType"] == 0
    assert params["includerawconcentrations"] == 0
    assert params["format"] == "application/json"
    assert params["verbose"] == 1
    assert params["API_KEY"] == "test-key"
    assert params["BBOX"] == "-118.25,33.75,-117.75,34.25"
    assert "T" in params["startDate"] and ":" not in params["startDate"]


def test_observation_skips_missing_aqi_and_degrades():
    """-999/null AQI rows are skipped; no valid AQI -> None; HTTP/malformed -> None."""
    obs, _ = _fetch_observation([
        _obs_row("PM2.5", -999),
        _obs_row("PM10", None),
    ])
    assert obs is None

    # An earlier hour with valid AQI still yields an observation.
    obs, _ = _fetch_observation([
        _obs_row("PM2.5", 42, utc="2026-08-04T18:00:00", site_name="Site A"),
    ])
    assert obs["aqi"] == 42
    assert obs["primary_pollutant"] == "PM2.5"
    assert obs["reporting_area"] == "Site A"

    assert _fetch_observation([], status=500)[0] is None
    assert _fetch_observation({"not": "a list"})[0] is None
    assert _fetch_observation([])[0] is None
    assert _fetch_observation([_obs_row("PM2.5", 42)], json_raises=ValueError("bad json"))[0] is None

    client = make_client([_obs_row("PM2.5", 42)])
    with patch("backend.services.airnow.httpx.AsyncClient", return_value=client), \
         patch("backend.services.airnow.AIRNOW_KEY", ""):
        assert asyncio.run(fetch_airnow_observation(34.0, -118.0)) is None


# ---------------------------------------------------------------------------
# Distance ceiling: a far AirNow ratio is treated as missing.
# ---------------------------------------------------------------------------

def test_distance_ceiling_rejects_far_sites_accepts_near_sites():
    """A site beyond airnow_ratio_max_distance_km (15 km) yields no ratio; a
    site within the ceiling returns the ratio plus its distance_km."""
    # ~20 km north (0.18 deg lat): beyond the ceiling -> ratio treated as missing.
    far_rows = [
        _row("PM2.5", 10.0, lat=34.18, site_name="Far", aqsid="1"),
        _row("PM10", 25.0, lat=34.18, site_name="Far", aqsid="1"),
    ]
    far_result, _ = _fetch(far_rows)
    assert far_result is None

    # ~8 km north (0.07 deg lat): inside the ceiling -> ratio + distance_km.
    near_rows = [
        _row("PM2.5", 10.0, lat=34.07, site_name="Near", aqsid="1"),
        _row("PM10", 25.0, lat=34.07, site_name="Near", aqsid="1"),
    ]
    near_result, _ = _fetch(near_rows)
    assert near_result is not None
    assert near_result["pm25_pm10_ratio"] == 0.4
    assert near_result["site"]["distance_km"] == pytest.approx(7.78, abs=0.1)


def test_distance_ceiling_is_tunable_via_active_params():
    """use_params can tighten or relax the AirNow ratio distance ceiling."""
    from backend.engine.params import Params, use_params

    rows = [
        _row("PM2.5", 10.0, lat=34.10, site_name="Mid", aqsid="1"),  # ~11 km
        _row("PM10", 25.0, lat=34.10, site_name="Mid", aqsid="1"),
    ]

    with use_params(Params(airnow_ratio_max_distance_km=5.0)):
        client = make_client(rows)
        with patch("backend.services.airnow.httpx.AsyncClient", return_value=client), \
             patch("backend.services.airnow.AIRNOW_KEY", "test-key"):
            assert asyncio.run(fetch_airnow_concentrations(34.0, -118.0)) is None

    with use_params(Params(airnow_ratio_max_distance_km=12.0)):
        client = make_client(rows)
        with patch("backend.services.airnow.httpx.AsyncClient", return_value=client), \
             patch("backend.services.airnow.AIRNOW_KEY", "test-key"):
            result = asyncio.run(fetch_airnow_concentrations(34.0, -118.0))
    assert result is not None
    assert result["pm25_pm10_ratio"] == 0.4
    assert result["site"]["distance_km"] == pytest.approx(11.12, abs=0.1)


# ---------------------------------------------------------------------------
# Signals-level wiring: AirNow fallback only fires when OpenAQ lacks PM10.
# ---------------------------------------------------------------------------

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


def _sensors():
    return {
        101: {"name": "pm25", "units": "µg/m³"},
        102: {"name": "pm10", "units": "µg/m³"},
    }


def test_collect_openaq_signal_uses_airnow_ratio_when_openaq_lacks_pm10():
    pm25 = {"value": 10.0, "unit": "µg/m³", "as_of": "2026-08-04T19:00:00+00:00"}
    airnow = {
        "pm25": 8.0,
        "pm10": 40.0,
        "pm25_pm10_ratio": 0.2,
        "site": {"name": "AirNow Site", "aqsid": "060370001", "distance_km": 1.2,
                 "as_of": "2026-08-04T20:00:00+00:00"},
    }
    with patch("backend.engine.signals.discover_reference_monitors", new_callable=AsyncMock, return_value=[_candidate(111)]), \
         patch("backend.engine.signals.fetch_latest", new_callable=AsyncMock, return_value=_readings(pm25=pm25)), \
         patch("backend.engine.signals.fetch_location_sensors", new_callable=AsyncMock, return_value=_sensors()), \
         patch("backend.engine.signals.fetch_daily_baseline", new_callable=AsyncMock, return_value=None), \
         patch("backend.engine.signals.fetch_same_hour_baseline", new_callable=AsyncMock, return_value=None), \
         patch("backend.engine.signals.fetch_airnow_concentrations", new_callable=AsyncMock, return_value=airnow) as airnow_mock:
        signal = asyncio.run(collect_openaq_signal(34.0, -118.0))

    assert signal["status"] == "present"
    assert signal["pm25"] == 10.0  # OpenAQ concentration unchanged
    assert signal["pm10"] is None
    assert signal["pm25_pm10_ratio"] == 0.2
    assert signal["ratio_source"] == "airnow"
    assert signal["ratio_monitor_distance_km"] == 1.2
    assert "PM2.5/PM10 ratio from AirNow monitor AirNow Site 1.2 km away" in signal["details"]
    airnow_mock.assert_awaited_once_with(34.0, -118.0)


def test_collect_openaq_signal_marks_openaq_ratio_when_co_located():
    pm25 = {"value": 10.0, "unit": "µg/m³", "as_of": "2026-08-04T19:00:00+00:00"}
    pm10 = {"value": 25.0, "unit": "µg/m³", "as_of": "2026-08-04T19:00:00+00:00"}
    with patch("backend.engine.signals.discover_reference_monitors", new_callable=AsyncMock, return_value=[_candidate(111)]), \
         patch("backend.engine.signals.fetch_latest", new_callable=AsyncMock, return_value=_readings(pm25=pm25, pm10=pm10)), \
         patch("backend.engine.signals.fetch_location_sensors", new_callable=AsyncMock, return_value=_sensors()), \
         patch("backend.engine.signals.fetch_daily_baseline", new_callable=AsyncMock, return_value=None), \
         patch("backend.engine.signals.fetch_same_hour_baseline", new_callable=AsyncMock, return_value=None), \
         patch("backend.engine.signals.fetch_airnow_concentrations", new_callable=AsyncMock) as airnow_mock:
        signal = asyncio.run(collect_openaq_signal(34.0, -118.0))

    assert signal["pm25_pm10_ratio"] == 0.4
    assert signal["ratio_source"] == "openaq"
    assert signal["ratio_monitor_distance_km"] is None
    assert "AirNow" not in signal["details"]
    airnow_mock.assert_not_awaited()
