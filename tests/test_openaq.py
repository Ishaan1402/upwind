"""Unit tests for the OpenAQ reference-monitor service."""

import asyncio
from datetime import datetime, timedelta, timezone as dt_timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.services.openaq import (
    discover_reference_monitor,
    fetch_daily_baseline,
    fetch_latest,
    fetch_same_hour_baseline,
    normalize_reading,
)


@pytest.fixture(autouse=True)
def clear_openaq_cache():
    """Isolate the module-level TTL cache between tests."""
    from backend.services import openaq

    openaq._CACHE_BUCKETS.clear()
    yield
    openaq._CACHE_BUCKETS.clear()


def make_response(status=200, payload=None):
    resp = Mock()
    resp.status_code = status
    resp.json.return_value = payload if payload is not None else {}
    return resp


def make_client(responses):
    """httpx.AsyncClient mock routing by URL substring."""
    client = Mock()

    async def fake_get(url, params=None, **kwargs):
        for key, resp in responses.items():
            if key in str(url):
                return resp
        return make_response(404)

    client.get = AsyncMock(side_effect=fake_get)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def test_normalize_reading_units():
    assert normalize_reading("pm25", 12.5, "µg/m³") == (12.5, "µg/m³")
    assert normalize_reading("pm10", 40, "ug/m3") == (40.0, "µg/m³")
    assert normalize_reading("o3", 0.04, "ppm") == (40.0, "ppb")
    assert normalize_reading("no2", 60, "ppb") == (60.0, "ppb")
    assert normalize_reading("so2", 0.1, "ppm") == (100.0, "ppb")
    assert normalize_reading("co", 1.5, "ppm") == (1.5, "ppm")

    # Unrecognized / non-canonical units are skipped, not guessed.
    assert normalize_reading("o3", 100, "µg/m³") is None
    assert normalize_reading("co", 2, "mg/m³") is None
    assert normalize_reading("pm25", None, "µg/m³") is None
    assert normalize_reading("pm25", "nope", "µg/m³") is None
    assert normalize_reading("bc", 1, "µg/m³") is None


def test_discover_reference_monitor_picks_nearest_and_filters():
    client = make_client({
        "v3/locations": make_response(200, {
            "results": [
                {
                    "id": 111,
                    "name": "Far Monitor",
                    "coordinates": {"latitude": 35.5, "longitude": -118.0},
                    "timezone": "America/Los_Angeles",
                    "provider": {"name": "US EPA"},
                    "owner": {"name": "State Agency"},
                },
                {
                    "id": 222,
                    "name": "Near Monitor",
                    "coordinates": {"latitude": 34.05, "longitude": -118.0},
                    "timezone": "America/Los_Angeles",
                    "provider": {"name": "US EPA"},
                    "owner": {"name": "County Agency"},
                },
            ]
        }),
    })
    with patch("backend.services.openaq.httpx.AsyncClient", return_value=client), \
         patch("backend.services.openaq.OPENAQ_API_KEY", "test-key"):
        result = asyncio.run(discover_reference_monitor(34.0, -118.0))

    assert result["location_id"] == 222
    assert result["distance_km"] < 10
    assert result["provider"] == "US EPA"
    params = client.get.await_args_list[0].kwargs["params"]
    assert params["monitor"] == "true"
    assert params["mobile"] == "false"
    assert params["iso"] == "US"
    assert params["radius"] == "25000"


def test_discover_reference_monitor_degrades():
    with patch("backend.services.openaq.OPENAQ_API_KEY", ""):
        assert asyncio.run(discover_reference_monitor(34.0, -118.0)) is None

    client = make_client({"v3/locations": make_response(500)})
    with patch("backend.services.openaq.httpx.AsyncClient", return_value=client), \
         patch("backend.services.openaq.OPENAQ_API_KEY", "test-key"):
        assert asyncio.run(discover_reference_monitor(34.0, -118.0)) is None

    client = make_client({"v3/locations": make_response(200, {"results": []})})
    with patch("backend.services.openaq.httpx.AsyncClient", return_value=client), \
         patch("backend.services.openaq.OPENAQ_API_KEY", "test-key"):
        assert asyncio.run(discover_reference_monitor(34.0, -118.0)) is None


def test_fetch_latest_maps_normalizes_and_filters_stale():
    now = datetime.now(dt_timezone.utc)
    fresh = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    stale = (now - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    client = make_client({
        "v3/locations/42/latest": make_response(200, {
            "results": [
                {"sensorsId": 101, "value": 12.5, "datetime": {"utc": fresh}},
                {"sensorsId": 102, "value": 0.04, "datetime": {"utc": fresh}},
                {"sensorsId": 103, "value": 30.0, "datetime": {"utc": fresh}},  # µg/m³ gas -> skipped
                {"sensorsId": 101, "value": 99.0, "datetime": {"utc": stale}},  # stale -> dropped
            ]
        }),
    })
    sensors_map = {
        101: {"name": "pm25", "units": "µg/m³"},
        102: {"name": "o3", "units": "ppm"},
        103: {"name": "no2", "units": "µg/m³"},
    }
    with patch("backend.services.openaq.httpx.AsyncClient", return_value=client), \
         patch("backend.services.openaq.OPENAQ_API_KEY", "test-key"), \
         patch("backend.services.openaq.fetch_location_sensors", new_callable=AsyncMock, return_value=sensors_map):
        readings = asyncio.run(fetch_latest(42))

    assert set(readings.keys()) == {"pm25", "o3"}
    assert readings["pm25"]["value"] == 12.5
    assert readings["pm25"]["unit"] == "µg/m³"
    assert readings["o3"]["value"] == 40.0  # ppm -> ppb
    assert "no2" not in readings


def test_fetch_latest_caches_and_degrades():
    now = datetime.now(dt_timezone.utc)
    fresh = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    client = make_client({
        "v3/locations/42/latest": make_response(200, {
            "results": [{"sensorsId": 101, "value": 8.0, "datetime": {"utc": fresh}}]
        }),
    })
    sensors_map = {101: {"name": "pm25", "units": "µg/m³"}}
    with patch("backend.services.openaq.httpx.AsyncClient", return_value=client), \
         patch("backend.services.openaq.OPENAQ_API_KEY", "test-key"), \
         patch("backend.services.openaq.fetch_location_sensors", new_callable=AsyncMock, return_value=sensors_map):
        first = asyncio.run(fetch_latest(42))
        second = asyncio.run(fetch_latest(42))

    assert first == second
    assert client.get.await_count == 1  # second call served from cache

    with patch("backend.services.openaq.OPENAQ_API_KEY", ""):
        assert asyncio.run(fetch_latest(999)) == {}  # fresh location, no cache


def test_fetch_daily_baseline_filters_incomplete_and_computes_percentile():
    today = datetime.now(dt_timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    records = []
    for i in range(5):
        dt = (today - timedelta(days=4 - i)).strftime("%Y-%m-%dT00:00:00Z")
        records.append({
            "value": 10.0 + 2.0 * i,
            "period": {"datetimeTo": {"utc": dt}},
            "coverage": {"percentComplete": 100.0},
        })
    # Latest day is today; incomplete record is excluded from the distribution.
    records.append({
        "value": 20.0,
        "period": {"datetimeTo": today.strftime("%Y-%m-%dT00:00:00Z")},
        "coverage": {"percentComplete": 50.0},
    })
    client = make_client({"v3/sensors/7/days": make_response(200, {"results": records})})
    with patch("backend.services.openaq.httpx.AsyncClient", return_value=client), \
         patch("backend.services.openaq.OPENAQ_API_KEY", "test-key"):
        baseline = asyncio.run(fetch_daily_baseline(7))

    assert baseline["count"] == 5
    assert baseline["today_value"] == 18.0
    assert baseline["percentile"] == 100.0

    # Cached: no second HTTP request.
    with patch("backend.services.openaq.httpx.AsyncClient", return_value=client), \
         patch("backend.services.openaq.OPENAQ_API_KEY", "test-key"):
        asyncio.run(fetch_daily_baseline(7))
    assert client.get.await_count == 1


def test_fetch_daily_baseline_degrades():
    client = make_client({"v3/sensors/7/days": make_response(500)})
    with patch("backend.services.openaq.httpx.AsyncClient", return_value=client), \
         patch("backend.services.openaq.OPENAQ_API_KEY", "test-key"):
        assert asyncio.run(fetch_daily_baseline(7)) is None

    with patch("backend.services.openaq.OPENAQ_API_KEY", ""):
        assert asyncio.run(fetch_daily_baseline(7)) is None


def test_fetch_same_hour_baseline_math_and_min_samples():
    now = datetime.now(dt_timezone.utc).replace(minute=0, second=0, microsecond=0)
    records = [
        {"value": float(10 + 2 * i), "period": {"datetimeTo": {"local": (now - timedelta(days=i)).isoformat()}}}
        for i in range(10)
    ]
    with patch("backend.services.openaq._fetch_aggregate_series", new_callable=AsyncMock, return_value=records):
        baseline = asyncio.run(fetch_same_hour_baseline(7, "UTC", 28.0))

    assert baseline["count"] == 10
    assert baseline["percentile"] == 100.0
    assert baseline["median"] == 19.0

    with patch("backend.services.openaq._fetch_aggregate_series", new_callable=AsyncMock, return_value=records[:3]):
        assert asyncio.run(fetch_same_hour_baseline(7, "UTC", 28.0)) is None

    with patch("backend.services.openaq._fetch_aggregate_series", new_callable=AsyncMock, return_value=[]):
        assert asyncio.run(fetch_same_hour_baseline(7, "UTC", 28.0)) is None

    assert asyncio.run(fetch_same_hour_baseline(7, "UTC", None)) is None
