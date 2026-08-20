"""Unit tests for the Open-Meteo weather service."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.services.openmeteo import fetch_openmeteo_weather


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


def run_fetch(client):
    with patch("backend.services.openmeteo.httpx.AsyncClient", return_value=client):
        return asyncio.run(fetch_openmeteo_weather(45.0, -117.0))


@pytest.mark.honesty
def test_fetch_openmeteo_weather_parses_gust_and_precip():
    """wind_gusts_10m lands in wind_gust_mph and the daily precipitation_sum
    series (past 30 days including today, in mm) sums into precip_30d_in (in)."""
    # 30 days of 12.7 mm plus today = 31 entries; the most recent 30 are
    # 30 * 12.7 mm = 381 mm = 15.0 in.
    precip = [12.7] * 31
    resp = make_response(payload={
        "current": {
            "temperature_2m": 78.0,
            "wind_speed_10m": 18.5,
            "wind_direction_10m": 260.0,
            "wind_gusts_10m": 45.0,
            "boundary_layer_height": 1200,
        },
        "daily": {"precipitation_sum": precip},
    })
    client = make_client({"/v1/forecast": resp})

    result = run_fetch(client)

    assert result["wind_speed_mph"] == 18.5
    assert result["wind_direction_deg"] == 260.0
    assert result["wind_gust_mph"] == 45.0
    assert result["boundary_layer_height_m"] == 1200
    assert result["temperature_f"] == 78.0
    assert result["precip_30d_in"] == 15.0


@pytest.mark.honesty
def test_fetch_openmeteo_weather_rounds_precip_to_two_decimals():
    """A partial/odd series is summed in mm and converted to inches at 2dp."""
    # 0.635 mm/day -> 0.025 in/day; 30 days = 0.75 in.
    resp = make_response(payload={
        "current": {"wind_gusts_10m": 41.0},
        "daily": {"precipitation_sum": [0.635] * 30},
    })
    client = make_client({"/v1/forecast": resp})

    result = run_fetch(client)

    assert result["wind_gust_mph"] == 41.0
    assert result["precip_30d_in"] == 0.75


def test_fetch_openmeteo_weather_missing_daily_returns_none_precip():
    """An absent/empty daily series yields precip_30d_in=None, never a crash."""
    resp = make_response(payload={
        "current": {"wind_gusts_10m": 30.0},
        "daily": {},
    })
    client = make_client({"/v1/forecast": resp})

    result = run_fetch(client)

    assert result["wind_gust_mph"] == 30.0
    assert result["precip_30d_in"] is None


def test_fetch_openmeteo_weather_requests_gust_and_30d_precip():
    """The request asks for wind_gusts_10m and the 30-day precipitation history
    on the SAME call (daily=precipitation_sum, past_days=30)."""
    requested = {}

    async def fake_get(url, params=None, **kwargs):
        requested.update(params or {})
        return make_response(payload={"current": {}, "daily": {}})

    client = Mock()
    client.get = AsyncMock(side_effect=fake_get)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    run_fetch(client)

    assert "wind_gusts_10m" in requested["current"]
    assert requested["daily"] == "precipitation_sum"
    assert requested["past_days"] == 30
    assert requested["wind_speed_unit"] == "mph"


def test_fetch_openmeteo_weather_http_error_returns_none():
    resp = make_response(status=500)
    client = make_client({"/v1/forecast": resp})

    assert run_fetch(client) is None
