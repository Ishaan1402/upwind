import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from backend.main import app
from backend.db import init_db, get_cached_geocode, set_cached_geocode
from backend.services.geocode import geocode_location

FAKE_LOCATION = {
    "lat": 34.09,
    "lon": -118.41,
    "name": "Beverly Hills, CA 90210",
    "zip_code": "90210",
    "state": "CA",
    "city": "Beverly Hills",
}

FAKE_OBSERVATION = {
    "source": "AirNow",
    "aqi": 42,
    "primary_pollutant": "PM2.5",
    "category": "Good",
    "category_color": "#00e400",
    "category_text_color": "#000000",
    "category_description": "Air quality is satisfactory.",
    "reporting_area": "Los Angeles",
    "pollutants": {"PM2.5": 10.0},
    "as_of": "2026-07-27T12:00:00Z",
}


@pytest.fixture(autouse=True)
def setup_test_db():
    init_db()


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rate_limiter_aqi_burst(client):
    """AQI limiter trips without depending on live AirNow/Open-Meteo."""
    headers = {"X-Forwarded-For": "203.0.113.10"}

    with (
        patch("backend.middleware.rate_limit.TRUST_PROXY", True),
        patch("backend.middleware.rate_limit.RATE_LIMIT_AQI_PER_MIN", 3),
        patch("backend.routers.aqi.geocode_location", return_value=FAKE_LOCATION),
        patch("backend.routers.aqi.fetch_airnow_observation", new_callable=AsyncMock, return_value=FAKE_OBSERVATION),
        patch("backend.routers.aqi.fetch_openmeteo_aqi", new_callable=AsyncMock, return_value=None),
    ):
        statuses = [client.get("/api/aqi?zip=90210", headers=headers).status_code for _ in range(5)]

    assert statuses[:3] == [200, 200, 200]
    assert statuses[3:] == [429, 429]


def test_rate_limiter_why_stream_burst(client):
    """Why limiter uses X-Forwarded-For when TRUST_PROXY is enabled on the middleware module."""
    headers = {"X-Forwarded-For": "203.0.113.20"}

    with (
        patch("backend.middleware.rate_limit.TRUST_PROXY", True),
        patch("backend.middleware.rate_limit.RATE_LIMIT_WHY_PER_HOUR", 2),
        patch("backend.engine.signals.fetch_openmeteo_weather", new_callable=AsyncMock, return_value={
            "wind_speed_mph": 5.0,
            "wind_direction_deg": 180.0,
            "temperature_f": 70.0,
            "boundary_layer_height_m": 800,
        }),
        patch("backend.engine.signals.fetch_aod_signal", new_callable=AsyncMock, return_value={
            "status": "absent",
            "details": "none",
        }),
        patch("backend.engine.signals.fetch_firms_hotspots", new_callable=AsyncMock, return_value={
            "status": "absent",
            "count": 0,
            "total_count": 0,
            "hotspots": [],
            "details": "none",
        }),
        patch("backend.engine.signals.fetch_hms_smoke", new_callable=AsyncMock, return_value={
            "status": "absent",
            "density": None,
            "details": "no plume",
        }),
        patch("backend.engine.signals.fetch_wfigs_incident", new_callable=AsyncMock, return_value={
            "status": "absent",
            "incident": None,
            "count": 0,
            "alignment": None,
            "details": "none",
        }),
        patch("backend.engine.signals.collect_openaq_signal", new_callable=AsyncMock, return_value={
            "id": "openaq_concentrations",
            "label": "Local Monitor Concentrations (OpenAQ)",
            "status": "unavailable",
            "details": "none",
        }),
        patch("backend.engine.signals.fetch_place_context", new_callable=AsyncMock, return_value={
            "status": "unavailable",
            "population": None,
            "rural": None,
            "details": "none",
        }),
        patch("backend.engine.signals.search_fire_incident_name", new_callable=AsyncMock, return_value=None),
        patch("backend.routers.why.get_cached_narrative", return_value="Cached brief."),
    ):
        statuses = [
            client.get("/api/why/stream?lat=34.05&lon=-118.25&aqi=42", headers=headers).status_code
            for _ in range(4)
        ]

    assert statuses[:2] == [200, 200]
    assert statuses[2:] == [429, 429]


def test_geocode_caching():
    query = "test_city_cache_unique_123"
    set_cached_geocode(query, {"lat": 10.0, "lon": 20.0, "name": "Cached Test City", "zip_code": None})

    cached = get_cached_geocode(query)
    assert cached is not None
    assert cached["name"] == "Cached Test City"

    with patch("backend.services.geocode._nom_geocodes.geocode") as mock_nom:
        result = geocode_location("test_city_cache_unique_123")
        assert result is not None
        assert result["name"] == "Cached Test City"
        mock_nom.assert_not_called()
