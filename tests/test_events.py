"""Tests for the /api/events user-behavior endpoint and its rate limiting."""

from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

from backend.main import app


def test_events_endpoint_records_event():
    with patch("backend.routers.events.record_user_event") as mock:
        client = TestClient(app)
        r = client.post("/api/events", json={"event": "why_open", "detail": "90210"})
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        mock.assert_called_once_with("why_open", "90210")


def test_events_endpoint_accepts_missing_detail():
    with patch("backend.routers.events.record_user_event") as mock:
        client = TestClient(app)
        r = client.post("/api/events", json={"event": "aqi_view"})
        assert r.status_code == 200
        mock.assert_called_once_with("aqi_view", None)


def test_events_endpoint_rejects_bad_names():
    client = TestClient(app)
    for bad in ["", "has space", "UPPER", "a" * 65]:
        r = client.post("/api/events", json={"event": bad})
        assert r.status_code == 400


def test_events_endpoint_is_rate_limited():
    """Track Part 2 #7: /api/events must be throttled like the other routes."""
    headers = {"X-Forwarded-For": "203.0.113.50"}
    with (
        patch("backend.middleware.rate_limit.TRUST_PROXY", True),
        patch("backend.middleware.rate_limit.RATE_LIMIT_EVENTS_PER_MIN", 2),
        patch("backend.routers.events.record_user_event"),
    ):
        client = TestClient(app)
        statuses = [client.post("/api/events", json={"event": "why_open"}, headers=headers).status_code for _ in range(4)]
    assert statuses[:2] == [200, 200]
    assert statuses[2:] == [429, 429]


def test_rate_limited_requests_are_recorded():
    """Track Part 2 #6: 429 responses must still reach the telemetry tables."""
    headers = {"X-Forwarded-For": "203.0.113.60"}
    recorded = Mock()
    with (
        patch("backend.middleware.rate_limit.TRUST_PROXY", True),
        patch("backend.middleware.rate_limit.RATE_LIMIT_AQI_PER_MIN", 1),
        patch("backend.middleware.rate_limit.record_request", recorded),
        patch("backend.routers.aqi.geocode_location", return_value={"lat": 34.09, "lon": -118.41, "name": "BH", "zip_code": "90210"}),
        patch("backend.routers.aqi.fetch_airnow_observation", new=AsyncMock(return_value=None)),
        patch("backend.routers.aqi.fetch_openmeteo_aqi", new=AsyncMock(return_value=None)),
    ):
        client = TestClient(app)
        client.get("/api/aqi?zip=90210", headers=headers)
        client.get("/api/aqi?zip=90210", headers=headers)
    # The second (rate-limited) request is recorded with a 429 status.
    assert any(call.args[2] == 429 for call in recorded.call_args_list)
