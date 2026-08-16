"""Endpoint tests verifying OpenAQ signal wiring in both /api/why paths."""

import asyncio
import json
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from tests.fixtures import SMOKE_LOCATION

PRESENT_SIGNAL = {
    "id": "openaq_concentrations",
    "label": "Local Monitor Concentrations (OpenAQ)",
    "status": "present",
    "pm25": 18.0,
    "pm10": 25.0,
    "o3_ppb": 40.0,
    "no2_ppb": 12.0,
    "co_ppm": 0.5,
    "so2_ppb": 3.0,
    "pm25_pm10_ratio": 0.72,
    "monitor": {"name": "Downtown", "distance_km": 2.3, "provider": "US EPA", "owner": "State Agency"},
    "as_of": "2026-08-04T12:00:00+00:00",
    "daily_percentile": 95.0,
    "same_hour_percentile": 90.0,
    "same_hour_median": 8.0,
    "details": "Nearest EPA reference monitor Downtown (2.3 km away)",
}

WEATHER = {
    "wind_speed_mph": 5.0,
    "wind_direction_deg": 180.0,
    "temperature_f": 70.0,
    "boundary_layer_height_m": 900.0,
}

AOD_RESULT = {"status": "absent", "density": None, "aod_value": 0.1, "details": "clear"}
FIRMS_RESULT = {
    "status": "absent",
    "count": 0,
    "total_count": 0,
    "nearest": None,
    "hotspots": [],
    "alignment": None,
    "details": "none",
}
HMS_RESULT = {"status": "absent", "density": None, "details": "no plume"}
WFIGS_RESULT = {
    "status": "absent",
    "incident": None,
    "count": 0,
    "alignment": None,
    "details": "no federal incidents nearby",
}
PLACE_RESULT = {
    "status": "unavailable",
    "population": None,
    "rural": None,
    "details": "no census key in test",
}


def test_why_includes_openaq_signal_and_trace_step():
    from backend.main import app

    with patch("backend.engine.signals.fetch_openmeteo_weather", new_callable=AsyncMock, return_value=WEATHER), \
         patch("backend.engine.signals.fetch_aod_signal", new_callable=AsyncMock, return_value=AOD_RESULT), \
         patch("backend.engine.signals.fetch_hms_smoke", new_callable=AsyncMock, return_value=HMS_RESULT), \
         patch("backend.engine.signals.fetch_wfigs_incident", new_callable=AsyncMock, return_value=WFIGS_RESULT), \
         patch("backend.engine.signals.fetch_firms_hotspots", new_callable=AsyncMock, return_value=FIRMS_RESULT), \
         patch("backend.engine.signals.search_fire_incident_name", new_callable=AsyncMock, return_value=None), \
         patch("backend.engine.signals.collect_openaq_signal", new_callable=AsyncMock, return_value=PRESENT_SIGNAL), \
         patch("backend.engine.signals.fetch_place_context", new_callable=AsyncMock, return_value=PLACE_RESULT), \
         patch("backend.routers.why.generate_narrative_briefing", new_callable=AsyncMock, return_value="test narrative"), \
         patch("backend.routers.why.judge_narrative", new_callable=AsyncMock, return_value={"verdict": "pass"}), \
         patch("backend.routers.why.get_cached_narrative", return_value=None), \
         patch("backend.routers.why.set_cached_narrative", return_value=None):
        with TestClient(app) as client:
            resp = client.post(
                "/api/why",
                json={
                    "location": SMOKE_LOCATION,
                    "observation": {"aqi": 85, "primary_pollutant": "PM2.5", "category": "Moderate"},
                },
            )

    assert resp.status_code == 200
    data = resp.json()
    assert any(s["id"] == "openaq_concentrations" and s["status"] == "present" for s in data["signals"])
    assert any(t["step"] == "openaq_monitors" for t in data["execution_trace"])


def test_why_cache_key_includes_aqi_and_pollutant():
    """Same location + hour must not share a narrative across different AQI levels."""
    from backend.main import app

    captured = {}
    def fake_set(key, narrative, payload, verdict=None):
        captured["key"] = key

    with patch("backend.engine.signals.fetch_openmeteo_weather", new_callable=AsyncMock, return_value=WEATHER), \
         patch("backend.engine.signals.fetch_aod_signal", new_callable=AsyncMock, return_value=AOD_RESULT), \
         patch("backend.engine.signals.fetch_hms_smoke", new_callable=AsyncMock, return_value=HMS_RESULT), \
         patch("backend.engine.signals.fetch_wfigs_incident", new_callable=AsyncMock, return_value=WFIGS_RESULT), \
         patch("backend.engine.signals.fetch_firms_hotspots", new_callable=AsyncMock, return_value=FIRMS_RESULT), \
         patch("backend.engine.signals.search_fire_incident_name", new_callable=AsyncMock, return_value=None), \
         patch("backend.engine.signals.collect_openaq_signal", new_callable=AsyncMock, return_value=PRESENT_SIGNAL), \
         patch("backend.engine.signals.fetch_place_context", new_callable=AsyncMock, return_value=PLACE_RESULT), \
         patch("backend.routers.why.generate_narrative_briefing", new_callable=AsyncMock, return_value="test narrative"), \
         patch("backend.routers.why.judge_narrative", new_callable=AsyncMock, return_value={"verdict": "pass"}), \
         patch("backend.routers.why.get_cached_narrative", return_value=None), \
         patch("backend.routers.why.set_cached_narrative", side_effect=fake_set):
        with TestClient(app) as client:
            resp = client.post(
                "/api/why",
                json={
                    "location": SMOKE_LOCATION,
                    "observation": {"aqi": 180, "primary_pollutant": "PM2.5", "category": "Unhealthy"},
                },
            )

    assert resp.status_code == 200
    key = captured.get("key", "")
    assert "_180_" in key
    assert "PM2.5" in key


def test_skipped_web_search_still_resolves():
    """On clean-air days the gated web search is skipped but must still emit a
    tool_done so its card resolves instead of spinning forever."""
    from backend.engine.signals import iter_evidence_signals

    async def collect():
        events = []
        async for kind, payload in iter_evidence_signals(
            {"lat": 34.05, "lon": -118.24, "zip_code": "90012", "state": "CA", "city": "Los Angeles"},
            {"aqi": 42, "primary_pollutant": "O3", "category": "Good"},
        ):
            events.append((kind, payload))
        return events

    with patch("backend.engine.signals.fetch_openmeteo_weather", new_callable=AsyncMock, return_value=WEATHER), \
         patch("backend.engine.signals.fetch_aod_signal", new_callable=AsyncMock, return_value=AOD_RESULT), \
         patch("backend.engine.signals.fetch_hms_smoke", new_callable=AsyncMock, return_value=HMS_RESULT), \
         patch("backend.engine.signals.fetch_wfigs_incident", new_callable=AsyncMock, return_value=WFIGS_RESULT), \
         patch("backend.engine.signals.fetch_firms_hotspots", new_callable=AsyncMock, return_value=FIRMS_RESULT), \
         patch("backend.engine.signals.collect_openaq_signal", new_callable=AsyncMock, return_value=PRESENT_SIGNAL), \
         patch("backend.engine.signals.fetch_place_context", new_callable=AsyncMock, return_value=PLACE_RESULT), \
         patch("backend.engine.signals.search_fire_incident_name", new_callable=AsyncMock, return_value=None):
        events = asyncio.run(collect())

    web_done = [p for k, p in events if k == "tool_done" and p["step"] == "web_search"]
    assert web_done and web_done[0]["status"] == "absent"


def test_stream_emits_openaq_events_and_signal():
    from backend.main import app

    async def fake_stream(*args, **kwargs):
        yield "test narrative"

    with patch("backend.engine.signals.fetch_openmeteo_weather", new_callable=AsyncMock, return_value=WEATHER), \
         patch("backend.engine.signals.fetch_aod_signal", new_callable=AsyncMock, return_value=AOD_RESULT), \
         patch("backend.engine.signals.fetch_hms_smoke", new_callable=AsyncMock, return_value=HMS_RESULT), \
         patch("backend.engine.signals.fetch_wfigs_incident", new_callable=AsyncMock, return_value=WFIGS_RESULT), \
         patch("backend.engine.signals.fetch_firms_hotspots", new_callable=AsyncMock, return_value=FIRMS_RESULT), \
         patch("backend.engine.signals.search_fire_incident_name", new_callable=AsyncMock, return_value=None), \
         patch("backend.engine.signals.collect_openaq_signal", new_callable=AsyncMock, return_value=PRESENT_SIGNAL), \
         patch("backend.engine.signals.fetch_place_context", new_callable=AsyncMock, return_value=PLACE_RESULT), \
         patch("backend.routers.why.generate_narrative_briefing_stream", side_effect=fake_stream), \
         patch("backend.routers.why.get_cached_narrative", return_value=None), \
         patch("backend.routers.why.set_cached_narrative", return_value=None), \
         patch("backend.routers.why.judge_narrative", new_callable=AsyncMock, return_value={"verdict": "pass"}):
        with TestClient(app) as client:
            with client.stream(
                "GET",
                "/api/why/stream?lat=34.05&lon=-118.24&zip_code=90012&city=Los+Angeles&state=CA"
                "&aqi=85&primary_pollutant=PM2.5&category=Moderate",
            ) as resp:
                assert resp.status_code == 200
                events = {}
                current_event = None
                for line in resp.iter_lines():
                    if line.startswith("event:"):
                        current_event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:") and current_event:
                        payload = json.loads(line.split(":", 1)[1].strip())
                        events.setdefault(current_event, []).append(payload)

    starts = [p["step"] for p in events.get("tool_start", [])]
    dones = [p["step"] for p in events.get("tool_done", [])]
    assert "openaq_monitors" in starts
    assert "openaq_monitors" in dones

    signals_ready = events.get("signals_ready", [])
    assert signals_ready
    assert any(s["id"] == "openaq_concentrations" for s in signals_ready[0]["signals"])


def test_stream_and_post_share_identical_signals():
    """Both /api/why paths must produce the same evidence, including the
    formerly-divergent unverified-news incident fields."""
    from backend.main import app

    async def fake_stream(*args, **kwargs):
        yield "test narrative"

    engine_patches = [
        patch("backend.engine.signals.fetch_openmeteo_weather", new_callable=AsyncMock, return_value=WEATHER),
        patch("backend.engine.signals.fetch_aod_signal", new_callable=AsyncMock, return_value=AOD_RESULT),
        patch("backend.engine.signals.fetch_hms_smoke", new_callable=AsyncMock, return_value=HMS_RESULT),
        patch("backend.engine.signals.fetch_wfigs_incident", new_callable=AsyncMock, return_value=WFIGS_RESULT),
        patch("backend.engine.signals.fetch_firms_hotspots", new_callable=AsyncMock, return_value=FIRMS_RESULT),
        patch("backend.engine.signals.search_fire_incident_name", new_callable=AsyncMock, return_value="Test Fire"),
        patch("backend.engine.signals.collect_openaq_signal", new_callable=AsyncMock, return_value=PRESENT_SIGNAL),
        patch("backend.engine.signals.fetch_place_context", new_callable=AsyncMock, return_value=PLACE_RESULT),
    ]
    router_patches = [
        patch("backend.routers.why.generate_narrative_briefing", new_callable=AsyncMock, return_value="test"),
        patch("backend.routers.why.generate_narrative_briefing_stream", side_effect=fake_stream),
        patch("backend.routers.why.judge_narrative", new_callable=AsyncMock, return_value={"verdict": "pass"}),
        patch("backend.routers.why.update_cached_verdict", return_value=None),
        patch("backend.routers.why.get_cached_narrative", return_value=None),
        patch("backend.routers.why.set_cached_narrative", return_value=None),
    ]

    with ExitStack() as stack:
        for p in engine_patches + router_patches:
            stack.enter_context(p)
        with TestClient(app) as client:
            post_resp = client.post(
                "/api/why",
                json={
                    "location": SMOKE_LOCATION,
                    "observation": {"aqi": 85, "primary_pollutant": "PM2.5", "category": "Moderate"},
                },
            )
            with client.stream(
                "GET",
                "/api/why/stream?lat=45.3199&lon=-117.8147&zip_code=97824&city=Cove&state=OR&name=Cove"
                "&aqi=85&primary_pollutant=PM2.5&category=Moderate",
            ) as stream_resp:
                events = {}
                current_event = None
                for line in stream_resp.iter_lines():
                    if line.startswith("event:"):
                        current_event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:") and current_event:
                        events.setdefault(current_event, []).append(json.loads(line.split(":", 1)[1].strip()))

    assert post_resp.status_code == 200
    post_signals = post_resp.json()["signals"]
    stream_signals = events["signals_ready"][0]["signals"]
    assert post_signals == stream_signals

    firms = next(s for s in stream_signals if s["id"] == "firms_upwind")
    assert firms["incident_name"] == "Test Fire"
    assert firms["unverified_news_incident"] == "Test Fire"
