import asyncio
import pytest
from backend.services.geocode import geocode_location
from backend.engine.score import score_hypotheses
from backend.services.incident_search import search_fire_incident_name

def test_geocode_zip():
    res = geocode_location("90210")
    assert res is not None
    assert res["zip_code"] == "90210"
    assert round(res["lat"], 1) == 34.1

def test_score_smoke_promoted():
    observation = {
        "aqi": 120,
        "primary_pollutant": "PM2.5",
        "category": "Unhealthy for Sensitive Groups"
    }
    signals = [
        {"id": "aerosol_plume", "status": "present", "density": "medium", "aod_value": 0.65},
        {"id": "firms_upwind", "status": "present", "count": 3, "nearest": {"distance_miles": 45.0, "bearing": "NW", "distance_km": 72.4}},
        {"id": "wind", "status": "present", "speed_mph": 12.0, "direction_deg": 315.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False}
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    top_h = hypotheses[0]
    assert top_h["id"] == "wildfire_smoke"
    assert top_h["confidence"] in ["medium", "high"]

def test_score_aloft_smoke_contradiction():
    observation = {
        "aqi": 35,
        "primary_pollutant": "PM2.5",
        "category": "Good"
    }
    signals = [
        {"id": "aerosol_plume", "status": "present", "density": "medium", "aod_value": 0.6},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 5.0, "direction_deg": 180.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": False},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False}
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["confidence"] == "low"
    assert any("aloft" in q.lower() for q in questions)

def test_score_ozone_hot_day():
    observation = {
        "aqi": 115,
        "primary_pollutant": "O3",
        "category": "Unhealthy for Sensitive Groups"
    }
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.08},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 4.0, "direction_deg": 90.0},
        {"id": "surface_pm_level", "status": "absent", "primary": False, "pm10_primary": False, "pm25_primary": False, "elevated": False},
        {"id": "ozone_heat", "status": "present", "primary": True, "hot_day": True, "temperature_f": 92.0}
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    top_h = hypotheses[0]
    assert top_h["id"] == "ozone_episode"
    assert top_h["confidence"] in ["medium", "high"]

def test_score_unexplained_pm():
    observation = {
        "aqi": 85,
        "primary_pollutant": "PM2.5",
        "category": "Moderate"
    }
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.09},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 45.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0}
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    top_h = hypotheses[0]
    assert top_h["id"] == "urban_industrial_pm"

def test_score_windblown_dust():
    observation = {
        "aqi": 140,
        "primary_pollutant": "PM10",
        "category": "Unhealthy for Sensitive Groups"
    }
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.12},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 20.0, "direction_deg": 270.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": True, "pm25_primary": False, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 75.0}
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    top_h = hypotheses[0]
    assert top_h["id"] == "windblown_dust"
    assert top_h["confidence"] == "high"

def test_score_winter_stagnation():
    observation = {
        "aqi": 110,
        "primary_pollutant": "PM2.5",
        "category": "Unhealthy for Sensitive Groups"
    }
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.10},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 2.0, "direction_deg": 0.0, "boundary_layer_height_m": 350.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 30.0}
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    top_h = hypotheses[0]
    assert top_h["id"] == "winter_stagnation"
    assert top_h["confidence"] == "high"

def test_incident_web_search():
    name = asyncio.run(search_fire_incident_name("CA", "Chico", 39.72, -121.83))
    assert name is None or isinstance(name, str)
