import asyncio
import pytest
from backend.services.geocode import geocode_location
from backend.engine.score import score_hypotheses
from backend.services.incident_search import search_fire_incident_name
from tests.fixtures import DUST_SIGNALS, STAGNATION_SIGNALS, URBAN_SIGNALS


def _openaq_signal(**overrides):
    signal = {
        "id": "openaq_concentrations",
        "label": "Local Monitor Concentrations (OpenAQ)",
        "status": "present",
        "pm25": None,
        "pm10": None,
        "o3_ppb": None,
        "no2_ppb": None,
        "co_ppm": None,
        "so2_ppb": None,
        "pm25_pm10_ratio": None,
        "daily_percentile": None,
        "same_hour_percentile": None,
        "same_hour_median": None,
    }
    signal.update(overrides)
    return signal


def _with_openaq(signals, openaq_signal):
    return [dict(s) for s in signals] + [openaq_signal]


def _unavailable_openaq_signal():
    return {
        "id": "openaq_concentrations",
        "label": "Local Monitor Concentrations (OpenAQ)",
        "status": "unavailable",
        "details": "No EPA reference monitor within 25 km",
    }

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


def test_openaq_coarse_ratio_boosts_dust():
    observation = {"aqi": 140, "primary_pollutant": "PM10", "category": "Unhealthy for Sensitive Groups"}
    signals = _with_openaq(
        DUST_SIGNALS,
        _openaq_signal(pm25=8.0, pm10=40.0, pm25_pm10_ratio=0.2),
    )
    hypotheses, _ = score_hypotheses(observation, signals)
    dust_h = next(h for h in hypotheses if h["id"] == "windblown_dust")
    assert any("small fraction of PM10" in s for s in dust_h["support"])
    assert dust_h["score"] >= 55
    assert dust_h["confidence"] in ["medium", "high"]
    assert hypotheses[0]["id"] == "windblown_dust"


def test_openaq_fine_ratio_boosts_smoke_with_plume_evidence():
    observation = {"aqi": 110, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.65, "density": "medium"},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0},
        _openaq_signal(pm25=17.0, pm10=20.0, pm25_pm10_ratio=0.85),
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert any("fine-particle dominated" in s for s in smoke_h["support"])
    assert any("PM2.5 measured at 17" in s for s in smoke_h["support"])
    assert smoke_h["score"] >= 60


def test_openaq_fine_ratio_clean_column_favors_urban():
    observation = {"aqi": 90, "primary_pollutant": "PM2.5", "category": "Moderate"}
    signals = _with_openaq(
        URBAN_SIGNALS,
        _openaq_signal(pm25=22.0, pm10=25.0, pm25_pm10_ratio=0.88),
    )
    hypotheses, _ = score_hypotheses(observation, signals)
    urban_h = next(h for h in hypotheses if h["id"] == "urban_industrial_pm")
    assert any("fine-particle dominated" in s for s in urban_h["support"])
    assert any("PM2.5 measured at 22" in s for s in urban_h["support"])
    assert urban_h["score"] >= 55
    assert hypotheses[0]["id"] == "urban_industrial_pm"


def test_openaq_no2_tracer_supports_urban():
    observation = {"aqi": 85, "primary_pollutant": "PM2.5", "category": "Moderate"}
    signals = _with_openaq(
        URBAN_SIGNALS,
        _openaq_signal(pm25=20.0, no2_ppb=62.0),
    )
    hypotheses, _ = score_hypotheses(observation, signals)
    urban_h = next(h for h in hypotheses if h["id"] == "urban_industrial_pm")
    assert any("nitrogen dioxide" in s for s in urban_h["support"])
    assert urban_h["score"] >= 55


def test_openaq_same_hour_anomaly_supports_stagnation():
    observation = {"aqi": 110, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = _with_openaq(
        STAGNATION_SIGNALS,
        _openaq_signal(pm25=30.0, same_hour_percentile=95.0, same_hour_median=10.0),
    )
    hypotheses, _ = score_hypotheses(observation, signals)
    stag_h = next(h for h in hypotheses if h["id"] == "winter_stagnation")
    assert any("usual reading for this time of day" in s for s in stag_h["support"])
    assert stag_h["score"] >= 70
    assert hypotheses[0]["id"] == "winter_stagnation"


def test_openaq_same_hour_anomaly_without_calm_cold_is_open_question():
    observation = {"aqi": 90, "primary_pollutant": "PM2.5", "category": "Moderate"}
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.09},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 10.0, "direction_deg": 180.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 72.0},
        _openaq_signal(pm25=28.0, same_hour_percentile=95.0, same_hour_median=8.0, daily_percentile=96.0),
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    stag_h = next(h for h in hypotheses if h["id"] == "winter_stagnation")
    assert not any("usual reading for this time of day" in s for s in stag_h["support"])
    assert any("unusually high for this time of day" in q for q in questions)
    assert any("unusual day for this area" in q for q in questions)


def test_openaq_unavailable_is_identical_to_no_signal():
    observation = {"aqi": 85, "primary_pollutant": "PM2.5", "category": "Moderate"}
    baseline_signals = [dict(s) for s in URBAN_SIGNALS]
    with_signal = _with_openaq(URBAN_SIGNALS, _unavailable_openaq_signal())

    baseline_h, baseline_q = score_hypotheses(observation, baseline_signals)
    with_h, with_q = score_hypotheses(observation, with_signal)

    assert baseline_h == with_h
    assert baseline_q == with_q

def test_incident_web_search():
    name = asyncio.run(search_fire_incident_name("CA", "Chico", 39.72, -121.83))
    assert name is None or isinstance(name, str)

def test_uncorroborated_news_fire_stays_low():
    """Uncorroborated news hit (clean AOD + no FIRMS) must NOT count as a fire vote or elevate smoke."""
    observation = {"aqi": 85, "primary_pollutant": "PM2.5", "category": "Moderate"}
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.08},
        {"id": "firms_upwind", "status": "absent", "count": 0, "incident_name": "Sandy Fire"},
        {"id": "wind", "status": "present", "speed_mph": 6.0, "direction_deg": 180.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 65.0}
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["confidence"] == "low"
    assert not any("Confirmed active" in s for s in smoke_h["support"])
    assert any("Sandy Fire" in q for q in questions)
    top_h = hypotheses[0]
    assert top_h["id"] != "wildfire_smoke"

def test_corroborated_news_fire_elevates_smoke():
    """News hit corroborated by FIRMS thermal hotspots is allowed as positive support."""
    observation = {"aqi": 110, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.15},
        {"id": "firms_upwind", "status": "present", "count": 2, "incident_name": "Creek Fire", "nearest": {"distance_miles": 20.0, "bearing": "N", "distance_km": 32.0}},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False}
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["confidence"] in ["medium", "high"]
    assert any("Creek Fire" in s for s in smoke_h["support"])

def test_clean_aod_elevated_pm_adds_smoke_against():
    """Elevated PM with clean overhead AOD adds clear-column contradiction to smoke_against."""
    observation = {"aqi": 90, "primary_pollutant": "PM2.5", "category": "Moderate"}
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.05},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 5.0, "direction_deg": 180.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False}
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert any("Clear column overhead" in a for a in smoke_h["against"])

def test_aod_only_medium_haze_urban_beats_smoke():
    """Manhattan-shaped case: medium AOD, no FIRMS, junk news name — urban should win, smoke not high."""
    observation = {"aqi": 56, "primary_pollutant": "PM2.5", "category": "Moderate"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.49, "density": "medium"},
        {"id": "firms_upwind", "status": "absent", "count": 0, "incident_name": "Man Starts Fire"},
        {"id": "wind", "status": "present", "speed_mph": 1.9, "direction_deg": 291.0, "boundary_layer_height_m": 470.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 74.4}
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    urban_h = next(h for h in hypotheses if h["id"] == "urban_industrial_pm")
    assert smoke_h["confidence"] == "medium"
    assert smoke_h["score"] < 70
    assert not any("Man Starts Fire" in s for s in smoke_h["support"])
    assert any("Man Starts Fire" in q for q in questions)
    assert urban_h["score"] > smoke_h["score"]
    assert hypotheses[0]["id"] == "urban_industrial_pm"

def test_heavy_aod_no_firms_keeps_smoke_competitive():
    """Long-range style: heavy AOD + elevated PM without nearby FIRMS still elevates smoke over urban."""
    observation = {"aqi": 110, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.85, "density": "heavy"},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0}
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["confidence"] == "medium"
    assert smoke_h["score"] >= 70
    assert hypotheses[0]["id"] == "wildfire_smoke"

def test_burns_light_haze_extreme_pm_favors_smoke():
    """Burns-shaped: light AOD (~0.38) + Very Unhealthy PM, no FIRMS — smoke should beat urban."""
    observation = {"aqi": 227, "primary_pollutant": "PM2.5", "category": "Very Unhealthy"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.38, "density": "light"},
        {"id": "firms_upwind", "status": "absent", "count": 0, "incident_name": "Little Fire"},
        {"id": "wind", "status": "present", "speed_mph": 3.0, "direction_deg": 270.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 72.0}
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    urban_h = next(h for h in hypotheses if h["id"] == "urban_industrial_pm")
    assert smoke_h["confidence"] == "medium"
    assert smoke_h["score"] >= 60
    assert smoke_h["score"] > urban_h["score"]
    assert hypotheses[0]["id"] == "wildfire_smoke"
    assert not any("Little Fire" in s for s in smoke_h["support"])

def test_food_truck_extreme_pm_clear_aod_stays_urban():
    """Extreme local PM with clear AOD and no FIRMS stays urban — do not demote on AQI alone."""
    observation = {"aqi": 227, "primary_pollutant": "PM2.5", "category": "Very Unhealthy"}
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.08},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 5.0, "direction_deg": 180.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 75.0}
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    urban_h = next(h for h in hypotheses if h["id"] == "urban_industrial_pm")
    assert hypotheses[0]["id"] == "urban_industrial_pm"
    assert urban_h["confidence"] == "high"
    assert urban_h["score"] >= 70
    assert smoke_h["confidence"] == "low"
    assert any("Clear column overhead" in a for a in smoke_h["against"])

def test_nearby_non_upwind_firms_elevates_smoke():
    """Nearby (non-upwind) FIRMS still elevates smoke over urban, slightly weaker than upwind."""
    observation = {"aqi": 160, "primary_pollutant": "PM2.5", "category": "Unhealthy"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.35, "density": "light"},
        {
            "id": "firms_upwind",
            "status": "present",
            "count": 4,
            "alignment": "nearby",
            "nearest": {"distance_miles": 40.0, "bearing": "E", "distance_km": 64.0},
        },
        {"id": "wind", "status": "present", "speed_mph": 4.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0}
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["confidence"] == "medium"
    assert smoke_h["score"] >= 60
    assert hypotheses[0]["id"] == "wildfire_smoke"
    assert any("nearby" in s.lower() for s in smoke_h["support"])
