import asyncio
import pytest
from backend.services.geocode import geocode_location
from backend.engine.score import score_hypotheses
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
        # PM10 AQI 140 implies a 24h average of 155-254 µg/m³, so the measured
        # reading must be inside that band or the disagreement gate suppresses it.
        _openaq_signal(pm25=34.0, pm10=170.0, pm25_pm10_ratio=0.2),
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
        {"id": "hms_smoke", "status": "present", "density": "light"},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0},
        _openaq_signal(pm25=30.0, pm10=35.0, pm25_pm10_ratio=0.86),
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert any("fine-particle dominated" in s for s in smoke_h["support"])
    assert any("PM2.5 measured at 30 micrograms per cubic meter" in s for s in smoke_h["support"])
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
    assert any("PM2.5 measured at 22 micrograms per cubic meter" in s for s in urban_h["support"])
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


def test_openaq_measured_conflict_suppresses_boosts_and_discloses():
    """AQI 150 falls in the 101-150 PM2.5 band (the code's lower bound is
    35.5 µg/m³); a 12 µg/m³ monitor reading is well under half of that, so
    monitor evidence must not boost hypotheses and must be disclosed as an
    open question instead."""
    observation = {"aqi": 150, "primary_pollutant": "PM2.5", "category": "Unhealthy"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.65, "density": "medium"},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0},
        _openaq_signal(
            pm25=12.0,
            pm10=15.0,
            pm25_pm10_ratio=0.8,
            monitor={"name": "Downtown", "distance_km": 8.0, "provider": "AirNow"},
        ),
    ]
    hypotheses, open_questions = score_hypotheses(observation, signals)

    for h in hypotheses:
        assert not any("PM2.5 measured at" in s for s in h["support"])
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["score"] < 60
    assert any(
        "The reported AQI is Unhealthy (150), but the nearest air quality monitor "
        "measures 12 micrograms per cubic meter right now (8 km away)" in q
        for q in open_questions
    )


def test_openaq_conflict_disclosure_only_for_nearby_monitors():
    """A far-away monitor reading low is spatial variation: suppress the boost
    but do not surface a confusing disclosure."""
    observation = {"aqi": 150, "primary_pollutant": "PM2.5", "category": "Unhealthy"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.65, "density": "medium"},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0},
        _openaq_signal(
            pm25=12.0,
            pm10=15.0,
            pm25_pm10_ratio=0.8,
            monitor={"name": "Far Site", "distance_km": 20.0, "provider": "AirNow"},
        ),
    ]
    hypotheses, open_questions = score_hypotheses(observation, signals)

    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["score"] < 60
    assert not any("The reported AQI is Unhealthy (150)" in q for q in open_questions)


def test_openaq_conflict_uses_current_epa_pm25_breakpoints():
    """Current EPA tables: AQI 51-100 implies >= 9.1 µg/m³ (half-floor 4.55)
    and AQI 201-300 implies >= 125.5 (half-floor 62.75). Readings above those
    leniency thresholds are not conflicts; the stale pre-2024 table (12.1 /
    150.5 floors) would wrongly flag them."""
    base = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.65, "density": "medium"},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0},
    ]

    moderate_hypotheses, moderate_questions = score_hypotheses(
        {"aqi": 55, "primary_pollutant": "PM2.5", "category": "Moderate"},
        base
        + [_openaq_signal(
            pm25=5.0,
            pm10=6.5,
            pm25_pm10_ratio=0.77,
            monitor={"name": "Downtown", "distance_km": 8.0, "provider": "AirNow"},
        )],
    )
    assert not any("The reported AQI is" in q for q in moderate_questions)
    assert any(
        "nearest reporting monitor" in s
        for h in moderate_hypotheses for s in h["support"]
    )
    assert any(
        "at the AirNow monitor" in s
        for h in moderate_hypotheses for s in h["support"]
    )

    _, very_unhealthy_questions = score_hypotheses(
        {"aqi": 220, "primary_pollutant": "PM2.5", "category": "Very Unhealthy"},
        base
        + [_openaq_signal(
            pm25=70.0,
            pm10=90.0,
            pm25_pm10_ratio=0.78,
            monitor={"name": "Downtown", "distance_km": 8.0, "provider": "AirNow"},
        )],
    )
    assert not any("The reported AQI is" in q for q in very_unhealthy_questions)


def test_openaq_measured_ozone_supports_ozone_hypothesis():
    observation = {"aqi": 120, "primary_pollutant": "O3", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.1},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 4.0, "direction_deg": 90.0},
        {"id": "surface_pm_level", "status": "absent", "primary": False, "pm10_primary": False, "pm25_primary": False, "elevated": False},
        {"id": "ozone_heat", "status": "present", "primary": True, "hot_day": False, "temperature_f": 80.0},
        _openaq_signal(o3_ppb=80.0, monitor={"name": "Downtown", "distance_km": 3.0, "provider": "AirNow"}),
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    o3_h = next(h for h in hypotheses if h["id"] == "ozone_episode")
    assert any("elevated ground-level ozone (80 parts per billion)" in s for s in o3_h["support"])
    assert o3_h["score"] >= 70


def test_openaq_measured_ozone_below_standard_does_not_boost():
    observation = {"aqi": 120, "primary_pollutant": "O3", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.1},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 4.0, "direction_deg": 90.0},
        {"id": "surface_pm_level", "status": "absent", "primary": False, "pm10_primary": False, "pm25_primary": False, "elevated": False},
        {"id": "ozone_heat", "status": "present", "primary": True, "hot_day": False, "temperature_f": 80.0},
        _openaq_signal(o3_ppb=40.0),
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    o3_h = next(h for h in hypotheses if h["id"] == "ozone_episode")
    assert not any("Monitor data shows elevated ground-level ozone" in s for s in o3_h["support"])
    assert o3_h["score"] == 60


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
    """Manhattan-shaped case: medium AOD, no FIRMS, junk news name - urban should win, smoke not high."""
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

def test_govcamp_wildfire_smoke_top_with_named_incident():
    """Government Camp regression: Hazardous PM, east wind, HMS medium, WFIGS upwind
    incident, FIRMS feed down, rural town - wildfire must be top/high with the
    federal incident named, and no 'no hotspots' absence claim may appear."""
    observation = {"aqi": 360, "primary_pollutant": "PM2.5", "category": "Hazardous"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "density": "medium", "aod_value": 0.49},
        {"id": "hms_smoke", "status": "present", "density": "medium", "details": "Location is inside HMS overhead smoke plume (medium density)"},
        {"id": "firms_upwind", "status": "unavailable", "count": 0, "details": "NASA FIRMS API key not configured"},
        {"id": "wfigs_incident", "status": "present", "count": 1, "alignment": "upwind",
         "incident": {"name": "Grasshopper Fire", "size_acres": 35000, "percent_contained": 19,
                      "state": "OR", "distance_miles": 22.0, "distance_km": 35.4, "bearing": "E", "is_upwind": True},
         "details": "Federal incident registry lists 'Grasshopper Fire' (35000 acres, 19% contained) 22.0 mi E"},
        {"id": "wind", "status": "present", "speed_mph": 6.9, "direction_deg": 90.0, "boundary_layer_height_m": 60.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 59.6},
        {"id": "place_context", "status": "present", "population": 180, "rural": True, "details": "ZCTA population 180"},
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    top_h = hypotheses[0]
    assert top_h["id"] == "wildfire_smoke"
    assert top_h["confidence"] == "high"
    assert any("Grasshopper Fire" in s for s in top_h["support"])
    assert not any("No nearby upwind thermal hotspots" in s for s in top_h["against"])
    # WFIGS-only place pointer: bearing/distance fall back to the incident.
    assert top_h["place"]["bearing"] == "E"
    assert top_h["place"]["approx_km"] == 35.4
    urban_h = next(h for h in hypotheses if h["id"] == "urban_industrial_pm")
    assert urban_h["score"] < top_h["score"]


def test_wenatchee_upwind_wfigs_extreme_pm_high_with_feeds_down():
    """Wenatchee regression: Very Unhealthy PM with a verified upwind federal
    fire must reach high confidence even when FIRMS/HMS are down and the CAMS
    column AOD is only moderate (0.5)."""
    observation = {"aqi": 249, "primary_pollutant": "PM2.5", "category": "Very Unhealthy"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "density": "medium", "aod_value": 0.5},
        {"id": "hms_smoke", "status": "unavailable", "density": None},
        {"id": "firms_upwind", "status": "unavailable", "count": 0},
        {"id": "wfigs_incident", "status": "present", "count": 1, "alignment": "upwind",
         "incident": {"name": "Pioneer Fire", "size_acres": 8000, "percent_contained": 10,
                      "state": "WA", "distance_miles": 40.0, "distance_km": 64.4, "bearing": "NW", "is_upwind": True},
         "details": "Federal incident registry lists 'Pioneer Fire' (8000 acres, 10% contained) 40.0 mi NW"},
        {"id": "wind", "status": "present", "speed_mph": 5.0, "direction_deg": 315.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 78.0},
        {"id": "place_context", "status": "unavailable"},
        {"id": "openaq_concentrations", "status": "unavailable"},
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["confidence"] == "high"
    assert smoke_h["score"] == 80
    assert hypotheses[0]["id"] == "wildfire_smoke"


def test_wfigs_metadata_formatting_and_km_fallback():
    """WFIGS support text never renders empty parens when size/containment are
    absent, and the place pointer derives approx_km from miles when km is absent."""
    observation = {"aqi": 110, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.1},
        {"id": "hms_smoke", "status": "absent", "density": None},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wfigs_incident", "status": "present", "count": 1, "alignment": "upwind",
         "incident": {"name": "Bare Fire", "size_acres": None, "percent_contained": None,
                      "state": "OR", "distance_miles": 20.0, "bearing": "E", "is_upwind": True}},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 90.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0},
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    wfigs_line = next(s for s in smoke_h["support"] if s.startswith("Federal incident registry"))
    assert "()" not in wfigs_line
    assert "Bare Fire" in wfigs_line
    assert "20.0 mi E" in wfigs_line
    assert smoke_h["place"]["approx_km"] == pytest.approx(20.0 * 1.609344)
    assert smoke_h["place"]["bearing"] == "E"


def test_heavy_hms_not_demoted_by_nearby_firms():
    """A verified heavy HMS plume must not be demoted by a non-upwind FIRMS hotspot."""
    observation = {"aqi": 150, "primary_pollutant": "PM2.5", "category": "Unhealthy"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "density": "medium", "aod_value": 0.6},
        {"id": "hms_smoke", "status": "present", "density": "heavy"},
        {"id": "firms_upwind", "status": "present", "count": 2, "alignment": "nearby",
         "nearest": {"distance_miles": 30.0, "bearing": "S", "distance_km": 48.0}},
        {"id": "wfigs_incident", "status": "absent", "incident": None, "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0},
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["confidence"] == "high"
    assert smoke_h["score"] == 85


def test_hms_present_but_low_surface_pm_stays_aloft():
    """R1: verified smoke overhead with clean ground air must not crown wildfire."""
    observation = {"aqi": 40, "primary_pollutant": "PM2.5", "category": "Good"}
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.1},
        {"id": "hms_smoke", "status": "present", "density": "medium"},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wfigs_incident", "status": "absent", "incident": None, "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": False},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0},
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["confidence"] == "low"
    assert smoke_h["score"] == 30
    assert any("aloft or regional" in s for s in smoke_h["against"])


def test_firms_present_but_low_surface_pm_stays_aloft():
    """A nearby FIRMS hotspot with clean ground air must not crown wildfire smoke."""
    observation = {"aqi": 35, "primary_pollutant": "PM2.5", "category": "Good"}
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.1},
        {"id": "hms_smoke", "status": "absent", "density": None},
        {"id": "firms_upwind", "status": "present", "count": 2, "alignment": "upwind",
         "nearest": {"distance_miles": 20.0, "bearing": "N", "distance_km": 32.0}},
        {"id": "wfigs_incident", "status": "absent", "incident": None, "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": False},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0},
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["confidence"] == "low"
    assert smoke_h["score"] == 30
    assert any("aloft or regional" in s for s in smoke_h["against"])


def test_news_name_corroborated_by_hms_only_appears_in_smoke_support():
    """News + HMS corroboration is a fire vote; no unverified-news open question."""
    observation = {"aqi": 110, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.3, "density": "light"},
        {"id": "hms_smoke", "status": "present", "density": "light"},
        {"id": "firms_upwind", "status": "absent", "count": 0, "incident_name": "Creek Fire"},
        {"id": "wfigs_incident", "status": "absent", "incident": None, "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0},
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert any("Creek Fire" in s for s in smoke_h["support"])
    assert not any("Creek Fire" in q for q in questions)
    assert smoke_h["score"] >= 70  # HMS light + elevated PM -> medium 70


def test_all_fire_feeds_unavailable_no_absence_claims():
    """All three verified feeds down: no hotspot-absence claims, each outage disclosed."""
    observation = {"aqi": 150, "primary_pollutant": "PM2.5", "category": "Unhealthy"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "density": "medium", "aod_value": 0.45},
        {"id": "hms_smoke", "status": "unavailable", "density": None},
        {"id": "firms_upwind", "status": "unavailable", "count": 0},
        {"id": "wfigs_incident", "status": "unavailable", "incident": None, "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 6.9, "direction_deg": 90.0, "boundary_layer_height_m": 60.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 59.6},
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert not any("No nearby upwind thermal hotspots" in s for s in smoke_h["against"])
    assert not any("No active upwind fires" in s for s in smoke_h["against"])
    outage_qs = [q for q in questions if "unavailable" in q.lower() and "not evidence" in q.lower()]
    assert len(outage_qs) >= 3  # FIRMS + HMS + WFIGS outages disclosed


def test_firms_outage_does_not_flip_urban_to_smoke():
    """A FIRMS outage with medium haze must not silently crown wildfire smoke over urban."""
    observation = {"aqi": 110, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "density": "medium", "aod_value": 0.6},
        {"id": "hms_smoke", "status": "absent", "density": None},
        {"id": "firms_upwind", "status": "unavailable", "count": 0, "details": "NASA FIRMS API key not configured"},
        {"id": "wfigs_incident", "status": "absent", "incident": None, "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 6.9, "direction_deg": 90.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 59.6},
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    urban_h = next(h for h in hypotheses if h["id"] == "urban_industrial_pm")
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert urban_h["score"] == 70
    assert urban_h["confidence"] == "high"
    assert urban_h["score"] > smoke_h["score"]
    assert hypotheses[0]["id"] == "urban_industrial_pm"


def test_far_downwind_wfigs_does_not_corroborate_transport():
    """A WFIGS fire 250 mi downwind must not corroborate a news name or boost
    smoke - it stays a weak (55) signal and urban wins."""
    observation = {"aqi": 110, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.1},
        {"id": "hms_smoke", "status": "absent", "density": None},
        {"id": "firms_upwind", "status": "absent", "count": 0, "incident_name": "Far Fire"},
        {"id": "wfigs_incident", "status": "present", "count": 1, "alignment": "nearby",
         "incident": {"name": "Far Fire", "size_acres": 5000, "percent_contained": 10,
                      "state": "CA", "distance_miles": 250.0, "bearing": "W", "is_upwind": False}},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 90.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0},
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    # The news mention is NOT corroborated (no upwind/close verified feed).
    assert not any(s.startswith("Recent news mention") for s in smoke_h["support"])
    assert any("Far Fire" in q for q in questions)
    # Non-aligned far WFIGS does not promote smoke at all; urban wins decisively.
    assert smoke_h["score"] == 25
    assert smoke_h["confidence"] == "low"
    urban_h = next(h for h in hypotheses if h["id"] == "urban_industrial_pm")
    assert urban_h["score"] > smoke_h["score"]
    assert hypotheses[0]["id"] == "urban_industrial_pm"


def test_rural_unexplained_pm_caps_urban():
    """Small-community gate: elevated PM with no verified fire/dust/stagnation and a rural ZCTA caps urban at medium/40."""
    observation = {"aqi": 120, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.08},
        {"id": "hms_smoke", "status": "absent", "density": None},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wfigs_incident", "status": "absent", "incident": None, "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 6.9, "direction_deg": 90.0, "boundary_layer_height_m": 60.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 59.6},
        {"id": "place_context", "status": "present", "population": 180, "rural": True, "details": "ZCTA population 180"},
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    urban_h = next(h for h in hypotheses if h["id"] == "urban_industrial_pm")
    assert urban_h["confidence"] == "medium"
    assert urban_h["score"] == 40
    assert any("small community" in q for q in questions)


def test_rural_gate_tracer_lifts_to_medium_but_not_high():
    """Rural gate + measured NO2 tracer: urban is lifted to medium/55 (specific
    local-combustion evidence), but never returns to high."""
    observation = {"aqi": 120, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.08},
        {"id": "hms_smoke", "status": "absent", "density": None},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wfigs_incident", "status": "absent", "incident": None, "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 6.9, "direction_deg": 90.0, "boundary_layer_height_m": 60.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 59.6},
        {"id": "place_context", "status": "present", "population": 180, "rural": True},
        {"id": "openaq_concentrations", "status": "present", "no2_ppb": 62.0},
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    urban_h = next(h for h in hypotheses if h["id"] == "urban_industrial_pm")
    assert urban_h["confidence"] == "medium"
    assert urban_h["score"] == 55


def test_firms_unavailable_is_not_verified_absence():
    """A down FIRMS feed must not read as 'no nearby hotspots' - surface an open question instead."""
    observation = {"aqi": 120, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.49, "density": "medium"},
        {"id": "firms_upwind", "status": "unavailable", "count": 0, "details": "NASA FIRMS API key not configured"},
        {"id": "wind", "status": "present", "speed_mph": 6.9, "direction_deg": 90.0, "boundary_layer_height_m": 60.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 59.6}
    ]
    hypotheses, questions = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    urban_h = next(h for h in hypotheses if h["id"] == "urban_industrial_pm")

    # No absence-of-hotspots claim may be made while the feed is unreachable.
    assert not any("No nearby upwind thermal hotspots" in s for s in smoke_h["against"])
    assert not any("No active upwind fires" in s for s in smoke_h["against"])
    assert not any("no nearby fire hotspots" in s for s in urban_h["support"])

    # The feed outage is disclosed as an open question instead.
    assert any("unavailable" in q.lower() and "not evidence" in q.lower() for q in questions)

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
    """Burns-shaped: light AOD (~0.38) + Very Unhealthy PM, no FIRMS - smoke should beat urban."""
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
    """Extreme local PM with clear AOD and no FIRMS stays urban - do not demote on AQI alone."""
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
