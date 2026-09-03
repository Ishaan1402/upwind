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
        "ratio_source": None,
        "ratio_monitor_distance_km": None,
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


def test_airnow_ratio_distance_used_in_dominance_text():
    """The fine/coarse dominance lines name the AirNow monitor and its distance
    when the ratio is AirNow-sourced, never the OpenAQ monitor's distance."""
    observation = {"aqi": 140, "primary_pollutant": "PM10", "category": "Unhealthy for Sensitive Groups"}
    signals = _with_openaq(
        DUST_SIGNALS,
        _openaq_signal(
            pm25=34.0,
            pm10=170.0,
            pm25_pm10_ratio=0.2,
            monitor={"name": "OpenAQ Monitor", "distance_km": 5.0, "provider": "AirNow"},
            ratio_source="airnow",
            ratio_monitor_distance_km=3.0,
        ),
    )
    hypotheses, _ = score_hypotheses(observation, signals)
    dust_h = next(h for h in hypotheses if h["id"] == "windblown_dust")
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")

    dust_line = next(s for s in dust_h["support"] if "small fraction of PM10" in s)
    assert "AirNow monitor 3 km away" in dust_line
    assert "nearest reporting monitor 5 km away" not in dust_line

    smoke_line = next(a for a in smoke_h["against"] if "coarse-particle dominated" in a)
    assert "AirNow monitor 3 km away" in smoke_line
    assert "nearest reporting monitor 5 km away" not in smoke_line


def test_airnow_fine_ratio_distance_in_urban_line():
    """Fine-dominance urban support also names the AirNow monitor when the ratio
    is AirNow-sourced, rather than the OpenAQ monitor."""
    observation = {"aqi": 90, "primary_pollutant": "PM2.5", "category": "Moderate"}
    signals = _with_openaq(
        URBAN_SIGNALS,
        _openaq_signal(
            pm25=22.0,
            pm10=25.0,
            pm25_pm10_ratio=0.88,
            monitor={"name": "OpenAQ Monitor", "distance_km": 8.0, "provider": "AirNow"},
            ratio_source="airnow",
            ratio_monitor_distance_km=4.0,
        ),
    )
    hypotheses, _ = score_hypotheses(observation, signals)
    urban_h = next(h for h in hypotheses if h["id"] == "urban_industrial_pm")
    line = next(s for s in urban_h["support"] if "fine-particle dominated" in s)
    assert "AirNow monitor 4 km away" in line
    assert "nearest reporting monitor 8 km away" not in line


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

@pytest.mark.honesty
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


@pytest.mark.honesty
def test_news_only_place_desc_tagged_unknown_size():
    """Track C Part 1: a news-sourced name rendered as the place pointer must be
    tagged news-reported/unknown-size, never a bare authoritative 'Incident: X'
    that invites an invented acreage/rank in the narrative."""
    observation = {"aqi": 110, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.15},
        {"id": "firms_upwind", "status": "present", "count": 2, "incident_name": "Creek Fire",
         "nearest": {"distance_miles": 20.0, "bearing": "N", "distance_km": 32.0}},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False,
         "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False}
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    desc = smoke_h["place"]["description"]
    assert "Creek Fire" in desc
    assert desc != "Incident: Creek Fire"
    assert "unknown" in desc.lower()
    assert "news" in desc.lower()

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

@pytest.mark.honesty
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
    # AOD is modeled column loading, never a fire vote: medium haze with no
    # verified fire evidence falls through to low/25 instead of medium/55.
    assert smoke_h["confidence"] == "low"
    assert smoke_h["score"] == 25
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
    # AOD is no longer a fire vote, and a medium haze without verified fire
    # evidence routes to the haze branch (high/70) rather than the plain-clear
    # urban branch (high/75). Urban must still decisively beat smoke.
    assert urban_h["score"] == 70
    assert urban_h["confidence"] == "high"
    assert urban_h["score"] > smoke_h["score"]
    assert hypotheses[0]["id"] == "urban_industrial_pm"


@pytest.mark.honesty
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


@pytest.mark.honesty
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

def test_heavy_aod_no_firms_does_not_crown_smoke():
    """Long-range style: heavy AOD + elevated PM without nearby FIRMS is modeled
    column loading, NOT fire evidence - smoke falls through to low and urban wins."""
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
    urban_h = next(h for h in hypotheses if h["id"] == "urban_industrial_pm")
    assert smoke_h["confidence"] == "low"
    assert smoke_h["score"] == 25
    assert hypotheses[0]["id"] == "urban_industrial_pm"
    assert urban_h["score"] > smoke_h["score"]

@pytest.mark.honesty
def test_light_haze_extreme_pm_without_fire_does_not_crown_smoke():
    """Burns-shaped: light AOD (~0.38) + Very Unhealthy PM, no FIRMS - smoke
    can no longer be crowned on AOD alone and falls to low; urban wins."""
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
    assert smoke_h["confidence"] == "low"
    assert smoke_h["score"] == 25
    assert urban_h["score"] > smoke_h["score"]
    assert hypotheses[0]["id"] == "urban_industrial_pm"
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


# ---------------------------------------------------------------------------
# Track A honesty: AOD is modeled column loading, not fire evidence; and windy
# PM10-primary days without a fine/coarse ratio must cap smoke in favor of dust.
# ---------------------------------------------------------------------------

@pytest.mark.honesty
def test_aod_only_heavy_plume_no_fire_evidence_smoke_low():
    """Guards B1/B2d: a heavy modeled AOD plume with elevated PM and NO verified
    fire evidence falls through to low/25 instead of the old medium/70."""
    observation = {"aqi": 110, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.9, "density": "heavy"},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0}
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["confidence"] == "low"
    assert smoke_h["score"] == 25
    # AOD alone must never reach medium/high smoke confidence.
    assert smoke_h["confidence"] != "medium"
    assert smoke_h["confidence"] != "high"


@pytest.mark.honesty
def test_aod_does_not_count_as_fire_signal_for_high_smoke():
    """Guards B1: heavy AOD + exactly one real fire vote (a single upwind FIRMS
    hotspot) must stay medium/65 - the old behavior counted AOD as a second
    fire signal and crowned smoke at high/90."""
    observation = {"aqi": 110, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.9, "density": "heavy"},
        {"id": "firms_upwind", "status": "present", "count": 1, "alignment": "upwind",
         "nearest": {"distance_miles": 45.0, "bearing": "NW", "distance_km": 72.4}},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0}
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["confidence"] == "medium"
    assert smoke_h["score"] == 65
    assert smoke_h["score"] < 90
    assert smoke_h["confidence"] != "high"


@pytest.mark.honesty
def test_ratio_missing_windy_pm10_primary_caps_smoke_dust_ranks_first():
    """Guards C: PM10-primary + elevated + high wind with NO fine/coarse ratio
    caps smoke at 45/low even with fire evidence, and windblown dust wins.
    Two fire signals (upwind FIRMS + medium HMS) would crown smoke at high/90
    pre-cap, so the dust-ranks-first assertion actually bites."""
    observation = {"aqi": 140, "primary_pollutant": "PM10", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.65, "density": "medium"},
        {"id": "hms_smoke", "status": "present", "density": "medium"},
        {"id": "firms_upwind", "status": "present", "count": 3, "alignment": "upwind",
         "nearest": {"distance_miles": 45.0, "bearing": "NW", "distance_km": 72.4}},
        {"id": "wind", "status": "present", "speed_mph": 14.0, "direction_deg": 270.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": True, "pm25_primary": False, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 75.0},
        # PM10-only monitor: pm10 present, but pm25 and the ratio are missing.
        _openaq_signal(pm25=None, pm10=170.0, pm25_pm10_ratio=None),
    ]
    hypotheses, open_questions = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    dust_h = next(h for h in hypotheses if h["id"] == "windblown_dust")
    # Pre-cap smoke would be high/90 (two fire signals > dust 85); the cap must
    # drop it to 45/low so dust genuinely outranks it.
    assert smoke_h["score"] <= 45
    assert smoke_h["confidence"] == "low"
    assert smoke_h["score"] < dust_h["score"]
    assert hypotheses[0]["id"] == "windblown_dust"
    assert any("no fine/coarse particle ratio" in q for q in open_questions)


@pytest.mark.honesty
def test_fine_ratio_present_prevents_ratio_missing_dust_cap():
    """Regression guard: fine_dominated (ratio >= 0.70) + elevated PM + smoke
    corroboration must NOT be capped by the ratio-missing dust rule."""
    observation = {"aqi": 140, "primary_pollutant": "PM10", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.65, "density": "medium"},
        {"id": "firms_upwind", "status": "present", "count": 2, "alignment": "upwind",
         "nearest": {"distance_miles": 45.0, "bearing": "NW", "distance_km": 72.4}},
        {"id": "wind", "status": "present", "speed_mph": 14.0, "direction_deg": 270.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": True, "pm25_primary": False, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 75.0},
        # Fine-dominated ratio present: pm25 well above the AQI-140 PM10 lower
        # bound of 155 (so no measured conflict) and the ratio reads fine.
        _openaq_signal(pm25=78.0, pm10=90.0, pm25_pm10_ratio=0.87),
    ]
    hypotheses, open_questions = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["confidence"] in ["medium", "high"]
    assert smoke_h["score"] >= 60
    assert smoke_h["score"] > 45
    assert not any("no fine/coarse particle ratio" in q for q in open_questions)


def test_heavy_aod_light_hms_smoke_medium_not_high():
    """Negative guard: heavy AOD + light HMS with elevated PM is medium/70,
    never high/85 - heavy column loading does not upgrade an analyst-verified
    but light smoke plume."""
    observation = {"aqi": 110, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.9, "density": "heavy"},
        {"id": "hms_smoke", "status": "present", "density": "light"},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0}
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["confidence"] == "medium"
    assert smoke_h["score"] == 70
    assert smoke_h["score"] != 85


def test_upwind_wfigs_heavy_aod_non_extreme_pm_medium_not_high():
    """Negative guard: upwind WFIGS + heavy AOD + non-extreme PM (AQI < 150) is
    medium/70, never high/80 - column loading alone must not upgrade smoke."""
    observation = {"aqi": 110, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.9, "density": "heavy"},
        {"id": "hms_smoke", "status": "absent", "density": None},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wfigs_incident", "status": "present", "count": 1, "alignment": "upwind",
         "incident": {"name": "Guard Fire", "size_acres": 5000, "percent_contained": 10,
                      "state": "OR", "distance_miles": 20.0, "bearing": "E", "is_upwind": True}},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 90.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0}
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["confidence"] == "medium"
    assert smoke_h["score"] == 70
    assert smoke_h["score"] != 80


@pytest.mark.honesty
def test_openaq_down_windy_pm10_primary_caps_smoke_dust_first():
    """OpenAQ entirely down: pm10-primary + elevated + high wind + two fire
    signals leaves no fine/coarse ratio, so the honesty cap still fires and
    windblown dust outranks smoke."""
    observation = {"aqi": 140, "primary_pollutant": "PM10", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.65, "density": "medium"},
        {"id": "hms_smoke", "status": "present", "density": "medium"},
        {"id": "firms_upwind", "status": "present", "count": 3, "alignment": "upwind",
         "nearest": {"distance_miles": 45.0, "bearing": "NW", "distance_km": 72.4}},
        {"id": "wind", "status": "present", "speed_mph": 14.0, "direction_deg": 270.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": True, "pm25_primary": False, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 75.0},
        _unavailable_openaq_signal(),
    ]
    hypotheses, open_questions = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    dust_h = next(h for h in hypotheses if h["id"] == "windblown_dust")
    assert smoke_h["score"] <= 45
    assert smoke_h["confidence"] == "low"
    assert smoke_h["score"] < dust_h["score"]
    assert hypotheses[0]["id"] == "windblown_dust"
    assert any("no fine/coarse particle ratio" in q for q in open_questions)
    assert any("cannot be confirmed against a windblown-dust alternative" in a for a in smoke_h["against"])


@pytest.mark.honesty
def test_calm_pm10_primary_ratio_missing_does_not_cap_smoke():
    """Without high wind, a missing fine/coarse ratio must NOT cap smoke: the
    dust alternative lacks lofting conditions, so fire evidence may still crown."""
    observation = {"aqi": 140, "primary_pollutant": "PM10", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.65, "density": "medium"},
        {"id": "hms_smoke", "status": "present", "density": "medium"},
        {"id": "firms_upwind", "status": "present", "count": 3, "alignment": "upwind",
         "nearest": {"distance_miles": 45.0, "bearing": "NW", "distance_km": 72.4}},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 270.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": True, "pm25_primary": False, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 75.0},
        _unavailable_openaq_signal(),
    ]
    hypotheses, open_questions = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert smoke_h["confidence"] == "high"
    assert smoke_h["score"] == 90
    assert smoke_h["score"] > 45
    assert hypotheses[0]["id"] == "wildfire_smoke"
    assert not any("no fine/coarse particle ratio" in q for q in open_questions)


@pytest.mark.honesty
def test_aod_only_light_haze_extreme_pm_urban_medium_40():
    """ISSUE 1 pin: light haze + AQI >= 150 + no fire evidence routes to the
    light-haze branch and demotes urban to medium/40, never high/75."""
    observation = {"aqi": 180, "primary_pollutant": "PM2.5", "category": "Unhealthy"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.38, "density": "light"},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0}
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    urban_h = next(h for h in hypotheses if h["id"] == "urban_industrial_pm")
    assert urban_h["confidence"] == "medium"
    assert urban_h["score"] == 40
    assert urban_h["score"] != 75


@pytest.mark.honesty
def test_aod_only_heavy_plume_urban_medium_50():
    """ISSUE 1 pin: heavy AOD with no fire evidence routes to the heavy-haze
    branch and caps urban at medium/50, never high/75."""
    observation = {"aqi": 110, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.85, "density": "heavy"},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False, "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0}
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    urban_h = next(h for h in hypotheses if h["id"] == "urban_industrial_pm")
    assert urban_h["confidence"] == "medium"
    assert urban_h["score"] == 50
    assert urban_h["score"] != 75


@pytest.mark.honesty
def test_downwind_nearest_place_desc_says_nearby_not_upwind():
    """Track C ISSUE 1: ``firms_upwind`` (an upwind cluster exists) still gates
    the confidence ladder, but the place pointer names the strongest OVERALL
    cluster - which can be a large downwind fire. The description must describe
    that named cluster as 'Nearby', never 'Upwind', and the smoke support line
    must say 'nearby (not wind-aligned)' for it."""
    observation = {"aqi": 110, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "absent", "aod_value": 0.15},
        {"id": "firms_upwind", "status": "present", "count": 1, "total_count": 2,
         "alignment": "upwind",
         "nearest": {"distance_miles": 55.0, "bearing": "S", "distance_km": 88.5, "is_upwind": False}},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 0.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False,
         "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 70.0},
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    # firms_upwind still gates the confidence ladder (an upwind cluster exists).
    assert smoke_h["confidence"] == "medium"
    assert smoke_h["score"] == 65
    # The smoke support line labels the NAMED cluster (downwind), not the upwind subset.
    assert any("nearby (not wind-aligned)" in s for s in smoke_h["support"])
    # The place pointer describes the named downwind cluster, NOT 'Upwind'.
    assert smoke_h["place"]["description"] == "Nearby hotspots 55.0 mi S"
    assert "Upwind" not in smoke_h["place"]["description"]


@pytest.mark.honesty
def test_dust_confirmed_flag_beats_high_smoke():
    """Gap 3 positive: gust >= 40 mph over antecedent-dry ground (precip <= 0.6
    in) on a PM10-primary elevated day CONFIRMS windblown dust, forcing it to
    high/>=85 and ranking it first even though two fire signals make smoke
    otherwise high/90."""
    observation = {"aqi": 140, "primary_pollutant": "PM10", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.65, "density": "medium"},
        {"id": "hms_smoke", "status": "present", "density": "medium"},
        {"id": "firms_upwind", "status": "present", "count": 3, "alignment": "upwind",
         "nearest": {"distance_miles": 45.0, "bearing": "NW", "distance_km": 72.4}},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 270.0,
         "wind_gust_mph": 45.0, "precip_30d_in": 0.1},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": True, "pm25_primary": False, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 75.0},
        _unavailable_openaq_signal(),
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    dust_h = next(h for h in hypotheses if h["id"] == "windblown_dust")
    # Sanity: without the flag the smoke ladder alone would be high/90.
    assert smoke_h["score"] == 90
    # The confirmation forces dust high, scores >= 85, and ranks first.
    assert dust_h["confidence"] == "high"
    assert dust_h["score"] >= 85
    assert dust_h["score"] > smoke_h["score"]
    assert hypotheses[0]["id"] == "windblown_dust"
    # Support line names the gust/dry confirmation; smoke carries the against line.
    assert any("confirms windblown dust" in s for s in dust_h["support"])
    assert any("Dust is confirmed over smoke" in a for a in smoke_h["against"])


@pytest.mark.honesty
def test_dust_confirmed_flag_does_not_fire_wet_ground():
    """Gap 3 negative: same gusty PM10-primary day but 30-day precip of 2.0 in
    (wet) must NOT force dust high - the flag does not fire and smoke keeps the
    top spot."""
    observation = {"aqi": 140, "primary_pollutant": "PM10", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.65, "density": "medium"},
        {"id": "hms_smoke", "status": "present", "density": "medium"},
        {"id": "firms_upwind", "status": "present", "count": 3, "alignment": "upwind",
         "nearest": {"distance_miles": 45.0, "bearing": "NW", "distance_km": 72.4}},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 270.0,
         "wind_gust_mph": 45.0, "precip_30d_in": 2.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": True, "pm25_primary": False, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 75.0},
        _unavailable_openaq_signal(),
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    dust_h = next(h for h in hypotheses if h["id"] == "windblown_dust")
    assert hypotheses[0]["id"] == "wildfire_smoke"
    assert smoke_h["confidence"] == "high"
    assert smoke_h["score"] == 90
    # The flag must not fire: dust stays medium from the ordinary ladder.
    assert dust_h["confidence"] != "high"
    assert dust_h["score"] < 85
    assert not any("confirms windblown dust" in s for s in dust_h["support"])
    assert not any("Dust is confirmed over smoke" in a for a in smoke_h["against"])


# ---------------------------------------------------------------------------
# Gap 3b: NWS dust alert / nearby METAR blowing-dust confirmations.
# One-sided signals: presence confirms dust, absence means nothing.
# ---------------------------------------------------------------------------

def _external_dust_signals(**overrides):
    """A PM10-primary, elevated day with a verified smoke plume (HMS medium
    would otherwise crown smoke at high/85); dust signals are overridable."""
    base = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.65, "density": "medium"},
        {"id": "hms_smoke", "status": "present", "density": "medium"},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 270.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": True,
         "pm25_primary": False, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 75.0},
        _unavailable_openaq_signal(),
    ]
    base.append({
        "id": "nws_dust_alert",
        "label": "NWS Dust Warning/Advisory",
        "status": "absent",
        "event": None,
        "headline": None,
    })
    base.append({
        "id": "metar_dust",
        "label": "METAR Dust Report",
        "status": "absent",
        "station": None,
        "phenomenon": None,
    })
    by_id = {s["id"]: s for s in base}
    for sig_id, updates in overrides.items():
        by_id[sig_id].update(updates)
    return list(by_id.values())


@pytest.mark.honesty
def test_nws_dust_alert_present_confirms_dust_ranks_first():
    """Gap 3b positive (NWS): an NWS dust warning overlapping a PM10-primary
    elevated day CONFIRMS windblown dust at high/>=90 and ranks it first over
    an otherwise-high (85) verified smoke plume."""
    observation = {"aqi": 140, "primary_pollutant": "PM10", "category": "Unhealthy for Sensitive Groups"}
    signals = _external_dust_signals(
        nws_dust_alert={
            "status": "present",
            "event": "Dust Storm Warning",
            "headline": "Dust Storm Warning issued for the El Paso area",
            "severity": "Severe",
        },
    )
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    dust_h = next(h for h in hypotheses if h["id"] == "windblown_dust")

    # Sanity: the verified HMS plume alone would put smoke high/85.
    assert smoke_h["score"] == 85
    assert smoke_h["score"] < dust_h["score"]
    assert dust_h["confidence"] == "high"
    assert dust_h["score"] >= 90
    assert hypotheses[0]["id"] == "windblown_dust"
    assert any("NWS issued a Dust Storm Warning overlapping this location" in s for s in dust_h["support"])
    assert any("An official/observed dust signal confirms dust over smoke" in a for a in smoke_h["against"])


@pytest.mark.honesty
def test_metar_dust_present_confirms_dust_ranks_first():
    """Gap 3b positive (METAR): a nearby METAR station reporting blowing dust
    CONFIRMS windblown dust at high/>=90 and ranks it first over smoke."""
    observation = {"aqi": 140, "primary_pollutant": "PM10", "category": "Unhealthy for Sensitive Groups"}
    signals = _external_dust_signals(
        metar_dust={
            "status": "present",
            "station": "KDUL",
            "phenomenon": "BLDU",
            "raw": "METAR KDUL 201151Z 24025KT 2SM BLDU HZ",
        },
    )
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    dust_h = next(h for h in hypotheses if h["id"] == "windblown_dust")

    assert smoke_h["score"] < dust_h["score"]
    assert dust_h["confidence"] == "high"
    assert dust_h["score"] >= 90
    assert hypotheses[0]["id"] == "windblown_dust"
    assert any("A nearby METAR station reported BLDU (blowing dust)" in s for s in dust_h["support"])
    assert any("An official/observed dust signal confirms dust over smoke" in a for a in smoke_h["against"])


@pytest.mark.honesty
def test_nws_dust_confirmation_beats_max_smoke_score():
    """Tie-breaker: when verified fire evidence would pin wildfire_smoke at the
    ladder ceiling (90) AND an NWS dust warning confirms dust on the same
    PM10-primary elevated day, dust must still rank first (pinned to 95) rather
    than losing the stable-sort tie to the earlier-inserted smoke hypothesis."""
    observation = {"aqi": 140, "primary_pollutant": "PM10", "category": "Unhealthy for Sensitive Groups"}
    signals = _external_dust_signals(
        firms_upwind={"status": "present", "count": 3, "alignment": "upwind",
                      "nearest": {"distance_miles": 45.0, "bearing": "NW", "distance_km": 72.4}},
        nws_dust_alert={"status": "present", "event": "Dust Storm Warning",
                        "headline": "Dust Storm Warning issued for the El Paso area",
                        "severity": "Severe"},
    )
    hypotheses, _ = score_hypotheses(observation, signals)
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    dust_h = next(h for h in hypotheses if h["id"] == "windblown_dust")

    # Sanity: the fire evidence alone would pin smoke at the 90 ceiling.
    assert smoke_h["score"] == 90
    assert dust_h["score"] == 95
    assert dust_h["score"] > smoke_h["score"]
    assert hypotheses[0]["id"] == "windblown_dust"


@pytest.mark.honesty
def test_dust_confirmation_absent_does_not_move_dust():
    """Gap 3b negative: absent NWS/METAR signals are NOT evidence - dust must
    be scored exactly as if the feeds were never queried (one-sided signals)."""
    observation = {"aqi": 140, "primary_pollutant": "PM10", "category": "Unhealthy for Sensitive Groups"}
    with_signals = _external_dust_signals()
    without_signals = [dict(s) for s in with_signals if s["id"] not in ("nws_dust_alert", "metar_dust")]

    with_h, _ = score_hypotheses(observation, with_signals)
    without_h, _ = score_hypotheses(observation, without_signals)
    assert with_h == without_h

    dust_h = next(h for h in with_h if h["id"] == "windblown_dust")
    smoke_h = next(h for h in with_h if h["id"] == "wildfire_smoke")
    # Without confirmation, smoke (85) still beats the ordinary dust ladder (55).
    assert smoke_h["score"] > dust_h["score"]
    assert dust_h["confidence"] != "high"
    assert dust_h["score"] == 55
    assert not any("An official/observed dust signal confirms dust over smoke" in a for a in smoke_h["against"])


@pytest.mark.honesty
def test_dust_confirmation_requires_elevated_pm10_suspect_day():
    """Gap 3b guard: an NWS dust alert alone (or METAR BLDU alone) must NOT
    crown dust on a PM2.5-primary day - the confirmation is gated on a
    dust-suspect, PM-elevated day."""
    observation = {"aqi": 140, "primary_pollutant": "PM2.5", "category": "Unhealthy for Sensitive Groups"}
    signals = [
        {"id": "aerosol_plume", "status": "present", "aod_value": 0.65, "density": "medium"},
        {"id": "hms_smoke", "status": "present", "density": "medium"},
        {"id": "firms_upwind", "status": "absent", "count": 0},
        {"id": "wind", "status": "present", "speed_mph": 8.0, "direction_deg": 270.0},
        {"id": "surface_pm_level", "status": "present", "primary": True, "pm10_primary": False,
         "pm25_primary": True, "elevated": True},
        {"id": "ozone_heat", "status": "absent", "primary": False, "hot_day": False, "temperature_f": 75.0},
        {"id": "nws_dust_alert", "status": "present", "event": "Blowing Dust Advisory"},
        {"id": "metar_dust", "status": "present", "station": "KDUL", "phenomenon": "BLDU"},
    ]
    hypotheses, _ = score_hypotheses(observation, signals)
    dust_h = next(h for h in hypotheses if h["id"] == "windblown_dust")
    smoke_h = next(h for h in hypotheses if h["id"] == "wildfire_smoke")
    assert dust_h["confidence"] != "high"
    assert dust_h["score"] < 90
    assert smoke_h["score"] > dust_h["score"]
    assert hypotheses[0]["id"] == "wildfire_smoke"
