"""Tests for the accuracy-eval label-derivation layer (Phase 1c).

Pure-function tests build small AqsDailyRecord/WeatherDailyRecord fixtures
directly — no store or network access — plus one store round-trip test.
"""

from backend.eval.accuracy.labels import (
    LABEL_CLASSES,
    LABEL_KIND,
    PARAMETER_TO_POLLUTANT,
    PRECISION_TIER_VALIDATED,
    build_observation,
    classify_sample,
)
from backend.eval.accuracy.records import (
    AqsDailyRecord,
    LabelRecord,
    Observation,
    WeatherDailyRecord,
)
from backend.eval.accuracy.store import AccuracyStore

SITE_ID = "06-037-0002"
DATE = "2023-07-01"

_PARAM_NAMES = {
    "88101": "PM2.5",
    "88502": "PM2.5",
    "81102": "PM10",
    "44201": "O3",
    "42602": "NO2",
    "42101": "CO",
    "42401": "SO2",
}


def _aqs(parameter_code, aqi, concentration=None, poc=1):
    return AqsDailyRecord(
        site_id=SITE_ID,
        state_code="06",
        county_code="037",
        site_num="0002",
        parameter_code=parameter_code,
        parameter_name=_PARAM_NAMES[parameter_code],
        poc=poc,
        lat=33.9372,
        lon=-118.1919,
        date_local=DATE,
        concentration=concentration,
        units=None,
        aqi=aqi,
        method_code=None,
    )


def _weather(tmin=None, wind=None):
    return WeatherDailyRecord(
        site_id=SITE_ID,
        lat=33.9372,
        lon=-118.1919,
        date_local=DATE,
        tmax_f=None,
        tmin_f=tmin,
        wind_max_mph=wind,
        wind_dir_dominant_deg=None,
    )


def _obs(aqi, primary):
    return Observation(aqi=aqi, primary_pollutant=primary, pollutant_aqi={}, concentrations={})


def _classify(obs, weather=None, smoke_density=None, upwind_fire=None, rural=None):
    return classify_sample(
        obs,
        weather,
        smoke_density=smoke_density,
        upwind_fire=upwind_fire,
        rural=rural,
        site_id=SITE_ID,
        date_local=DATE,
    )


# ---------------------------------------------------------------------------
# build_observation
# ---------------------------------------------------------------------------

def test_build_observation_max_aqi_and_primary():
    obs = build_observation([_aqs("88101", 42, 15.5), _aqs("44201", 95, 0.055)])
    assert obs.aqi == 95
    assert obs.primary_pollutant == "O3"
    assert obs.pollutant_aqi == {"PM2.5": 42, "O3": 95}
    assert obs.concentrations == {"PM2.5": 15.5, "O3": 0.055}


def test_build_observation_tie_break_prefers_pm25():
    obs = build_observation([_aqs("88101", 141), _aqs("81102", 141), _aqs("44201", 55)])
    assert obs.aqi == 141
    assert obs.primary_pollutant == "PM2.5"
    assert obs.pollutant_aqi == {"PM2.5": 141, "PM10": 141, "O3": 55}


def test_build_observation_tie_break_pm10_over_o3():
    obs = build_observation([_aqs("81102", 141), _aqs("44201", 141)])
    assert obs.primary_pollutant == "PM10"


def test_build_observation_aggregates_multiple_pocs():
    obs = build_observation([_aqs("88101", 42, 15.5, poc=1), _aqs("88101", 60, 21.0, poc=2)])
    assert obs.aqi == 60
    assert obs.primary_pollutant == "PM2.5"
    assert obs.pollutant_aqi == {"PM2.5": 60}
    assert obs.concentrations == {"PM2.5": 21.0}


def test_build_observation_no_non_null_aqi():
    obs = build_observation([_aqs("88101", None), _aqs("44201", None)])
    assert obs.aqi is None
    assert obs.primary_pollutant == ""
    assert obs.pollutant_aqi == {}


def test_build_observation_ignores_unknown_parameter():
    unknown = AqsDailyRecord(
        site_id=SITE_ID, state_code="06", county_code="037", site_num="0002",
        parameter_code="99999", parameter_name="Mystery", poc=1, lat=33.9372,
        lon=-118.1919, date_local=DATE, concentration=99.0, units=None,
        aqi=999, method_code=None,
    )
    obs = build_observation([unknown])
    assert obs.aqi is None
    assert obs.primary_pollutant == ""
    assert obs.pollutant_aqi == {}


# ---------------------------------------------------------------------------
# classify_sample — each label class
# ---------------------------------------------------------------------------

def test_clean_at_and_below_elevated_threshold():
    for aqi in (0, 35, 50):
        rec = _classify(_obs(aqi, "PM2.5"), weather=_weather(tmin=28, wind=2))
        assert rec.label == "clean"


def test_missing_aqi_is_ambiguous_never_clean():
    # A day with no determinable AQI must be "ambiguous", not "clean" — even
    # with fire/smoke/weather signals present.
    rec = _classify(_obs(None, ""), weather=_weather(tmin=28, wind=2))
    assert rec.label == "ambiguous"
    assert rec.aqi is None
    assert "no determinable AQI" in rec.reasoning
    rec2 = _classify(
        _obs(None, "PM2.5"), smoke_density="medium", upwind_fire=True,
        weather=_weather(tmin=70, wind=6),
    )
    assert rec2.label == "ambiguous"
    assert "no determinable AQI" in rec2.reasoning


def test_wildfire_smoke_medium_plume():
    rec = _classify(
        _obs(141, "PM2.5"), smoke_density="medium", upwind_fire=False,
        weather=_weather(tmin=70, wind=6),
    )
    assert rec.label == "wildfire_smoke"


def test_wildfire_smoke_upwind_fire():
    rec = _classify(_obs(141, "PM2.5"), smoke_density=None, upwind_fire=True)
    assert rec.label == "wildfire_smoke"


def test_wildfire_smoke_light_smoke_with_upwind_fire():
    rec = _classify(_obs(141, "PM2.5"), smoke_density="light", upwind_fire=True)
    assert rec.label == "wildfire_smoke"
    assert "upwind fire" in rec.reasoning


def test_winter_stagnation_cold_and_calm():
    rec = _classify(
        _obs(110, "PM2.5"), smoke_density=None, upwind_fire=False,
        weather=_weather(tmin=28, wind=2),
    )
    assert rec.label == "winter_stagnation"


def test_windblown_dust_pm10_with_wind():
    rec = _classify(_obs(155, "PM10"), weather=_weather(tmin=60, wind=22))
    assert rec.label == "windblown_dust"


def test_ozone_episode():
    rec = _classify(_obs(125, "O3"), weather=_weather(tmin=80, wind=5))
    assert rec.label == "ozone_episode"


def test_urban_industrial_no2_primary():
    rec = _classify(_obs(90, "NO2"), weather=_weather(tmin=60, wind=8))
    assert rec.label == "urban_industrial_pm"


def test_urban_industrial_pm25_default():
    rec = _classify(
        _obs(85, "PM2.5"), smoke_density=None, upwind_fire=False, rural=False,
        weather=_weather(tmin=68, wind=8),
    )
    assert rec.label == "urban_industrial_pm"


def test_ambiguous_light_haze_cold_calm():
    rec = _classify(
        _obs(110, "PM2.5"), smoke_density="light", upwind_fire=False,
        weather=_weather(tmin=28, wind=2),
    )
    assert rec.label == "ambiguous"


def test_ambiguous_rural_pm_no_source():
    rec = _classify(
        _obs(120, "PM2.5"), smoke_density=None, upwind_fire=None, rural=True,
        weather=_weather(tmin=60, wind=7),
    )
    assert rec.label == "ambiguous"


def test_ambiguous_pm10_without_wind():
    rec = _classify(_obs(155, "PM10"), weather=_weather(tmin=60, wind=4))
    assert rec.label == "ambiguous"


def test_ambiguous_no_determinable_primary():
    rec = _classify(_obs(60, ""), weather=_weather(tmin=60, wind=8))
    assert rec.label == "ambiguous"


# ---------------------------------------------------------------------------
# classify_sample — precedence edges
# ---------------------------------------------------------------------------

def test_medium_smoke_trumps_cold_calm_stagnation():
    """Verified medium plume beats inversion: smoke, not stagnation."""
    rec = _classify(
        _obs(141, "PM2.5"), smoke_density="medium", upwind_fire=False,
        weather=_weather(tmin=20, wind=2),
    )
    assert rec.label == "wildfire_smoke"


def test_light_smoke_with_upwind_fire_trumps_cold_calm():
    rec = _classify(
        _obs(141, "PM2.5"), smoke_density="light", upwind_fire=True,
        weather=_weather(tmin=20, wind=2),
    )
    assert rec.label == "wildfire_smoke"


def test_cold_calm_without_smoke_is_stagnation_not_ambiguous():
    rec = _classify(
        _obs(110, "PM2.5"), smoke_density=None, upwind_fire=False,
        weather=_weather(tmin=20, wind=2),
    )
    assert rec.label == "winter_stagnation"


# ---------------------------------------------------------------------------
# LabelRecord shape / identity
# ---------------------------------------------------------------------------

def test_classify_sample_produces_validated_label_record():
    rec = _classify(_obs(141, "PM2.5"), smoke_density="heavy")
    assert isinstance(rec, LabelRecord)
    assert rec.site_id == SITE_ID
    assert rec.date_local == DATE
    assert rec.aqi == 141
    assert rec.primary_pollutant == "PM2.5"
    assert rec.precision_tier == "validated"
    assert rec.natural_key == (SITE_ID, DATE)
    assert isinstance(rec.reasoning, str) and rec.reasoning
    # Labels are rule-derived: they re-apply production thresholds to the same
    # archives the scorer consumes (self-consistency, not independent truth).
    assert rec.label_kind == LABEL_KIND
    assert LABEL_KIND == "rule_derived"


def test_classify_sample_falls_back_to_weather_identity():
    rec = classify_sample(
        _obs(141, "PM2.5"), _weather(tmin=70, wind=6), smoke_density="medium",
    )
    assert rec.site_id == SITE_ID
    assert rec.date_local == DATE
    assert rec.label == "wildfire_smoke"


def test_classify_sample_unknown_weather_is_neither_cold_nor_calm():
    # Missing weather means temp/wind unknown -> not cold/calm, so PM2.5 falls
    # through to urban/industrial (no fire/smoke/rural attribution).
    rec = _classify(_obs(85, "PM2.5"), weather=None)
    assert rec.label == "urban_industrial_pm"


def test_label_classes_mirror_scorer_hypotheses():
    assert LABEL_CLASSES == (
        "wildfire_smoke",
        "ozone_episode",
        "windblown_dust",
        "winter_stagnation",
        "urban_industrial_pm",
        "clean",
        "ambiguous",
    )


def test_parameter_to_pollutant_schema_map():
    assert PARAMETER_TO_POLLUTANT == {
        "88101": "PM2.5",
        "88502": "PM2.5",
        "81102": "PM10",
        "44201": "O3",
        "42602": "NO2",
        "42101": "CO",
        "42401": "SO2",
    }


def test_build_observation_treats_88502_as_pm25_mass():
    # IMPROVE speciation sites report PM2.5 mass under the non-FRM 88502 code;
    # it must aggregate exactly like FRM/FEM 88101 (AQI + concentration).
    obs = build_observation([_aqs("88502", 141, 55.0)])
    assert obs.aqi == 141
    assert obs.primary_pollutant == "PM2.5"
    assert obs.pollutant_aqi == {"PM2.5": 141}
    assert obs.concentrations == {"PM2.5": 55.0}


def test_build_observation_88502_competes_with_88101():
    # A site reporting both FRM/FEM and non-FRM PM2.5 buckets both into PM2.5;
    # the max AQI/concentration wins (no double counting across buckets).
    obs = build_observation([_aqs("88101", 42, 15.5), _aqs("88502", 60, 21.0)])
    assert obs.aqi == 60
    assert obs.primary_pollutant == "PM2.5"
    assert obs.pollutant_aqi == {"PM2.5": 60}
    assert obs.concentrations == {"PM2.5": 21.0}


# ---------------------------------------------------------------------------
# Store round-trip
# ---------------------------------------------------------------------------

def test_labels_store_roundtrip(tmp_path):
    records = [
        LabelRecord(
            site_id=SITE_ID, date_local=DATE, aqi=141, primary_pollutant="PM2.5",
            label="wildfire_smoke", precision_tier="validated",
            reasoning="PM2.5 primary with analyst-verified medium smoke plume",
        ),
        LabelRecord(
            site_id="41-003-0001", date_local="2023-07-01", aqi=90,
            primary_pollutant="O3", label="ozone_episode",
            precision_tier="validated", reasoning="O3 primary — ozone episode",
        ),
        LabelRecord(
            site_id="41-003-0001", date_local="2023-07-02", aqi=40,
            primary_pollutant="PM2.5", label="clean",
            precision_tier="validated", reasoning="AQI 40 at/below elevated threshold 50",
        ),
    ]

    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        assert store.insert_labels(records) == 3
        assert store.count_labels() == 3

        fetched = store.fetch_labels()
        assert len(fetched) == 3
        by_key = {r.natural_key: r for r in fetched}
        for rec in records:
            assert by_key[rec.natural_key] == rec

        # Filter by label class.
        ozone = store.fetch_labels(label="ozone_episode")
        assert [r.natural_key for r in ozone] == [("41-003-0001", "2023-07-01")]

        # INSERT OR REPLACE is idempotent under the natural key.
        replaced = LabelRecord(
            site_id=SITE_ID, date_local=DATE, aqi=150, primary_pollutant="PM2.5",
            label="wildfire_smoke", precision_tier="validated",
            reasoning="PM2.5 primary with analyst-verified heavy smoke plume",
        )
        assert store.insert_labels([replaced]) == 1
        assert store.count_labels() == 3
        updated = {r.natural_key: r for r in store.fetch_labels()}[replaced.natural_key]
        assert updated.aqi == 150
        assert updated.reasoning == replaced.reasoning
    finally:
        store.close()
