"""Tests for scorer-input reconstruction from archive records (Phase 1d-1).

All fixtures are constructed by inserting canonical records through the store's
``insert_*`` helpers — no network access. The smoke-day fixture wires AQS
PM2.5 elevation + weather + an HMS polygon containing the site + two upwind
FIRMS hotspots, then asserts the production ``score_hypotheses`` ranks
``wildfire_smoke`` first; a clean-day fixture asserts a low-confidence top.
"""

import json

import pytest

from backend.eval.accuracy.metrics import compute_metrics
from backend.eval.accuracy.records import (
    AqsDailyRecord,
    FirmsHotspotRecord,
    HmsSmokeRecord,
    PredictionRecord,
    WeatherDailyRecord,
)
from backend.eval.accuracy.reconstruct import (
    _firms_res_from_hotspots,
    _hms_res_from_store,
    _hotspots_for_day,
    _openaq_sig_from_aqs,
    _weather_dict,
    reconstruct_signals,
    score_sample,
)
from backend.eval.accuracy.store import AccuracyStore

SITE_ID = "06-049-0003"
DATE = "2023-07-01"
LAT = 40.0
LON = -120.0

_PARAM_NAMES = {
    "88101": "PM2.5",
    "81102": "PM10",
    "44201": "O3",
    "42602": "NO2",
    "42101": "CO",
    "42401": "SO2",
}

# A polygon covering a big box around (LON, LAT) = (-120, 40).
_POLYGON_AROUND_SITE = {
    "type": "Polygon",
    "coordinates": [[
        [-121.0, 41.0], [-119.0, 41.0], [-119.0, 39.0], [-121.0, 39.0], [-121.0, 41.0],
    ]],
}


# ---------------------------------------------------------------------------
# Record fixtures
# ---------------------------------------------------------------------------


def _aqs(parameter_code, concentration, aqi, units="ug/m3 LC", poc=1):
    return AqsDailyRecord(
        site_id=SITE_ID,
        state_code="06",
        county_code="049",
        site_num="0003",
        parameter_code=parameter_code,
        parameter_name=_PARAM_NAMES[parameter_code],
        poc=poc,
        lat=LAT,
        lon=LON,
        date_local=DATE,
        concentration=concentration,
        units=units,
        aqi=aqi,
        method_code=None,
    )


def _weather(tmax, tmin, wind_max, wind_dir):
    return WeatherDailyRecord(
        site_id=SITE_ID,
        lat=LAT,
        lon=LON,
        date_local=DATE,
        tmax_f=tmax,
        tmin_f=tmin,
        wind_max_mph=wind_max,
        wind_dir_dominant_deg=wind_dir,
    )


def _hotspot(h_lat, h_lon, frp, acq_datetime, confidence):
    return FirmsHotspotRecord(
        lat=h_lat,
        lon=h_lon,
        frp=frp,
        acq_datetime=acq_datetime,
        confidence=confidence,
        satellite="N",
        daynight="D",
    )


def _hms(density, geometry, date_local=DATE):
    return HmsSmokeRecord(
        date_local=date_local,
        density=density,
        geometry_json=json.dumps(geometry, separators=(",", ":")),
    )


def _smoke_day_store(store):
    """Populate a store with a PM2.5-elevated day plus an HMS plume and two
    upwind FIRMS hotspots around the site."""
    store.insert_aqs_daily([_aqs("88101", 45.0, 120)])
    store.insert_weather_daily([_weather(tmax=95.0, tmin=66.0, wind_max=12.0, wind_dir=250)])
    store.insert_hms_smoke([_hms("light", _POLYGON_AROUND_SITE)])
    store.insert_firms_hotspots([
        _hotspot(41.0, -121.3, 8.0, "2023-07-01T08:00:00+00:00", "high"),
        _hotspot(40.5, -121.0, 5.0, "2023-07-01T06:00:00+00:00", "nominal"),
    ])


# ---------------------------------------------------------------------------
# _weather_dict
# ---------------------------------------------------------------------------


def test_weather_dict_uses_tmax_with_tmin_fallback():
    full = _weather_dict(_weather(tmax=95.0, tmin=66.0, wind_max=12.0, wind_dir=250))
    assert full == {
        "wind_speed_mph": 12.0,
        "wind_direction_deg": 250,
        "temperature_f": 95.0,
        "boundary_layer_height_m": None,
    }
    # Missing tmax falls back to tmin; missing wind fields stay None.
    fallback = _weather_dict(_weather(tmax=None, tmin=66.0, wind_max=None, wind_dir=None))
    assert fallback["temperature_f"] == 66.0
    assert fallback["wind_speed_mph"] is None
    assert fallback["wind_direction_deg"] is None


def test_weather_dict_none_record_is_all_none():
    assert _weather_dict(None) == {
        "wind_speed_mph": None,
        "wind_direction_deg": None,
        "temperature_f": None,
        "boundary_layer_height_m": None,
    }


# ---------------------------------------------------------------------------
# _openaq_sig_from_aqs
# ---------------------------------------------------------------------------


def test_openaq_sig_units_and_ratio():
    sig = _openaq_sig_from_aqs(
        [
            _aqs("88101", 20.0, 60),
            _aqs("81102", 30.0, 55),
            _aqs("44201", 0.055, None, units="ppm"),  # -> 55.0 ppb
            _aqs("42602", 12.0, None, units="ppb"),
            _aqs("42401", 8.0, None, units="ppb"),
            _aqs("42101", 0.4, None, units="ppm"),
        ],
        DATE,
    )
    assert sig["status"] == "present"
    assert sig["pm25"] == 20.0
    assert sig["pm10"] == 30.0
    assert sig["pm25_pm10_ratio"] == pytest.approx(0.67)
    assert sig["o3_ppb"] == pytest.approx(55.0)
    assert sig["no2_ppb"] == 12.0
    assert sig["so2_ppb"] == 8.0
    assert sig["co_ppm"] == 0.4
    assert sig["monitor"] is None
    assert sig["as_of"] == DATE


def test_openaq_sig_unavailable_without_pm():
    sig = _openaq_sig_from_aqs([_aqs("44201", 0.055, None, units="ppm")], DATE)
    assert sig["status"] == "unavailable"
    assert "No PM concentration records" in sig["details"]


def test_openaq_sig_poc_selection_highest_aqi_wins():
    # Two POCs for the same parameter: the row with the MAX non-null AQI wins
    # (matching build_observation), not the last row in file order.
    sig = _openaq_sig_from_aqs(
        [
            _aqs("88101", 20.0, 60, poc=1),
            _aqs("88101", 40.0, 80, poc=2),
        ],
        DATE,
    )
    assert sig["status"] == "present"
    assert sig["pm25"] == 40.0
    assert sig["pm25_pm10_ratio"] is None


def test_openaq_sig_poc_selection_tie_break_by_concentration():
    # Same AQI on both POCs: the higher concentration wins the tie.
    sig = _openaq_sig_from_aqs(
        [
            _aqs("88101", 20.0, 60, poc=1),
            _aqs("88101", 25.0, 60, poc=2),
        ],
        DATE,
    )
    assert sig["pm25"] == 25.0


def test_openaq_sig_poc_selection_no_aqi_falls_back_to_max_concentration():
    # No row carries an AQI: the highest concentration wins.
    sig = _openaq_sig_from_aqs(
        [
            _aqs("88101", 20.0, None, poc=1),
            _aqs("88101", 25.0, None, poc=2),
        ],
        DATE,
    )
    assert sig["status"] == "present"
    assert sig["pm25"] == 25.0


# ---------------------------------------------------------------------------
# _hms_res_from_store (point-in-polygon hit/miss/unavailable)
# ---------------------------------------------------------------------------


def test_hms_hit_inside_polygon(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        store.insert_hms_smoke([
            _hms("medium", _POLYGON_AROUND_SITE),
            _hms("light", {
                "type": "Polygon",
                "coordinates": [[
                    [-125.0, 43.0], [-124.0, 43.0], [-124.0, 42.0], [-125.0, 42.0],
                    [-125.0, 43.0],
                ]],
            }),
        ])
        res = _hms_res_from_store(store, DATE, LAT, LON)
        assert res["status"] == "present"
        assert res["density"] == "medium"  # strongest matched density wins
    finally:
        store.close()


def test_hms_miss_outside_all_polygons(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        store.insert_hms_smoke([
            _hms("heavy", {
                "type": "Polygon",
                "coordinates": [[
                    [-125.0, 43.0], [-124.0, 43.0], [-124.0, 42.0], [-125.0, 42.0],
                    [-125.0, 43.0],
                ]],
            }),
        ])
        res = _hms_res_from_store(store, DATE, LAT, LON)
        assert res["status"] == "absent"
        assert res["density"] is None
    finally:
        store.close()


def test_hms_miss_inside_polygon_hole(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        store.insert_hms_smoke([_hms("heavy", {
            "type": "Polygon",
            "coordinates": [
                [[-121.0, 41.0], [-119.0, 41.0], [-119.0, 39.0], [-121.0, 39.0],
                 [-121.0, 41.0]],
                [[-120.5, 40.5], [-119.5, 40.5], [-119.5, 39.5], [-120.5, 39.5],
                 [-120.5, 40.5]],
            ],
        })])
        # The site sits inside the exterior ring but also inside the hole.
        res = _hms_res_from_store(store, DATE, LAT, LON)
        assert res["status"] == "absent"
    finally:
        store.close()


def test_hms_no_polygons_for_date_is_unavailable(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        store.insert_hms_smoke([_hms("light", _POLYGON_AROUND_SITE, date_local="2023-07-02")])
        res = _hms_res_from_store(store, DATE, LAT, LON)
        assert res["status"] == "unavailable"
        assert res["density"] is None
        assert "no HMS polygons ingested" in res["details"]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# _hotspots_for_day / _firms_res_from_hotspots
# ---------------------------------------------------------------------------


def test_hotspots_for_day_filters_by_bbox_and_window(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        store.insert_firms_hotspots([
            _hotspot(41.0, -121.3, 8.0, "2023-07-01T08:00:00+00:00", "high"),
            # Far outside the bbox: lon -124 is beyond lon - radius/(69*cos 40).
            _hotspot(41.0, -124.0, 20.0, "2023-07-01T08:00:00+00:00", "high"),
            # Well before the 48h window (reference is 2023-07-01 23:59:59 UTC).
            _hotspot(41.0, -121.2, 15.0, "2023-06-25T08:00:00+00:00", "high"),
            # No look-ahead: a next-day detection is outside the window.
            _hotspot(41.0, -121.1, 12.0, "2023-07-02T10:00:00+00:00", "high"),
            # "low" confidence detections are dropped by weight.
            _hotspot(40.6, -121.0, 30.0, "2023-07-01T08:00:00+00:00", "low"),
        ])
        pixels = _hotspots_for_day(store, LAT, LON, DATE, wind_dir_deg=250, radius_mi=75)
        assert len(pixels) == 1
        assert pixels[0]["lat"] == 41.0
        assert pixels[0]["lon"] == -121.3
        assert pixels[0]["is_upwind"] is True
        # The reference is end-of-day 23:59:59 UTC, so a 08:00 detection is
        # ~16h old (no noon reference, no +24h look-ahead).
        assert pixels[0]["age_hours"] == 16.0
        assert pixels[0]["confidence_weight"] == 1.0
    finally:
        store.close()


def test_firms_res_present_upwind():
    pixels = [
        {
            "lat": 41.0, "lon": -121.3, "frp": 8.0, "age_hours": 4.0,
            "is_upwind": True, "confidence": "high", "confidence_weight": 1.0,
            "distance_km": 99.0, "distance_miles": 61.5, "bearing": "NW",
            "bearing_deg": 316.0, "relevance": 0.0,
        },
        {
            "lat": 40.5, "lon": -121.0, "frp": 5.0, "age_hours": 6.0,
            "is_upwind": True, "confidence": "nominal", "confidence_weight": 0.7,
            "distance_km": 65.0, "distance_miles": 40.4, "bearing": "NW",
            "bearing_deg": 304.0, "relevance": 0.0,
        },
    ]
    res = _firms_res_from_hotspots(pixels, LAT, LON, wind_dir_deg=250, radius_mi=75)
    assert res["status"] == "present"
    assert res["alignment"] == "upwind"
    assert res["count"] == 2
    assert res["total_count"] == 2
    assert res["nearest"]["is_upwind"] is True
    assert len(res["clusters"]) == 2
    assert len(res["hotspots"]) == 2


def test_firms_res_absent_for_empty_pixels():
    res = _firms_res_from_hotspots([], LAT, LON, wind_dir_deg=250, radius_mi=75)
    assert res["status"] == "absent"
    assert res["count"] == 0
    assert res["nearest"] is None
    assert res["alignment"] is None


# ---------------------------------------------------------------------------
# reconstruct_signals / score_sample end-to-end
# ---------------------------------------------------------------------------


def test_reconstruct_signals_smoke_day_ranks_wildfire_smoke_first(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        _smoke_day_store(store)
        observation, signals = reconstruct_signals(store, SITE_ID, DATE)

        # observation is the dict the engine consumes.
        assert observation["aqi"] == 120
        assert observation["primary_pollutant"] == "PM2.5"
        assert observation["pollutants"] == {"PM2.5": 45.0}

        sig_map = {s["id"]: s for s in signals}
        assert {
            "aerosol_plume", "hms_smoke", "firms_upwind", "wind",
            "surface_pm_level", "ozone_heat", "place_context",
            "openaq_concentrations",
        } <= set(sig_map)

        assert sig_map["openaq_concentrations"]["status"] == "present"
        assert sig_map["openaq_concentrations"]["pm25"] == 45.0
        assert sig_map["firms_upwind"]["status"] == "present"
        assert sig_map["firms_upwind"]["count"] == 2
        assert sig_map["firms_upwind"]["alignment"] == "upwind"
        assert sig_map["hms_smoke"]["status"] == "present"
        assert sig_map["hms_smoke"]["density"] == "light"
        assert sig_map["aerosol_plume"]["status"] == "unavailable"
        assert sig_map["wfigs_incident"]["status"] == "unavailable"
        assert sig_map["place_context"]["status"] == "unavailable"

        observation2, hypotheses, open_questions = score_sample(store, SITE_ID, DATE)
        assert observation2 == observation
        assert isinstance(open_questions, list)
        top = hypotheses[0]
        assert top["id"] == "wildfire_smoke"
        assert top["score"] >= 75
        assert top["confidence"] == "high"
    finally:
        store.close()


def test_reconstruct_signals_clean_day_top_hypothesis_low_confidence(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        store.insert_aqs_daily([_aqs("88101", 8.1, 35)])
        store.insert_weather_daily([_weather(tmax=75.0, tmin=60.0, wind_max=6.0, wind_dir=180)])

        observation, signals = reconstruct_signals(store, SITE_ID, DATE)
        sig_map = {s["id"]: s for s in signals}
        assert sig_map["firms_upwind"]["status"] == "absent"
        assert sig_map["hms_smoke"]["status"] == "unavailable"

        _, hypotheses, _ = score_sample(store, SITE_ID, DATE)
        assert hypotheses[0]["confidence"] == "low"
        assert hypotheses[0]["score"] < 50
    finally:
        store.close()


def test_score_sample_missing_day_degrades_gracefully(tmp_path):
    """A site-day with no records at all must still reconstruct and score."""
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        observation, signals = reconstruct_signals(store, "06-001-0001", "2020-01-01")
        assert observation["aqi"] is None
        assert observation["category"] is None
        sig_map = {s["id"]: s for s in signals}
        assert sig_map["openaq_concentrations"]["status"] == "unavailable"
        assert sig_map["firms_upwind"]["status"] == "absent"
        assert sig_map["hms_smoke"]["status"] == "unavailable"

        _, hypotheses, _ = score_sample(store, "06-001-0001", "2020-01-01")
        assert len(hypotheses) == 5
    finally:
        store.close()


# ---------------------------------------------------------------------------
# predictions store round-trip
# ---------------------------------------------------------------------------


def test_predictions_store_roundtrip_and_idempotent(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        records = [
            PredictionRecord(
                site_id=SITE_ID, date_local=DATE,
                true_label="wildfire_smoke", predicted_label="wildfire_smoke",
                top_score=90, top_confidence="high",
            ),
            PredictionRecord(
                site_id="41-003-0001", date_local="2023-07-01",
                true_label="clean", predicted_label="clean",
                top_score=25, top_confidence="low",
            ),
        ]
        assert store.insert_predictions(records) == 2
        assert store.count_predictions() == 2

        fetched = store.fetch_predictions()
        assert len(fetched) == 2
        by_key = {r.natural_key: r for r in fetched}
        for rec in records:
            assert by_key[rec.natural_key] == rec

        # INSERT OR REPLACE under the natural key is idempotent.
        replaced = PredictionRecord(
            site_id=SITE_ID, date_local=DATE,
            true_label="wildfire_smoke", predicted_label="ozone_episode",
            top_score=45, top_confidence="medium",
        )
        assert store.insert_predictions([replaced]) == 1
        assert store.count_predictions() == 2
        updated = {r.natural_key: r for r in store.fetch_predictions()}[replaced.natural_key]
        assert updated.predicted_label == "ozone_episode"
        assert updated.top_score == 45
    finally:
        store.close()


def test_predictions_feed_metrics(tmp_path):
    """Predictions + labels round-trip through compute_metrics end-to-end."""
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        _smoke_day_store(store)
        store.insert_predictions([
            PredictionRecord(
                site_id=SITE_ID, date_local=DATE,
                true_label="wildfire_smoke", predicted_label="wildfire_smoke",
                top_score=90, top_confidence="high",
            ),
        ])
        results = [
            (r.true_label, r.predicted_label) for r in store.fetch_predictions()
        ]
        metrics = compute_metrics(results)
        assert metrics["total"] == 1
        assert metrics["top1_accuracy"] == 1.0
        assert metrics["per_class"]["wildfire_smoke"]["f1"] == 1.0
    finally:
        store.close()
