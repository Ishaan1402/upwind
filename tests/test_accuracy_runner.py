"""End-to-end tests for the accuracy-eval runner + CLI (Phase 1d-2).

Fixtures are built by inserting canonical records through the store's
``insert_*`` helpers — no network access, no CLI subprocesses. The smoke-day
fixture reuses the reconstruct test's recipe (PM2.5 elevation + HMS plume +
two upwind FIRMS hotspots), the clean day is a low-AQI PM2.5 day, and the
ozone day an O3-primary day. The scorer outputs asserted here are the
deterministic outputs of ``score_hypotheses`` over the reconstructed signals
(encoded): the smoke day and ozone day rank their true label first, and the
clean day (AQI 35) is mapped to ``"clean"`` by the runner's clean mapping
while still recording the top hypothesis's score (wildfire_smoke at 25), so
``top1_accuracy`` for the 3-sample fixture is 1.0.
"""

import json

from backend.eval.accuracy.__main__ import (
    label_command,
    main,
    report_command,
)
from backend.eval.accuracy.metrics import compute_metrics
from backend.eval.accuracy.records import (
    AqsDailyRecord,
    FirmsHotspotRecord,
    HmsSmokeRecord,
    WeatherDailyRecord,
)
from backend.eval.accuracy.runner import (
    build_samples,
    label_sample,
    run_accuracy_eval,
)
from backend.eval.accuracy.store import AccuracyStore

SITE_ID = "06-049-0003"
SMOKE_DATE = "2023-07-01"
CLEAN_DATE = "2023-07-02"
OZONE_DATE = "2023-07-03"
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
        [-121.0, 41.0], [-119.0, 41.0], [-119.0, 39.0], [-121.0, 39.0],
        [-121.0, 41.0],
    ]],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _aqs(date_local, parameter_code, concentration, aqi, units="ug/m3 LC", poc=1):
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
        date_local=date_local,
        concentration=concentration,
        units=units,
        aqi=aqi,
        method_code=None,
    )


def _weather(date_local, tmax, tmin, wind_max, wind_dir):
    return WeatherDailyRecord(
        site_id=SITE_ID,
        lat=LAT,
        lon=LON,
        date_local=date_local,
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


def _hms(density, geometry, date_local):
    return HmsSmokeRecord(
        date_local=date_local,
        density=density,
        geometry_json=json.dumps(geometry, separators=(",", ":")),
    )


def _populate_store(store):
    """Populate a store with the three fixture days:
    - smoke day: elevated PM2.5 + HMS plume overhead + two upwind FIRMS fires,
    - clean day: low PM2.5 AQI with no fire/smoke evidence,
    - ozone day: O3-primary at an elevated AQI with no fire/smoke evidence.
    """
    store.insert_aqs_daily([_aqs(SMOKE_DATE, "88101", 45.0, 120)])
    store.insert_weather_daily(
        [_weather(SMOKE_DATE, tmax=95.0, tmin=66.0, wind_max=12.0, wind_dir=250)]
    )
    store.insert_hms_smoke([_hms("light", _POLYGON_AROUND_SITE, SMOKE_DATE)])
    store.insert_firms_hotspots([
        _hotspot(41.0, -121.3, 8.0, "2023-07-01T08:00:00+00:00", "high"),
        _hotspot(40.5, -121.0, 5.0, "2023-07-01T06:00:00+00:00", "nominal"),
    ])

    store.insert_aqs_daily([_aqs(CLEAN_DATE, "88101", 8.1, 35)])
    store.insert_weather_daily(
        [_weather(CLEAN_DATE, tmax=75.0, tmin=60.0, wind_max=6.0, wind_dir=180)]
    )

    store.insert_aqs_daily([_aqs(OZONE_DATE, "44201", 0.070, 125, units="ppm")])
    store.insert_weather_daily(
        [_weather(OZONE_DATE, tmax=80.0, tmin=62.0, wind_max=5.0, wind_dir=180)]
    )


# ---------------------------------------------------------------------------
# build_samples
# ---------------------------------------------------------------------------


def test_build_samples_returns_distinct_site_days_in_range(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        _populate_store(store)
        # A record outside the window must be excluded by the range filter.
        store.insert_aqs_daily([_aqs("2023-07-04", "88101", 5.0, 30)])

        assert build_samples(store, "2023-07-01", "2023-07-03") == [
            (SITE_ID, SMOKE_DATE),
            (SITE_ID, CLEAN_DATE),
            (SITE_ID, OZONE_DATE),
        ]
        # Inclusive single-day window.
        assert build_samples(store, "2023-07-02", "2023-07-02") == [
            (SITE_ID, CLEAN_DATE),
        ]
        # Distinctness: an extra parameter row for the same site-day must not
        # introduce a duplicate (site_id, date_local) pair.
        store.insert_aqs_daily([_aqs(SMOKE_DATE, "44201", 0.04, 45, units="ppm")])
        assert build_samples(store, "2023-07-01", "2023-07-03") == [
            (SITE_ID, SMOKE_DATE),
            (SITE_ID, CLEAN_DATE),
            (SITE_ID, OZONE_DATE),
        ]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# label_sample
# ---------------------------------------------------------------------------


def test_label_sample_assigns_correct_labels(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        _populate_store(store)

        smoke = label_sample(store, SITE_ID, SMOKE_DATE)
        assert smoke.label == "wildfire_smoke"
        assert "upwind fire" in smoke.reasoning

        clean = label_sample(store, SITE_ID, CLEAN_DATE)
        assert clean.label == "clean"

        ozone = label_sample(store, SITE_ID, OZONE_DATE)
        assert ozone.label == "ozone_episode"
        assert ozone.aqi == 125
        assert ozone.primary_pollutant == "O3"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# run_accuracy_eval
# ---------------------------------------------------------------------------


def test_run_accuracy_eval_persists_predictions_and_metrics(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        _populate_store(store)

        calls = []
        result = run_accuracy_eval(
            store,
            "2023-07-01",
            "2023-07-03",
            progress=lambda i, total: calls.append((i, total)),
        )

        # Progress callback fires once per sample, in order, with the total.
        assert calls == [(1, 3), (2, 3), (3, 3)]
        assert result["samples"] == 3
        assert store.count_predictions() == 3

        by_key = {r.natural_key: r for r in store.fetch_predictions()}
        smoke = by_key[(SITE_ID, SMOKE_DATE)]
        assert smoke.true_label == "wildfire_smoke"
        assert smoke.predicted_label == "wildfire_smoke"
        assert smoke.top_score == 90
        assert smoke.top_confidence == "high"

        clean = by_key[(SITE_ID, CLEAN_DATE)]
        assert clean.true_label == "clean"
        # The scorer has no "clean" hypothesis id, so the runner maps the
        # non-elevated-AQI day to "clean" while still recording the top
        # hypothesis's score/confidence for transparency.
        assert clean.predicted_label == "clean"
        assert clean.top_score == 25
        assert clean.top_confidence == "low"

        ozone = by_key[(SITE_ID, OZONE_DATE)]
        assert ozone.true_label == "ozone_episode"
        assert ozone.predicted_label == "ozone_episode"
        assert ozone.top_score == 60
        assert ozone.top_confidence == "medium"

        metrics = result["metrics"]
        assert metrics["total"] == 3
        assert metrics["ambiguous_count"] == 0
        assert metrics["coverage"] == 3
        assert metrics["clean_count"] == 1
        assert metrics["elevated_count"] == 2
        # All three days now agree with their true label (the clean day is
        # mapped to "clean"), so top-1 and the clean-excluded accuracy are 1.0.
        assert metrics["top1_accuracy"] == 1.0
        assert metrics["non_clean_top1_accuracy"] == 1.0
        assert metrics["confusion"]["wildfire_smoke"]["wildfire_smoke"] == 1
        assert metrics["confusion"]["clean"]["clean"] == 1
        assert metrics["confusion"]["ozone_episode"]["ozone_episode"] == 1
        # No more clean->smoke false positive: smoke has TP=1, FP=0, FN=0.
        smoke_f1 = metrics["per_class"]["wildfire_smoke"]["f1"]
        assert smoke_f1 == 1.0
        assert metrics["per_class"]["wildfire_smoke"]["precision"] == 1.0
        assert metrics["per_class"]["wildfire_smoke"]["recall"] == 1.0
        assert metrics["per_class"]["ozone_episode"]["f1"] == 1.0
    finally:
        store.close()


def test_run_accuracy_eval_limit_caps_samples(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        _populate_store(store)
        result = run_accuracy_eval(store, "2023-07-01", "2023-07-03", limit=2)
        assert result["samples"] == 2
        assert store.count_predictions() == 2
        # Deterministic order means the first two samples are the smoke day
        # and the clean day.
        keys = [r.natural_key for r in store.fetch_predictions()]
        assert keys == [(SITE_ID, SMOKE_DATE), (SITE_ID, CLEAN_DATE)]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# CLI label/report paths (command functions called directly, no subprocess)
# ---------------------------------------------------------------------------


def test_label_and_report_cli_commands(tmp_path):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path) as store:
        _populate_store(store)

    label_res = label_command(str(db_path), "2023-07-01", "2023-07-03")
    assert label_res["samples"] == 3
    assert label_res["label_counts"] == {
        "wildfire_smoke": 1,
        "clean": 1,
        "ozone_episode": 1,
    }

    # A label-only run must not write any predictions.
    with AccuracyStore(db_path) as store:
        assert store.count_predictions() == 0

    # The label distribution must match compute_metrics over the eval's own
    # true labels.
    with AccuracyStore(db_path) as store:
        run_accuracy_eval(store, "2023-07-01", "2023-07-03")

    report_res = report_command(str(db_path))
    assert report_res["samples"] == 3
    assert report_res["metrics"]["total"] == 3
    assert report_res["metrics"]["top1_accuracy"] == 1.0

    # Cross-check against compute_metrics directly over the stored pairs.
    with AccuracyStore(db_path) as store:
        fetched = store.fetch_predictions()
    direct = compute_metrics((p.true_label, p.predicted_label) for p in fetched)
    assert report_res["metrics"]["confusion"] == direct["confusion"]


def test_main_label_subcommand_prints_distribution(capsys, tmp_path):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path) as store:
        _populate_store(store)

    rc = main([
        "label",
        "--start", "2023-07-01",
        "--end", "2023-07-03",
        "--db", str(db_path),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"samples": 3' in out
    assert '"wildfire_smoke": 1' in out
    assert '"ozone_episode": 1' in out
