"""Focused tests for the multi-year smoke-vs-dust ablation benchmark.

Covers the three required pieces: dataset assembly (``build_dataset``), one
ablation end-to-end (``severity_only`` on a perfectly-separable fixture), and
the leave-year-out splitter (``leave_year_out``). Fixtures are built by
inserting canonical records through the store's ``insert_*`` helpers — no
network access.

The chemistry fixture uses the same IMPROVE method names and element
combinations as ``test_speclabels`` / ``test_accuracy_runner``: smoke days are
K/Fe (KNON = K - 0.6*Fe >= 0.1 with no soil), dust days are the five soil
elements (Malm formula, no K), mixed days have both signatures, secondary
days SO4+NO3, and ambiguous days nothing elevated. All label derivation goes
through ``speclabels``; this benchmark never re-derives a smoke/dust rule.
"""

import math
from datetime import date, timedelta

import pytest

from backend.eval.accuracy.records import (
    AqsDailyRecord,
    SpeciationRow,
    TransportWindRecord,
    WeatherDailyRecord,
)
from backend.eval.accuracy.speclabels import (
    classify_speciation,
    derive_components,
)
from backend.eval.accuracy.specbench import (
    POSITIVE_CLASS,
    SpecBenchSample,
    _dust_rule_stats,
    antecedent_precip_30d_in,
    build_dataset,
    dust_opportunity_rule_diagnostic,
    dust_opportunity_rule_prediction,
    leave_year_out,
    make_feature_sets,
    region_for_state,
    severity_matched_subset,
    transport_wind_features,
)
from backend.eval.accuracy.store import AccuracyStore

_METHOD_IMPROVE_A = "IMPROVE Module A - Elements by X-Ray Fluorescence"

# Perfectly-separable fixture: smoke days sit at AQI 150, dust days at AQI 58.
SMOKE_DAYS = [
    ("2016-07-10", "06-037-1003"),
    ("2016-08-11", "35-013-0002"),
    ("2017-07-12", "06-037-1003"),
    ("2017-08-13", "35-013-0002"),
    ("2018-07-14", "06-037-1003"),
    ("2018-08-15", "35-013-0002"),
]
DUST_DAYS = [
    ("2016-04-10", "06-037-1003"),
    ("2016-05-11", "35-013-0002"),
    ("2017-04-12", "06-037-1003"),
    ("2017-05-13", "35-013-0002"),
    ("2018-04-14", "06-037-1003"),
    ("2018-05-15", "35-013-0002"),
]


def _pm(site_id, date_local, code, conc, aqi):
    return AqsDailyRecord(
        site_id=site_id,
        state_code=site_id[:2],
        county_code=site_id[3:6],
        site_num=site_id[7:],
        parameter_code=code,
        parameter_name="PM2.5" if code in ("88101", "88502") else "PM10",
        poc=1,
        lat=40.0,
        lon=-115.0,
        date_local=date_local,
        concentration=conc,
        units="ug/m3 LC",
        aqi=aqi,
        method_code=None,
    )


def _spec(site_id, date_local, code, conc):
    return SpeciationRow(
        site_id=site_id,
        date_local=date_local,
        parameter_code=code,
        parameter_name=f"Param {code}",
        method_code="800",
        method_name=_METHOD_IMPROVE_A,
        concentration=conc,
        units="ug/m3 LC",
    )


def _weather(site_id, date_local, wind=10.0, gust=None, precip_mm=None):
    return WeatherDailyRecord(
        site_id=site_id,
        lat=40.0,
        lon=-115.0,
        date_local=date_local,
        tmax_f=80.0,
        tmin_f=60.0,
        wind_max_mph=wind,
        wind_dir_dominant_deg=250,
        precipitation_mm=precip_mm,
        wind_gust_max_mph=gust,
    )


def _smoke_chemistry(site_id, date_local):
    """K/Fe -> KNON = 1.0 - 0.6*0.5 = 0.7 >= 0.1, no soil -> biomass_smoke."""
    return [
        _spec(site_id, date_local, "88180", 1.0),  # K
        _spec(site_id, date_local, "88126", 0.5),  # Fe
    ]


def _dust_chemistry(site_id, date_local):
    """Five soil elements (Malm formula) with no K -> mineral_dust.
    Soil = 2.53+2.86+1.87+2.78+2.23 = 12.27 >= 1.0."""
    return [
        _spec(site_id, date_local, "88104", 1.0),  # Al
        _spec(site_id, date_local, "88165", 1.0),  # Si
        _spec(site_id, date_local, "88111", 1.0),  # Ca
        _spec(site_id, date_local, "88126", 1.0),  # Fe
        _spec(site_id, date_local, "88161", 1.0),  # Ti
    ]


def _mixed_chemistry(site_id, date_local):
    """Both signatures: KNON elevated AND soil >= MIXED_SOIL_MIN."""
    return _smoke_chemistry(site_id, date_local) + [
        _spec(site_id, date_local, "88104", 0.1),  # Al (pushes soil up)
        _spec(site_id, date_local, "88165", 0.5),  # Si
        _spec(site_id, date_local, "88111", 0.1),  # Ca
        _spec(site_id, date_local, "88161", 0.1),  # Ti
    ]


def _secondary_chemistry(site_id, date_local):
    return [
        _spec(site_id, date_local, "88403", 1.5),  # SO4
        _spec(site_id, date_local, "88306", 1.0),  # NO3 (sum 2.5 >= 2.0)
    ]


def _ambiguous_chemistry(site_id, date_local):
    return [
        _spec(site_id, date_local, "88180", 0.05),  # K -> KNON 0.02 < 0.1
        _spec(site_id, date_local, "88126", 0.05),  # Fe
    ]


def _populate_store(store, wind_missing_on_first_smoke=True):
    """Populate the perfectly-separable 2016-2018 fixture.

    Returns a dict of audit expectations used by the tests:
    joinable / pm_elevated / spec_distribution / binary smoke+dust counts /
    wind_imputed / ratio_missing.
    """
    for i, (date_local, site_id) in enumerate(SMOKE_DAYS):
        # PM2.5 elevated (AQI 150) plus PM10 so the ratio is present.
        store.insert_aqs_daily([
            _pm(site_id, date_local, "88101", 45.0, 150),
            _pm(site_id, date_local, "81102", 60.0, 50),
        ])
        store.insert_speciation(_smoke_chemistry(site_id, date_local))
        if not (wind_missing_on_first_smoke and i == 0):
            store.insert_weather_daily([_weather(site_id, date_local)])

    for date_local, site_id in DUST_DAYS:
        # PM10-only elevation (AQI 58); no PM2.5 mass -> ratio is missing.
        store.insert_aqs_daily([_pm(site_id, date_local, "81102", 120.0, 58)])
        store.insert_speciation(_dust_chemistry(site_id, date_local))
        store.insert_weather_daily([_weather(site_id, date_local)])

    # Excluded-but-counted days (2017).
    mixed_day, mixed_site = ("2017-06-20", "06-037-1003")
    store.insert_aqs_daily([_pm(mixed_site, mixed_day, "88101", 40.0, 120)])
    store.insert_speciation(_mixed_chemistry(mixed_site, mixed_day))
    store.insert_weather_daily([_weather(mixed_site, mixed_day)])

    secondary_day, secondary_site = ("2017-06-21", "35-013-0002")
    store.insert_aqs_daily([_pm(secondary_site, secondary_day, "88101", 35.0, 130)])
    store.insert_speciation(_secondary_chemistry(secondary_site, secondary_day))
    store.insert_weather_daily([_weather(secondary_site, secondary_day)])

    ambiguous_day, ambiguous_site = ("2017-06-22", "06-037-1003")
    store.insert_aqs_daily([_pm(ambiguous_site, ambiguous_day, "88101", 30.0, 110)])
    store.insert_speciation(_ambiguous_chemistry(ambiguous_site, ambiguous_day))
    store.insert_weather_daily([_weather(ambiguous_site, ambiguous_day)])

    # Non-elevated day (AQI 35): smoke chemistry but excluded by the PM filter.
    clean_day, clean_site = ("2017-06-23", "35-013-0002")
    store.insert_aqs_daily([_pm(clean_site, clean_day, "88101", 8.0, 35)])
    store.insert_speciation(_smoke_chemistry(clean_site, clean_day))
    store.insert_weather_daily([_weather(clean_site, clean_day)])

    return {
        "joinable": len(SMOKE_DAYS) + len(DUST_DAYS) + 4,
        "pm_elevated": len(SMOKE_DAYS) + len(DUST_DAYS) + 3,
        "spec_distribution": {
            "biomass_smoke": len(SMOKE_DAYS),
            "mineral_dust": len(DUST_DAYS),
            "mixed": 1,
            "secondary_aerosol": 1,
            "ambiguous": 1,
        },
        "binary": len(SMOKE_DAYS) + len(DUST_DAYS),
        "wind_imputed": 1 if wind_missing_on_first_smoke else 0,
        "ratio_missing": len(DUST_DAYS),
    }


def _populate_severity_only_store(store):
    """Fixture where pm_aqi is the ONLY discriminating feature.

    Smoke and dust days share the same months (July/August), identical PM2.5
    and PM10 concentrations (identical pm25_pm10_ratio), identical wind, and
    mixed regions — so severity alone must separate them. Smoke sits at
    pm_aqi=150, dust at pm_aqi=58 (all other features carry no signal)."""
    smoke = [
        ("2016-07-10", "06-037-1003"),
        ("2016-08-11", "35-013-0002"),
        ("2017-07-12", "06-037-1003"),
        ("2017-08-13", "35-013-0002"),
        ("2018-07-14", "06-037-1003"),
        ("2018-08-15", "35-013-0002"),
    ]
    dust = [
        ("2016-07-15", "35-013-0002"),
        ("2016-08-16", "06-037-1003"),
        ("2017-07-17", "35-013-0002"),
        ("2017-08-18", "06-037-1003"),
        ("2018-07-19", "35-013-0002"),
        ("2018-08-20", "06-037-1003"),
    ]
    for date_local, site_id in smoke:
        store.insert_aqs_daily([
            _pm(site_id, date_local, "88101", 45.0, 150),
            _pm(site_id, date_local, "81102", 60.0, 50),
        ])
        store.insert_speciation(_smoke_chemistry(site_id, date_local))
        store.insert_weather_daily([_weather(site_id, date_local)])
    for date_local, site_id in dust:
        # Identical concentrations -> identical ratio; PM2.5 row below
        # elevation, PM10 row carries the elevation.
        store.insert_aqs_daily([
            _pm(site_id, date_local, "88101", 45.0, 40),
            _pm(site_id, date_local, "81102", 60.0, 58),
        ])
        store.insert_speciation(_dust_chemistry(site_id, date_local))
        store.insert_weather_daily([_weather(site_id, date_local)])


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------


def test_build_dataset_assembles_binary_smoke_dust(tmp_path):
    with AccuracyStore(tmp_path / "accuracy.db") as store:
        expectations = _populate_store(store)

        calls = []
        dataset = build_dataset(
            store, progress=lambda i, total: calls.append((i, total))
        )

    assert calls[-1] == (expectations["joinable"], expectations["joinable"])
    assert dataset.joinable_site_days == expectations["joinable"]
    assert dataset.pm_elevated == expectations["pm_elevated"]
    assert dataset.spec_distribution == expectations["spec_distribution"]
    assert len(dataset.samples) == expectations["binary"]

    # Only biomass_smoke and mineral_dust enter the binary set.
    labels = {s.spec_label for s in dataset.samples}
    assert labels == {POSITIVE_CLASS, "mineral_dust"}
    assert dataset.wind_imputed == expectations["wind_imputed"]
    assert dataset.ratio_missing == expectations["ratio_missing"]

    # Feature audit: the first smoke day has no weather (wind None) and all
    # dust days (PM10-only) have no ratio.
    by_key = {(s.site_id, s.date_local): s for s in dataset.samples}
    smoke_without_weather = by_key[("06-037-1003", "2016-07-10")]
    assert smoke_without_weather.wind_max_mph is None
    assert smoke_without_weather.pm25_pm10_ratio is not None

    dust = by_key[("06-037-1003", "2016-04-10")]
    assert dust.wind_max_mph == 10.0
    assert dust.pm25_pm10_ratio is None
    assert dust.pm_aqi == 58
    assert dust.region == region_for_state("06")

    smoke = by_key[("35-013-0002", "2016-08-11")]
    assert smoke.region == region_for_state("35")
    assert smoke.pm_aqi == 150

    # Labels come from speclabels (not re-derived here).
    comp = derive_components(
        {r.parameter_code: r.concentration for r in _smoke_chemistry("35-013-0002", "2016-08-11")},
        {r.parameter_code: r.method_name for r in _smoke_chemistry("35-013-0002", "2016-08-11")},
    )
    assert classify_speciation(comp, elevated=True) == POSITIVE_CLASS


def test_build_dataset_per_year_and_exclusion_counts(tmp_path):
    with AccuracyStore(tmp_path / "accuracy.db") as store:
        _populate_store(store)
        dataset = build_dataset(store)

    # Per-year composition counts cover the excluded classes too.
    assert dataset.per_year_spec[2017]["mixed"] == 1
    assert dataset.per_year_spec[2017]["secondary_aerosol"] == 1
    assert dataset.per_year_spec[2017]["ambiguous"] == 1
    assert sum(dataset.per_year_spec[2017].values()) == 7  # 2 smoke + 2 dust + 3
    assert dataset.per_year_spec[2016]["biomass_smoke"] == 2
    assert dataset.per_year_spec[2016]["mineral_dust"] == 2


def test_transport_wind_feature_encoding():
    """Meteorological direction FROM which the wind blows, circular sin/cos."""
    # A wind blowing TOWARD the east (u=+10, v=0) comes FROM the west (270°).
    speed, sin_dir, cos_dir = transport_wind_features(10.0, 0.0)
    assert speed == pytest.approx(10.0)
    assert sin_dir == pytest.approx(math.sin(math.radians(270.0)))
    assert cos_dir == pytest.approx(math.cos(math.radians(270.0)))

    # A wind blowing TOWARD the south (u=0, v=-10) comes FROM the north (0°).
    speed, sin_dir, cos_dir = transport_wind_features(0.0, -10.0)
    assert speed == pytest.approx(10.0)
    assert sin_dir == pytest.approx(0.0, abs=1e-9)
    assert cos_dir == pytest.approx(1.0, abs=1e-9)

    # A wind blowing TOWARD the NE (u=+10, v=+10) comes FROM the SW (225°).
    speed, sin_dir, cos_dir = transport_wind_features(10.0, 10.0)
    assert speed == pytest.approx(10.0 * math.sqrt(2.0))
    assert sin_dir == pytest.approx(math.sin(math.radians(225.0)))
    assert cos_dir == pytest.approx(math.cos(math.radians(225.0)))

    # Missing component -> all-None features (t850_missing flags the gap).
    assert transport_wind_features(None, 5.0) == (None, None, None)
    assert transport_wind_features(5.0, None) == (None, None, None)


def test_make_feature_sets_include_t850_ablations():
    feature_sets = make_feature_sets(["california", "mountain"])
    t850_cols = ["t850_speed", "t850_dir_sin", "t850_dir_cos", "t850_missing"]

    assert feature_sets["t850_only"] == t850_cols
    # Existing ablations are unchanged: `all` gains NO t850 columns.
    assert set(t850_cols).isdisjoint(feature_sets["all"])
    # all_plus_t850 = all + the t850 block.
    assert set(t850_cols).issubset(feature_sets["all_plus_t850"])
    assert set(feature_sets["all"]).issubset(feature_sets["all_plus_t850"])
    assert len(feature_sets["all_plus_t850"]) == len(feature_sets["all"]) + len(t850_cols)


def test_build_dataset_attaches_transport_wind(tmp_path):
    with AccuracyStore(tmp_path / "accuracy.db") as store:
        expectations = _populate_store(store)
        # Transport wind only for the first smoke day (wind FROM the west).
        store.insert_transport_wind([
            TransportWindRecord(
                site_id="06-037-1003",
                date_local="2016-07-10",
                u850=10.0,
                v850=0.0,
                source="ncep_daily",
            ),
        ])
        dataset = build_dataset(store)

    by_key = {(s.site_id, s.date_local): s for s in dataset.samples}
    smoke = by_key[("06-037-1003", "2016-07-10")]
    assert smoke.t850_speed == pytest.approx(10.0)
    assert smoke.t850_dir_sin == pytest.approx(math.sin(math.radians(270.0)))
    assert smoke.t850_dir_cos == pytest.approx(math.cos(math.radians(270.0)))

    # Every other binary sample lacks transport wind -> t850_missing audit.
    assert dataset.t850_missing == expectations["binary"] - 1
    assert dataset.t850_total == expectations["binary"]
    other = by_key[("35-013-0002", "2016-08-11")]
    assert other.t850_speed is None
    assert other.t850_dir_sin is None
    assert other.t850_dir_cos is None


# ---------------------------------------------------------------------------
# dust-opportunity features (Track B)
# ---------------------------------------------------------------------------


def test_antecedent_precip_30d_in_window_sum_and_missing_flags():
    """The 30-day window is the sample day + 29 prior days, converted to
    inches; missing days count as 0 but flag ``missing``."""
    # Full 30-day window with known values (all in mm -> inches).
    # 12.7 mm/day * 30 days = 381 mm = 15.0 in.
    precip = {
        (date(2016, 7, 31) - timedelta(days=i)).isoformat(): 12.7
        for i in range(30)
    }
    total, missing = antecedent_precip_30d_in(precip, "2016-07-31")
    assert missing is False
    assert total == pytest.approx(15.0)

    # Window spans a month boundary (2016-01-15 -> 2015-12-17..2016-01-15).
    precip = {
        (date(2016, 1, 15) - timedelta(days=i)).isoformat(): 25.4
        for i in range(30)
    }
    total, missing = antecedent_precip_30d_in(precip, "2016-01-15")
    assert missing is False
    assert total == pytest.approx(30.0)

    # A missing day in the middle counts as 0 for the sum but flags missing.
    precip = {
        (date(2016, 7, 31) - timedelta(days=i)).isoformat(): 12.7
        for i in range(30)
    }
    precip["2016-07-10"] = None  # known row, null precip
    del precip["2016-07-09"]  # no stored row at all
    total, missing = antecedent_precip_30d_in(precip, "2016-07-31")
    assert missing is True
    assert total == pytest.approx(15.0 - 12.7 * 2 / 25.4)

    # Zero known days -> the window is entirely unknown: (None, True).
    total, missing = antecedent_precip_30d_in({}, "2016-07-31")
    assert total is None
    assert missing is True

    # Days outside the window do not leak into the sum.
    precip = {"2016-05-01": 2540.0, "2016-07-31": 25.4}
    total, missing = antecedent_precip_30d_in(precip, "2016-07-31")
    assert missing is True
    assert total == pytest.approx(1.0)


def test_build_dataset_attaches_gust_and_precip_features(tmp_path):
    with AccuracyStore(tmp_path / "accuracy.db") as store:
        expectations = _populate_store(store)
        # Give every sample day a gust and a (tiny) precip value so the
        # features attach; the antecedent days have no rows, so the 30-day
        # window is gappy (precip_missing=1) but the partial sum is real.
        for s in store.fetch_weather_daily():
            store.insert_weather_daily([
                WeatherDailyRecord(
                    site_id=s.site_id, lat=s.lat, lon=s.lon, date_local=s.date_local,
                    tmax_f=s.tmax_f, tmin_f=s.tmin_f, wind_max_mph=s.wind_max_mph,
                    wind_dir_dominant_deg=s.wind_dir_dominant_deg,
                    precipitation_mm=12.7, wind_gust_max_mph=45.0,
                )
            ])
        dataset = build_dataset(store)

    by_key = {(s.site_id, s.date_local): s for s in dataset.samples}
    dust = by_key[("06-037-1003", "2016-04-10")]
    assert dust.gust_max_mph == 45.0
    # Sample day alone known: 12.7 mm = 0.5 in, window gappy.
    assert dust.precip_30d_in == pytest.approx(0.5)
    assert dust.precip_missing == 1

    smoke = by_key[("06-037-1003", "2016-07-10")]
    # No weather at all for the first smoke day -> gust missing, window unknown.
    assert smoke.gust_max_mph is None
    assert smoke.precip_30d_in is None
    assert smoke.precip_missing == 1

    # Audit: only the no-weather sample is gust-missing; every 30-day window
    # is gappy because the fixture only stores the sample day.
    assert dataset.gust_missing == 1
    assert dataset.gust_total == expectations["binary"]
    assert dataset.precip_missing == expectations["binary"]
    assert dataset.precip_total == expectations["binary"]


def test_dust_opportunity_rule_prediction_thresholds():
    """The literal Lamar rule: gust >= 40 mph AND antecedent 30-day precip
    <= 0.6 in predicts dust; everything else predicts smoke."""
    def sample(gust, precip):
        return SpecBenchSample(
            site_id="06-037-1003", date_local="2016-04-10", year=2016, month=4,
            region="california", spec_label="mineral_dust", pm_aqi=58,
            wind_max_mph=20.0, pm25_pm10_ratio=None,
            gust_max_mph=gust, precip_30d_in=precip,
        )

    # Gusty + dry -> dust.
    assert dust_opportunity_rule_prediction(sample(40.0, 0.6)) is True
    assert dust_opportunity_rule_prediction(sample(50.0, 0.0)) is True
    # Gusty but wet -> smoke.
    assert dust_opportunity_rule_prediction(sample(45.0, 0.61)) is False
    # Not gusty but dry -> smoke.
    assert dust_opportunity_rule_prediction(sample(39.9, 0.1)) is False
    # Missing gust or precip -> rule does not apply.
    assert dust_opportunity_rule_prediction(sample(None, 0.1)) is None
    assert dust_opportunity_rule_prediction(sample(45.0, None)) is None


def test_dust_opportunity_rule_diagnostic(tmp_path):
    def sample(year, label, gust, precip):
        return SpecBenchSample(
            site_id="06-037-1003", date_local=f"{year}-04-10", year=year, month=4,
            region="california", spec_label=label, pm_aqi=58,
            wind_max_mph=20.0, pm25_pm10_ratio=None,
            gust_max_mph=gust, precip_30d_in=precip,
        )

    # 10 dust samples, 6 gusty+dry (rule fires); 10 smoke samples, 2 falsely
    # gusty+dry. 2 dust samples have missing features (unclassified).
    samples = [
        sample(2016, "mineral_dust", 45.0, 0.1),   # TP
        sample(2017, "mineral_dust", 45.0, 0.2),   # TP
        sample(2018, "mineral_dust", 45.0, 0.3),   # TP
        sample(2019, "mineral_dust", 50.0, 0.0),   # TP
        sample(2016, "mineral_dust", 40.0, 0.6),   # TP
        sample(2017, "mineral_dust", 41.0, 0.5),   # TP
        sample(2018, "mineral_dust", 20.0, 0.1),   # FN (not gusty)
        sample(2019, "mineral_dust", 45.0, 0.9),   # FN (wet)
        sample(2016, "mineral_dust", None, 0.1),   # unclassified (missing gust)
        sample(2017, "mineral_dust", 45.0, None),  # unclassified (missing precip)
        sample(2016, "biomass_smoke", 45.0, 0.1),  # FP
        sample(2017, "biomass_smoke", 42.0, 0.4),  # FP
        sample(2018, "biomass_smoke", 10.0, 0.0),  # TN
        sample(2019, "biomass_smoke", 45.0, 2.0),  # TN
    ]
    stats = _dust_rule_stats(samples)
    assert stats["n"] == 14
    assert stats["classified"] == 12
    assert stats["coverage"] == pytest.approx(12 / 14)
    # pred_dust = 6 TP + 2 FP = 8; precision = 6/8, recall = 6/10.
    assert stats["tp"] == 6
    assert stats["fp"] == 2
    assert stats["fn"] == 4
    assert stats["precision"] == pytest.approx(0.75)
    assert stats["recall"] == pytest.approx(0.6)

    # The full diagnostic reports the same totals and a per-year breakdown.
    class FakeDataset:
        pass

    FakeDataset.samples = samples
    diag = dust_opportunity_rule_diagnostic(FakeDataset())
    assert diag["full"]["precision"] == pytest.approx(0.75)
    assert set(diag["per_year"]) == {2016, 2017, 2018, 2019}
    assert diag["per_year"][2016]["tp"] == 2  # two TP in 2016


def test_make_feature_sets_include_dust_ablations():
    feature_sets = make_feature_sets(["california", "mountain"])
    dust_cols = ["gust_max_mph", "precip_30d_in", "gust_missing", "precip_missing"]

    assert feature_sets["dust_opportunity"] == dust_cols
    # The existing `all` block does NOT contain the dust columns.
    assert set(dust_cols).isdisjoint(feature_sets["all"])
    # all_plus_dust = all + the dust block.
    assert set(dust_cols).issubset(feature_sets["all_plus_dust"])
    assert set(feature_sets["all"]).issubset(feature_sets["all_plus_dust"])
    assert len(feature_sets["all_plus_dust"]) == len(feature_sets["all"]) + len(dust_cols)
    # And the two independent additions compose: all_plus_dust still lacks t850.
    assert set(feature_sets["all_plus_t850"]).isdisjoint(dust_cols)


# ---------------------------------------------------------------------------
# one ablation + the leave-year-out splitter
# ---------------------------------------------------------------------------


def test_leave_year_out_and_severity_ablation_separates(tmp_path):
    with AccuracyStore(tmp_path / "accuracy.db") as store:
        _populate_severity_only_store(store)
        dataset = build_dataset(store)

    regions = sorted({s.region for s in dataset.samples})
    feature_sets = make_feature_sets(regions)
    years = (2016, 2017, 2018)

    result = leave_year_out(dataset, feature_sets, years=years)

    # One fold per year, per ablation, with the expected feature columns.
    assert set(result) == set(feature_sets)
    for name, folds in result.items():
        assert [f["year"] for f in folds] == list(years)
        for fold in folds:
            assert fold["n_test"] in (4,)
            assert fold["n_smoke"] == 2 and fold["n_dust"] == 2

    # pm_aqi alone gives 100% balanced accuracy on every held-out year
    # (severity is the only feature with signal in this fixture).
    sev = result["severity_only"]
    assert all(f["balanced_accuracy"] == 1.0 for f in sev)
    assert all(f["auc"] == 1.0 for f in sev)

    # Removing severity collapses to chance: no other feature carries signal.
    ams = result["all_minus_severity"]
    assert all(f["balanced_accuracy"] is not None for f in ams)
    assert all(f["balanced_accuracy"] < 1.0 for f in ams)

    # The `all` model also reaches the ceiling here (severity dominates).
    all_folds = result["all"]
    assert all(f["balanced_accuracy"] == 1.0 for f in all_folds)


def test_severity_matched_subset_requires_both_classes(tmp_path):
    with AccuracyStore(tmp_path / "accuracy.db") as store:
        _populate_store(store)
        dataset = build_dataset(store)

    # Smoke (AQI 150) and dust (AQI 58) are 92 AQI points apart — outside the
    # default tolerance of 10, so nothing matches (honest failure mode).
    assert severity_matched_subset(dataset.samples, tolerance=10.0) == []
    # With a generous tolerance every smoke sample matches a dust sample.
    pairs = severity_matched_subset(dataset.samples, tolerance=100.0)
    assert len(pairs) == len(SMOKE_DAYS)
    for smoke, dust in pairs:
        assert smoke.spec_label == POSITIVE_CLASS
        assert dust.spec_label == "mineral_dust"
        assert abs(smoke.pm_aqi - dust.pm_aqi) <= 100.0


def test_specbench_command_renders_report(tmp_path):
    """The ``specbench`` CLI path runs end-to-end on a store (no network) and
    renders the full text report with the benchmark sections."""
    from backend.eval.accuracy.specbench import specbench_command

    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path) as store:
        _populate_store(store)

    text = specbench_command(str(db_path))
    assert "MULTI-YEAR SMOKE-vs-DUST" in text
    assert "LEAVE-YEAR-OUT" in text
    assert "REGION-SEASON HOLDOUT" in text
    assert "SEVERITY-MATCHED SUBSET" in text
    assert "ABSTENTION CURVE" in text
    assert "FINDINGS" in text
    # Track B: the dust-opportunity ablation and the Lamar hard-rule section
    # must render alongside the existing benchmark sections.
    assert "dust_opportunity" in text
    assert "all_plus_dust" in text
    assert "DUST-OPPORTUNITY HARD RULE" in text
    assert "gust_max_mph" in text
    assert "precip_30d_in" in text
