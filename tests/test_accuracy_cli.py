"""Tests for the accuracy-eval ingest + status CLI and its store helpers.

No network access: the ingest entry points are exercised with the underlying
``ingest_*`` adapters mocked, and the store helpers run against a real tmp
SQLite file through ``AccuracyStore``. The CLI functions are invoked directly
(``main([...])``), never as subprocesses.
"""

from datetime import date, timedelta
from unittest.mock import Mock, patch

import pytest

from backend.eval.accuracy.__main__ import (
    CONUS_BBOX,
    _advance_watermark,
    _bbox_type,
    _date_type,
    main,
    status_command,
)
from backend.eval.accuracy.records import AqsDailyRecord
from backend.eval.accuracy.runner import build_samples, filter_sites_by_bbox
from backend.eval.accuracy.store import AccuracyStore


def _aqs_record(site_id, date_local, lat, lon, parameter_code="88101",
                concentration=10.0, aqi=40):
    return AqsDailyRecord(
        site_id=site_id, state_code="06", county_code="037", site_num="0002",
        parameter_code=parameter_code, parameter_name="PM2.5 - Local Conditions",
        poc=1, lat=lat, lon=lon, date_local=date_local,
        concentration=concentration, units="ug/m3 LC", aqi=aqi, method_code=None,
    )


# ---------------------------------------------------------------------------
# Watermark store helpers
# ---------------------------------------------------------------------------


def test_watermark_get_set_roundtrip(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        # A source with no watermark reads back None.
        assert store.get_watermark("hms") is None

        store.set_watermark("hms", "2020-01-05")
        assert store.get_watermark("hms") == "2020-01-05"

        # Overwriting advances the watermark.
        store.set_watermark("hms", "2020-01-06")
        assert store.get_watermark("hms") == "2020-01-06"

        # Sources are independent.
        store.set_watermark("aqs", "2020-12-31")
        assert store.get_watermark("aqs") == "2020-12-31"
        assert store.get_watermark("hms") == "2020-01-06"
        assert store.get_watermark("weather") is None
    finally:
        store.close()


def test_watermark_preserves_meta_on_plain_advance(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        store.set_watermark("firms", "2023-07-01", meta='{"bbox": "conus"}')
        store.set_watermark("firms", "2023-07-02")
        assert store.get_watermark("firms") == "2023-07-02"
        # A plain advance must not clobber metadata written earlier.
        row = store._conn.execute(
            "SELECT meta FROM ingest_state WHERE source = 'firms'"
        ).fetchone()
        assert row[0] == '{"bbox": "conus"}'
    finally:
        store.close()


def test_advance_watermark_max_semantics(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        # No existing watermark: the new end becomes the watermark.
        assert _advance_watermark(store, "aqs", "2019-12-31") == "2019-12-31"
        # A newer end advances the watermark.
        assert _advance_watermark(store, "aqs", "2021-12-31") == "2021-12-31"
        # An OLDER end must NOT regress the watermark (max semantics).
        assert _advance_watermark(store, "aqs", "2019-12-31") == "2021-12-31"
        assert store.get_watermark("aqs") == "2021-12-31"
    finally:
        store.close()


def test_fetch_aqs_sites_returns_distinct_site_lat_lon(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        store.insert_aqs_daily([
            _aqs_record("06-037-0002", "2023-07-01", 40.0, -120.0),
            # Same site on a later day: still one (site_id, lat, lon) row.
            _aqs_record("06-037-0002", "2023-07-02", 40.0, -120.0),
            _aqs_record("41-003-0001", "2023-07-01", 45.5, -122.6),
        ])
        sites = store.fetch_aqs_sites()
        assert sites == [
            ("06-037-0002", 40.0, -120.0),
            ("41-003-0001", 45.5, -122.6),
        ]
        assert store.fetch_aqs_date_bounds() == ("2023-07-01", "2023-07-02")
    finally:
        store.close()


def test_fetch_aqs_sites_empty_store(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        assert store.fetch_aqs_sites() == []
        assert store.fetch_aqs_date_bounds() is None
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Pure bbox + build_samples helpers
# ---------------------------------------------------------------------------


def test_filter_sites_by_bbox_includes_inside_and_excludes_outside():
    sites = [
        ("SITE_A", 40.0, -120.0),   # inside
        ("SITE_B", 45.5, -60.0),    # outside: lon east of the box
        ("SITE_C", 39.5, -119.0),   # inside
        ("SITE_D", 24.0, -125.0),   # exactly on the SW corner (inclusive)
        ("SITE_E", 55.0, -122.6),   # outside: lat north of the box
    ]
    bbox = CONUS_BBOX  # (-125, 24, -66, 50)
    filtered = filter_sites_by_bbox(sites, bbox)
    assert {s[0] for s in filtered} == {"SITE_A", "SITE_C", "SITE_D"}

    # A smaller regional box is a strict subset.
    regional = filter_sites_by_bbox(sites, (-121.0, 39.0, -119.0, 41.0))
    assert {s[0] for s in regional} == {"SITE_A", "SITE_C"}

    # No bbox keeps every site (the full-run default).
    assert {s[0] for s in filter_sites_by_bbox(sites, None)} == {
        "SITE_A", "SITE_B", "SITE_C", "SITE_D", "SITE_E",
    }
    # Sites with a missing coordinate are never included.
    assert filter_sites_by_bbox([("SITE_F", None, None)], bbox) == []


def test_build_samples_respects_site_ids_whitelist(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        store.insert_aqs_daily([
            _aqs_record("06-037-0002", "2023-07-01", 40.0, -120.0),
            _aqs_record("06-037-0002", "2023-07-02", 40.0, -120.0),
            _aqs_record("41-003-0001", "2023-07-01", 45.5, -122.6),
        ])
        all_samples = build_samples(store, "2023-07-01", "2023-07-02")
        assert all_samples == [
            ("06-037-0002", "2023-07-01"),
            ("06-037-0002", "2023-07-02"),
            ("41-003-0001", "2023-07-01"),
        ]
        # Whitelisting one site removes the other's site-days entirely.
        only_portland = build_samples(
            store, "2023-07-01", "2023-07-02", site_ids=["41-003-0001"]
        )
        assert only_portland == [("41-003-0001", "2023-07-01")]
        # An empty whitelist yields no samples.
        assert build_samples(store, "2023-07-01", "2023-07-02", site_ids=[]) == []
    finally:
        store.close()


# ---------------------------------------------------------------------------
# CLI arg parsing (functions called directly, no subprocess)
# ---------------------------------------------------------------------------


def test_bbox_and_date_arg_types():
    assert _bbox_type("-125,24,-66,50") == (-125.0, 24.0, -66.0, 50.0)
    assert _bbox_type(" -125 , 24 , -66 , 50 ") == (-125.0, 24.0, -66.0, 50.0)
    assert _date_type("2023-07-01") == "2023-07-01"
    with pytest.raises(Exception, match="bbox"):
        _bbox_type("1,2,3")
    with pytest.raises(Exception, match="date"):
        _date_type("not-a-date")


def test_ingest_aqs_cli_parses_years_and_sets_watermark(tmp_path):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path):
        pass  # create the schema

    mock_ingest = Mock(return_value=42)
    with patch("backend.eval.accuracy.__main__.ingest_aqs_year", mock_ingest), \
         patch("backend.eval.accuracy.__main__.RAW_DATA_DIR", tmp_path / "raw"):
        rc = main([
            "ingest", "aqs",
            "--year", "2020",
            "--years", "2019", "2021",
            "--db", str(db_path),
        ])

    assert rc == 0
    # Years are sorted and each is ingested once, in ascending order.
    assert [call.args[0] for call in mock_ingest.call_args_list] == [2019, 2020, 2021]
    # Downloads land under the standardized raw/<source>/<year>/ layout.
    assert mock_ingest.call_args_list[0].args[2] == tmp_path / "raw" / "aqs" / "2019"
    assert mock_ingest.call_args_list[-1].args[2] == tmp_path / "raw" / "aqs" / "2021"

    with AccuracyStore(db_path) as store:
        # The aqs watermark is the latest ingested year's end.
        assert store.get_watermark("aqs") == "2021-12-31"


def test_ingest_aqs_backfill_older_year_does_not_regress_watermark(tmp_path):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path) as store:
        # Newer data (2021) is already stored and the watermark reflects it.
        store.set_watermark("aqs", "2021-12-31")

    mock_ingest = Mock(return_value=42)
    with patch("backend.eval.accuracy.__main__.ingest_aqs_year", mock_ingest), \
         patch("backend.eval.accuracy.__main__.RAW_DATA_DIR", tmp_path / "raw"):
        rc = main([
            "ingest", "aqs",
            "--year", "2019",
            "--db", str(db_path),
        ])

    assert rc == 0
    assert mock_ingest.call_count == 1
    with AccuracyStore(db_path) as store:
        # Backfilling an older year must not regress the watermark.
        assert store.get_watermark("aqs") == "2021-12-31"


def test_ingest_speciation_backfill_older_year_does_not_regress_watermark(tmp_path):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path) as store:
        store.set_watermark("speciation", "2020-12-31")

    mock_ingest = Mock(return_value=42)
    with patch("backend.eval.accuracy.__main__.ingest_speciation_year", mock_ingest), \
         patch("backend.eval.accuracy.__main__.RAW_DATA_DIR", tmp_path / "raw"):
        rc = main([
            "ingest", "speciation",
            "--year", "2019",
            "--db", str(db_path),
        ])

    assert rc == 0
    assert mock_ingest.call_count == 1
    with AccuracyStore(db_path) as store:
        assert store.get_watermark("speciation") == "2020-12-31"


def test_ingest_speciation_cli_sets_watermark(tmp_path):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path):
        pass  # create the schema

    mock_ingest = Mock(return_value=42)
    with patch("backend.eval.accuracy.__main__.ingest_speciation_year", mock_ingest), \
         patch("backend.eval.accuracy.__main__.RAW_DATA_DIR", tmp_path / "raw"):
        rc = main([
            "ingest", "speciation",
            "--year", "2020",
            "--db", str(db_path),
        ])

    assert rc == 0
    assert mock_ingest.call_args.args[0] == 2020
    assert mock_ingest.call_args.args[2] == tmp_path / "raw" / "speciation" / "2020"
    with AccuracyStore(db_path) as store:
        assert store.get_watermark("speciation") == "2020-12-31"


def test_ingest_aqs_cli_accepts_retry_failed(tmp_path):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path):
        pass

    mock_ingest = Mock(return_value=0)
    with patch("backend.eval.accuracy.__main__.ingest_aqs_year", mock_ingest), \
         patch("backend.eval.accuracy.__main__.RAW_DATA_DIR", tmp_path / "raw"):
        rc = main(["ingest", "aqs", "--year", "2020", "--retry-failed",
                   "--db", str(db_path)])
    assert rc == 0
    assert mock_ingest.call_count == 1


def test_ingest_firms_cli_parses_bbox_and_defaults_to_conus(tmp_path):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path):
        pass

    calls = []
    mock_ingest = Mock(side_effect=lambda start, end, bbox, store, **kw:
                       calls.append((start, end, bbox)) or 3)
    with patch("backend.eval.accuracy.__main__.ingest_firms_historical", mock_ingest):
        rc = main([
            "ingest", "firms",
            "--start", "2023-07-01", "--end", "2023-07-03",
            "--bbox=-125,24,-66,50",
            "--db", str(db_path),
        ])
        assert rc == 0
        assert calls == [("2023-07-01", "2023-07-03", (-125.0, 24.0, -66.0, 50.0))]

        # No --bbox: firms defaults to the CONUS box.
        rc = main([
            "ingest", "firms",
            "--start", "2023-07-01", "--end", "2023-07-03",
            "--db", str(db_path),
        ])
        assert rc == 0
        assert calls[-1][2] == CONUS_BBOX

    with AccuracyStore(db_path) as store:
        assert store.get_watermark("firms") == "2023-07-03"


@pytest.mark.parametrize(
    "source,pre_existing,patched,args",
    [
        # weather: older fixed-range backfill after a newer watermark.
        (
            "weather",
            "2021-02-28",
            "backend.eval.accuracy.__main__.ingest_weather_for_sites",
            ["ingest", "weather", "--start", "2016-01-01", "--end", "2016-12-31"],
        ),
        # hms: older fixed-range backfill after a newer watermark.
        (
            "hms",
            "2021-02-28",
            "backend.eval.accuracy.__main__.ingest_hms_smoke_range",
            ["ingest", "hms", "--start", "2016-01-01", "--end", "2016-12-31"],
        ),
        # firms: older fixed-range backfill after a newer watermark.
        (
            "firms",
            "2021-02-28",
            "backend.eval.accuracy.__main__.ingest_firms_historical",
            ["ingest", "firms", "--start", "2016-01-01", "--end", "2016-12-31"],
        ),
    ],
)
def test_ingest_backfill_older_range_does_not_regress_watermark(
    tmp_path, source, pre_existing, patched, args
):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path) as store:
        store.set_watermark(source, pre_existing)
        # weather derives its site list from the AQS store.
        store.insert_aqs_daily([_aqs_record("06-037-0002", "2020-07-01", 40.0, -120.0)])

    mock_ingest = Mock(return_value=5)
    with patch(patched, mock_ingest), \
         patch("backend.eval.accuracy.__main__.RAW_DATA_DIR", tmp_path / "raw"):
        rc = main([*args, "--db", str(db_path)])

    assert rc == 0
    assert mock_ingest.call_count == 1
    with AccuracyStore(db_path) as store:
        # The older range must not regress the source's watermark.
        assert store.get_watermark(source) == pre_existing


def test_status_cli_reports_watermarks_counts_and_bounds(tmp_path, capsys):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path) as store:
        store.insert_aqs_daily([_aqs_record("06-037-0002", "2023-07-01", 40.0, -120.0)])
        store.set_watermark("aqs", "2023-12-31")

    res = status_command(str(db_path))
    assert res["watermarks"]["aqs"] == "2023-12-31"
    assert res["watermarks"]["hms"] is None
    assert res["row_counts"]["aqs_daily"] == 1
    assert res["row_counts"]["weather_daily"] == 0
    assert res["row_counts"]["labels"] == 0
    assert res["aqs_date_bounds"] == ("2023-07-01", "2023-07-01")

    rc = main(["status", "--db", str(db_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "aqs" in out and "2023-12-31" in out
    assert "aqs_daily" in out


def test_status_reports_empty_store(tmp_path):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path):
        pass
    res = status_command(str(db_path))
    assert res["watermarks"] == {
        "aqs": None, "weather": None, "hms": None, "firms": None,
        "speciation": None,
    }
    assert res["row_counts"] == {
        "aqs_daily": 0, "weather_daily": 0, "hms_smoke": 0,
        "firms_hotspots": 0, "speciation": 0, "labels": 0, "predictions": 0,
    }
    assert res["aqs_date_bounds"] is None


def test_run_cli_bbox_scopes_samples(tmp_path):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path) as store:
        store.insert_aqs_daily([
            _aqs_record("06-037-0002", "2023-07-01", 40.0, -120.0),
            _aqs_record("41-003-0001", "2023-07-01", 45.5, -122.6),
        ])

    mock_run = Mock()
    mock_run.return_value = {
        "metrics": {"confusion": {}, "total": 0, "notes": []},
        "samples": 0,
    }
    with patch("backend.eval.accuracy.__main__.run_accuracy_eval", mock_run):
        rc = main([
            "run",
            "--start", "2023-07-01", "--end", "2023-07-01",
            "--bbox=-121,39,-119,41",
            "--db", str(db_path),
        ])
    assert rc == 0
    assert mock_run.call_count == 1
    # Only the site inside the bbox is whitelisted for evaluation.
    assert mock_run.call_args.kwargs["site_ids"] == ["06-037-0002"]


# ---------------------------------------------------------------------------
# ingest hms --incremental (rolling-mode watermark advance)
# ---------------------------------------------------------------------------


def test_ingest_all_runs_every_source_and_sets_watermarks(tmp_path):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path) as store:
        # AQS data for the weather site derivation.
        store.insert_aqs_daily([_aqs_record("06-037-0002", "2023-07-01", 40.0, -120.0)])

    with patch("backend.eval.accuracy.__main__.ingest_aqs_year", Mock(return_value=5)) as aqs, \
         patch("backend.eval.accuracy.__main__.ingest_weather_for_sites", Mock(return_value=2)) as weather, \
         patch("backend.eval.accuracy.__main__.ingest_hms_smoke_range", Mock(return_value=3)) as hms, \
         patch("backend.eval.accuracy.__main__.ingest_firms_historical", Mock(return_value=4)) as firms, \
         patch("backend.eval.accuracy.__main__.RAW_DATA_DIR", tmp_path / "raw"):
        rc = main([
            "ingest", "all",
            "--start", "2023-07-01", "--end", "2023-07-03",
            "--year", "2020",
            "--bbox=-125,24,-66,50",
            "--db", str(db_path),
        ])

    assert rc == 0
    assert aqs.call_count == 1 and weather.call_count == 1
    assert hms.call_count == 1 and firms.call_count == 1
    # bbox flows to weather (site filter) and firms (bbox), not hms.
    assert weather.call_args[0][0] == [("06-037-0002", 40.0, -120.0)]
    assert firms.call_args[0][2] == (-125.0, 24.0, -66.0, 50.0)

    with AccuracyStore(db_path) as store:
        assert store.get_watermark("aqs") == "2020-12-31"
        assert store.get_watermark("weather") == "2023-07-03"
        assert store.get_watermark("hms") == "2023-07-03"
        assert store.get_watermark("firms") == "2023-07-03"


def test_ingest_weather_speciation_backfills_speciation_sites(tmp_path, capsys):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path) as store:
        store.insert_speciation_sites([
            ("06-037-1003", 33.9372, -118.1919),
            ("35-013-0002", 36.2664, -115.2201),
        ])

    mock_ingest = Mock(return_value=366)
    with patch("backend.eval.accuracy.__main__.ingest_weather_for_sites", mock_ingest):
        rc = main([
            "ingest", "weather", "--speciation",
            "--start", "2020-06-01", "--end", "2020-09-30",
            "--db", str(db_path),
        ])

    assert rc == 0
    # The site list comes from speciation_sites (IMPROVE sites), not aqs_daily.
    assert mock_ingest.call_args[0][0] == [
        ("06-037-1003", 33.9372, -118.1919),
        ("35-013-0002", 36.2664, -115.2201),
    ]
    assert mock_ingest.call_args[0][1] == "2020-06-01"
    assert mock_ingest.call_args[0][2] == "2020-09-30"
    # The AQS-site weather watermark must not be advanced by a speciation-only
    # backfill (a later incremental AQS-site run would otherwise skip the range).
    with AccuracyStore(db_path) as store:
        assert store.get_watermark("weather") is None
    assert "speciation sites" in capsys.readouterr().out


def test_ingest_weather_speciation_rejects_incremental(tmp_path):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path):
        pass
    rc = main([
        "ingest", "weather", "--speciation", "--incremental", "--db", str(db_path),
    ])
    assert rc == 2


def test_ingest_weather_speciation_requires_sites_and_dates(tmp_path, capsys):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path):
        pass

    # No speciation sites ingested yet.
    rc = main([
        "ingest", "weather", "--speciation",
        "--start", "2020-06-01", "--end", "2020-09-30",
        "--db", str(db_path),
    ])
    assert rc == 1
    assert "no speciation sites" in capsys.readouterr().out

    # --start/--end are mandatory for the speciation backfill.
    with AccuracyStore(db_path) as store:
        store.insert_speciation_sites([("06-037-1003", 33.9372, -118.1919)])
    rc = main(["ingest", "weather", "--speciation", "--db", str(db_path)])
    assert rc == 2
    assert "requires --start and --end" in capsys.readouterr().out


def test_ingest_hms_incremental_advances_watermark(tmp_path):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path) as store:
        store.set_watermark("hms", "2020-01-05")

    mock_ingest = Mock(return_value=7)
    with patch("backend.eval.accuracy.__main__.ingest_hms_smoke_range", mock_ingest), \
         patch("backend.eval.accuracy.__main__.RAW_DATA_DIR", tmp_path / "raw"):
        rc = main(["ingest", "hms", "--incremental", "--db", str(db_path)])

    assert rc == 0
    # The rolling window runs from the watermark through today, chunked by
    # year so downloads land under raw/hms/<year>/.
    assert mock_ingest.call_args_list[0].args[0] == "2020-01-05"
    assert mock_ingest.call_args_list[-1].args[1] == date.today().isoformat()
    assert mock_ingest.call_args_list[0].args[3] == tmp_path / "raw" / "hms" / "2020"
    assert mock_ingest.call_count == date.today().year - 2020 + 1

    with AccuracyStore(db_path) as store:
        assert store.get_watermark("hms") == date.today().isoformat()


def test_ingest_hms_incremental_first_run_uses_30_day_lookback(tmp_path):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path):
        pass  # no watermark yet

    mock_ingest = Mock(return_value=0)
    with patch("backend.eval.accuracy.__main__.ingest_hms_smoke_range", mock_ingest), \
         patch("backend.eval.accuracy.__main__.RAW_DATA_DIR", tmp_path / "raw"):
        rc = main(["ingest", "hms", "--incremental", "--db", str(db_path)])

    assert rc == 0
    expected_start = (date.today() - timedelta(days=30)).isoformat()
    assert mock_ingest.call_args_list[0].args[0] == expected_start
    with AccuracyStore(db_path) as store:
        assert store.get_watermark("hms") == date.today().isoformat()


def test_ingest_hms_incremental_does_not_advance_on_failure(tmp_path):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path) as store:
        store.set_watermark("hms", "2020-01-05")

    mock_ingest = Mock(side_effect=RuntimeError("archive unreachable"))
    with patch("backend.eval.accuracy.__main__.ingest_hms_smoke_range", mock_ingest), \
         patch("backend.eval.accuracy.__main__.RAW_DATA_DIR", tmp_path / "raw"):
        rc = main(["ingest", "hms", "--incremental", "--db", str(db_path)])

    # Every year chunk failed, so the command exits non-zero...
    assert rc == 1
    # ...and the watermark stays put so the next run retries the tail.
    with AccuracyStore(db_path) as store:
        assert store.get_watermark("hms") == "2020-01-05"
