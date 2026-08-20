"""Tests for the accuracy-eval weather + FIRMS-historical ingest adapters.

All HTTP is mocked at the ``httpx.Client`` level (sync client), mirroring the
``_mock_client`` pattern used across the test suite. No network access here.
"""

import sqlite3
import threading
from unittest.mock import Mock, patch

import httpx

from backend.eval.accuracy.ingest.firms_historical import (
    _date_windows,
    fetch_firms_historical,
    ingest_firms_historical,
)
from backend.eval.accuracy.ingest.weather import (
    WEATHER_ARCHIVE_URL,
    fetch_weather_daily,
    ingest_weather_for_sites,
)
from backend.eval.accuracy.records import (
    FirmsHotspotRecord,
    TransportWindRecord,
    WeatherDailyRecord,
)
from backend.eval.accuracy.store import AccuracyStore

# --------------------------------------------------------------------------
# Shared mock helpers (sync httpx.Client replacement)
# --------------------------------------------------------------------------


def _mock_http_response(status=200, json=None, text=""):
    resp = Mock()
    resp.status_code = status
    if 200 <= status < 300:
        resp.raise_for_status = Mock()
    else:
        # Mirror httpx: raise_for_status() raises HTTPStatusError on 4xx/5xx.
        def _raise():
            raise httpx.HTTPStatusError(
                f"HTTP {status}",
                request=httpx.Request("GET", "http://test"),
                response=resp,
            )

        resp.raise_for_status = Mock(side_effect=_raise)
    resp.json.return_value = json if json is not None else {}
    resp.text = text
    return resp


def _make_sync_client(handler):
    """Sync ``httpx.Client`` stand-in: ``handler(url, params, kwargs)`` -> resp."""
    client = Mock()
    client.get = Mock(side_effect=lambda url, params=None, **kwargs: handler(url, params))
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    return client


# --------------------------------------------------------------------------
# Weather (Open-Meteo archive)
# --------------------------------------------------------------------------

# Two days; the second day has a null wind-speed value (kept as None).
WEATHER_FIXTURE_JSON = {
    "latitude": 34.0522,
    "longitude": -118.2437,
    "daily_units": {
        "time": "iso8601",
        "temperature_2m_max": "°F",
        "temperature_2m_min": "°F",
        "wind_speed_10m_max": "mph",
        "wind_direction_10m_dominant": "°",
        "precipitation_sum": "mm",
        "wind_gusts_10m_max": "mph",
    },
    "daily": {
        "time": ["2023-07-01", "2023-07-02"],
        "temperature_2m_max": [95.4, 92.1],
        "temperature_2m_min": [66.2, 64.8],
        "wind_speed_10m_max": [18.7, None],
        "wind_direction_10m_dominant": [245, 230],
        "precipitation_sum": [0.25, None],
        "wind_gusts_10m_max": [32.0, 28.4],
    },
}


def _fetch_weather(site_id=None):
    calls = []

    def handler(url, params):
        calls.append(params)
        return _mock_http_response(json=WEATHER_FIXTURE_JSON)

    client = _make_sync_client(handler)
    with patch("backend.eval.accuracy.ingest.weather.httpx.Client", return_value=client):
        records = fetch_weather_daily(34.0522, -118.2437, "2023-07-01", "2023-07-02", site_id=site_id)
    return records, calls


def test_fetch_weather_daily_parses_records_and_params():
    records, calls = _fetch_weather()

    assert len(records) == 2
    first, second = records
    assert isinstance(first, WeatherDailyRecord)
    # Coordinate-derived site id when none is supplied.
    assert first.site_id == "34.0522,-118.2437"
    assert first.lat == 34.0522
    assert first.lon == -118.2437
    assert first.date_local == "2023-07-01"
    assert first.tmax_f == 95.4
    assert first.tmin_f == 66.2
    assert first.wind_max_mph == 18.7
    assert first.wind_dir_dominant_deg == 245
    assert first.precipitation_mm == 0.25
    assert first.wind_gust_max_mph == 32.0
    assert first.natural_key == ("34.0522,-118.2437", "2023-07-01")

    # Null wind + null precip on day two stay None while the rest of the day
    # (including gust) is kept.
    assert second.date_local == "2023-07-02"
    assert second.tmax_f == 92.1
    assert second.wind_max_mph is None
    assert second.wind_dir_dominant_deg == 230
    assert second.precipitation_mm is None
    assert second.wind_gust_max_mph == 28.4

    # Request params match the Open-Meteo archive contract.
    params = calls[0]
    assert params["latitude"] == 34.0522
    assert params["longitude"] == -118.2437
    assert params["start_date"] == "2023-07-01"
    assert params["end_date"] == "2023-07-02"
    assert params["daily"] == (
        "temperature_2m_max,temperature_2m_min,"
        "wind_speed_10m_max,wind_direction_10m_dominant,"
        "precipitation_sum,wind_gusts_10m_max"
    )
    assert params["temperature_unit"] == "fahrenheit"
    assert params["wind_speed_unit"] == "mph"
    assert params["timezone"] == "GMT"


def test_fetch_weather_skips_day_with_all_null_values():
    payload = {
        "daily": {
            "time": ["2023-07-01", "2023-07-02", "2023-07-03"],
            "temperature_2m_max": [95.4, 88.0, None],
            "temperature_2m_min": [66.2, 60.0, None],
            "wind_speed_10m_max": [18.7, None, None],
            "wind_direction_10m_dominant": [245, None, None],
        }
    }

    def handler(url, params):
        return _mock_http_response(json=payload)

    client = _make_sync_client(handler)
    with patch("backend.eval.accuracy.ingest.weather.httpx.Client", return_value=client):
        records = fetch_weather_daily(34.05, -118.24, "2023-07-01", "2023-07-03")

    # Day three is all-null and dropped; day two keeps its temperature values.
    assert [r.date_local for r in records] == ["2023-07-01", "2023-07-02"]
    assert records[1].tmax_f == 88.0
    assert records[1].wind_max_mph is None


def test_fetch_weather_daily_retries_on_429_then_succeeds():
    request = httpx.Request("GET", WEATHER_ARCHIVE_URL)
    rate_limited = httpx.HTTPStatusError(
        "rate limited",
        request=request,
        response=httpx.Response(429, request=request),
    )
    client = Mock()
    # Calls 1-2 raise HTTPStatusError (429); call 3 returns valid JSON.
    client.get = Mock(side_effect=[
        rate_limited,
        rate_limited,
        _mock_http_response(json=WEATHER_FIXTURE_JSON),
    ])
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)

    with patch("backend.eval.accuracy.ingest.weather.httpx.Client", return_value=client), \
         patch("backend.eval.accuracy.ingest.weather.time.sleep") as sleep:
        records = fetch_weather_daily(34.0522, -118.2437, "2023-07-01", "2023-07-02")

    # Both 429s were retried (with backoff) and the third attempt succeeded.
    assert len(records) == 2
    assert client.get.call_count == 3
    assert sleep.call_count == 2


def test_fetch_weather_daily_returns_empty_on_non_retryable_4xx():
    request = httpx.Request("GET", WEATHER_ARCHIVE_URL)
    client = Mock()
    client.get = Mock(side_effect=httpx.HTTPStatusError(
        "bad request",
        request=request,
        response=httpx.Response(400, request=request),
    ))
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)

    with patch("backend.eval.accuracy.ingest.weather.httpx.Client", return_value=client), \
         patch("backend.eval.accuracy.ingest.weather.time.sleep") as sleep:
        records = fetch_weather_daily(34.05, -118.24, "2023-07-01", "2023-07-02")

    # A 400 means "no data", not an error: no retries, no backoff sleep.
    assert records == []
    assert client.get.call_count == 1
    assert sleep.call_count == 0


def test_weather_store_roundtrip_and_idempotent(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        records, _ = _fetch_weather()
        assert store.insert_weather_daily(records) == 2
        assert len(store.fetch_weather_daily()) == 2

        # Re-inserting replaces in place: still one row per natural key.
        assert store.insert_weather_daily(records) == 2
        assert len(store.fetch_weather_daily()) == 2

        fetched = store.fetch_weather_daily()
        by_key = {r.natural_key: r for r in fetched}
        for rec in records:
            assert by_key[rec.natural_key] == rec

        # Site filter narrows to one site.
        assert len(store.fetch_weather_daily(site_id="34.0522,-118.2437")) == 2
        assert store.fetch_weather_daily(site_id="nope") == []

        # Direct SQL check: exactly one row per natural key.
        conn = sqlite3.connect(store.db_path)
        dupes = conn.execute(
            "SELECT site_id, date_local, COUNT(*) FROM weather_daily "
            "GROUP BY 1, 2 HAVING COUNT(*) > 1"
        ).fetchall()
        conn.close()
        assert dupes == []
    finally:
        store.close()


def _transport_wind_records(site_id):
    """Two deterministic 850 hPa days for ``site_id``; the second day has a
    missing v850 (kept as None, mirroring a partial NCSS response)."""
    return [
        TransportWindRecord(
            site_id=site_id,
            date_local="2023-07-01",
            u850=4.2,
            v850=-3.1,
            source="ncep_daily",
        ),
        TransportWindRecord(
            site_id=site_id,
            date_local="2023-07-02",
            u850=2.0,
            v850=None,
            source="ncep_daily",
        ),
    ]


def test_transport_wind_store_roundtrip_and_idempotent(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        records = _transport_wind_records("SITE_A")
        assert store.insert_transport_wind(records) == 2
        assert store.count_transport_wind() == 2

        # Re-inserting replaces in place: still one row per natural key.
        assert store.insert_transport_wind(records) == 2
        assert store.count_transport_wind() == 2

        fetched = store.fetch_transport_wind()
        by_key = {r.natural_key: r for r in fetched}
        for rec in records:
            assert by_key[rec.natural_key] == rec

        # The partial vector round-trips: v850 stays None, source is kept.
        partial = by_key[("SITE_A", "2023-07-02")]
        assert partial.u850 == 2.0
        assert partial.v850 is None
        assert partial.source == "ncep_daily"

        # Site filter narrows to one site.
        assert len(store.fetch_transport_wind(site_id="SITE_A")) == 2
        assert store.fetch_transport_wind(site_id="nope") == []

        # Direct SQL check: exactly one row per natural key.
        conn = sqlite3.connect(store.db_path)
        dupes = conn.execute(
            "SELECT site_id, date_local, COUNT(*) FROM transport_wind "
            "GROUP BY 1, 2 HAVING COUNT(*) > 1"
        ).fetchall()
        conn.close()
        assert dupes == []
    finally:
        store.close()


def test_has_transport_wind_coverage(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        assert store.has_transport_wind_coverage("SITE_A", "2023-07-01", "2023-07-02") is False

        store.insert_transport_wind(_transport_wind_records("SITE_A"))
        # A window fully spanned by the stored rows counts as coverage.
        assert store.has_transport_wind_coverage("SITE_A", "2023-07-01", "2023-07-02") is True
        assert store.has_transport_wind_coverage("SITE_A", "2023-07-01", "2023-07-01") is True

        # A window only partially overlapped is NOT covered, so it gets
        # re-fetched (same semantics as has_weather_coverage).
        assert store.has_transport_wind_coverage("SITE_A", "2023-06-01", "2023-07-01") is False
        assert store.has_transport_wind_coverage("SITE_A", "2023-07-02", "2023-07-31") is False
        assert store.has_transport_wind_coverage("SITE_B", "2023-07-01", "2023-07-02") is False
    finally:
        store.close()


def _weather_records(site_id):
    """Two deterministic days for ``site_id`` (2023-07-01/02), mirroring the
    shape of ``WEATHER_FIXTURE_JSON`` parsed by ``fetch_weather_daily``."""
    return [
        WeatherDailyRecord(
            site_id=site_id,
            lat=34.05,
            lon=-118.24,
            date_local=date_local,
            tmax_f=90.0 + idx,
            tmin_f=60.0,
            wind_max_mph=10.0,
            wind_dir_dominant_deg=200,
            precipitation_mm=0.0,
            wind_gust_max_mph=20.0,
        )
        for idx, date_local in enumerate(("2023-07-01", "2023-07-02"))
    ]


def test_ingest_weather_for_sites_throttles_before_each_fetch(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    requested_lats = []
    lock = threading.Lock()

    def handler(url, params):
        with lock:
            requested_lats.append(params["latitude"])
        return _mock_http_response(json=WEATHER_FIXTURE_JSON)

    client = _make_sync_client(handler)
    with patch("backend.eval.accuracy.ingest.weather.httpx.Client", return_value=client), \
         patch("backend.eval.accuracy.ingest.weather.time.sleep") as sleep:
        total = ingest_weather_for_sites(
            [("SITE_A", 34.05, -118.24), ("SITE_B", 45.52, -122.68)],
            "2023-07-01",
            "2023-07-02",
            store,
        )

    assert total == 4  # 2 sites x 2 days
    assert sorted(requested_lats) == [34.05, 45.52]
    # Each worker sleeps once, before its own request.
    assert sleep.call_count == 2
    assert {r.site_id for r in store.fetch_weather_daily()} == {"SITE_A", "SITE_B"}


def test_has_weather_coverage(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        assert store.has_weather_coverage("SITE_A", "2023-07-01", "2023-07-02") is False

        store.insert_weather_daily(_weather_records("SITE_A"))
        # A window fully spanned by the stored rows counts as coverage.
        assert store.has_weather_coverage("SITE_A", "2023-07-01", "2023-07-02") is True
        assert store.has_weather_coverage("SITE_A", "2023-07-01", "2023-07-01") is True

        # A window only partially overlapped (rows inside it but not spanning
        # it end-to-end) is NOT covered, so it gets re-fetched.
        assert store.has_weather_coverage("SITE_A", "2023-06-01", "2023-07-01") is False
        assert store.has_weather_coverage("SITE_A", "2023-07-02", "2023-07-31") is False

        # Non-overlapping windows or other sites report False.
        assert store.has_weather_coverage("SITE_A", "2023-08-01", "2023-08-31") is False
        assert store.has_weather_coverage("SITE_B", "2023-07-01", "2023-07-02") is False
    finally:
        store.close()


def test_has_weather_coverage_requires_precip_and_gust(tmp_path):
    """Coverage now means the window is spanned AND its rows carry precip +
    gust. Rows written before the Track-B columns existed (NULL precip/gust)
    must NOT count as covered so the backfill re-fetches them."""
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        # A full-span window whose rows lack precip/gust is NOT covered.
        store.insert_weather_daily(_weather_records("SITE_A"))
        conn = sqlite3.connect(store.db_path)
        conn.execute(
            "UPDATE weather_daily SET precipitation_mm = NULL, "
            "wind_gust_max_mph = NULL"
        )
        conn.commit()
        conn.close()
        assert store.has_weather_coverage("SITE_A", "2023-07-01", "2023-07-02") is False
        assert store.has_weather_coverage("SITE_A", "2023-07-01", "2023-07-01") is False

        # A partially-backfilled window (one day still NULL) is NOT covered.
        conn = sqlite3.connect(store.db_path)
        conn.execute(
            "UPDATE weather_daily SET precipitation_mm = 0.0, "
            "wind_gust_max_mph = 20.0 WHERE date_local = '2023-07-01'"
        )
        conn.commit()
        conn.close()
        assert store.has_weather_coverage("SITE_A", "2023-07-01", "2023-07-02") is False

        # Once every window day carries precip + gust, coverage is True.
        conn = sqlite3.connect(store.db_path)
        conn.execute(
            "UPDATE weather_daily SET precipitation_mm = 0.0, "
            "wind_gust_max_mph = 20.0 WHERE date_local = '2023-07-02'"
        )
        conn.commit()
        conn.close()
        assert store.has_weather_coverage("SITE_A", "2023-07-01", "2023-07-02") is True
        assert store.has_weather_coverage("SITE_A", "2023-07-01", "2023-07-01") is True
    finally:
        store.close()


def test_weather_daily_migrates_pre_existing_table(tmp_path):
    """A store created before the precip/gust columns existed must gain the
    columns idempotently (ALTER TABLE guarded by PRAGMA) WITHOUT losing rows."""
    db_path = tmp_path / "accuracy.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE weather_daily ("
        "    site_id TEXT NOT NULL, lat REAL, lon REAL, date_local TEXT NOT NULL,"
        "    tmax_f REAL, tmin_f REAL, wind_max_mph REAL,"
        "    wind_dir_dominant_deg INTEGER,"
        "    ingested_at TEXT NOT NULL DEFAULT (datetime('now')),"
        "    PRIMARY KEY (site_id, date_local)"
        ")"
    )
    conn.execute(
        "INSERT INTO weather_daily (site_id, lat, lon, date_local, tmax_f, "
        "tmin_f, wind_max_mph, wind_dir_dominant_deg) "
        "VALUES ('SITE_A', 34.05, -118.24, '2023-07-01', 90.0, 60.0, 10.0, 200)"
    )
    conn.commit()
    conn.close()

    store = AccuracyStore(db_path)
    try:
        # The pre-existing row survived the migration and now reads back with
        # NULL (not-yet-backfilled) precip/gust fields.
        rows = store.fetch_weather_daily(site_id="SITE_A")
        assert len(rows) == 1
        row = rows[0]
        assert row.precipitation_mm is None
        assert row.wind_gust_max_mph is None

        # Re-opening the same store is a no-op (columns already exist).
        store.close()
        store = AccuracyStore(db_path)
        assert len(store.fetch_weather_daily(site_id="SITE_A")) == 1

        # INSERT now writes the enriched columns for the migrated table.
        assert store.insert_weather_daily(_weather_records("SITE_A")) == 2
        fetched = {r.date_local: r for r in store.fetch_weather_daily(site_id="SITE_A")}
        assert fetched["2023-07-01"].precipitation_mm == 0.0
        assert fetched["2023-07-01"].wind_gust_max_mph == 20.0
    finally:
        store.close()


def test_ingest_weather_for_sites_skips_existing(tmp_path, capsys):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        store.insert_weather_daily(_weather_records("SITE_A"))
        called = []

        def fake_fetch(lat, lon, start_date, end_date, site_id=None):
            called.append(site_id)
            if site_id == "SITE_A":
                raise AssertionError("covered site must not be re-fetched")
            return _weather_records(site_id)

        with patch(
            "backend.eval.accuracy.ingest.weather.fetch_weather_daily",
            side_effect=fake_fetch,
        ), patch("backend.eval.accuracy.ingest.weather.time.sleep"):
            total = ingest_weather_for_sites(
                [("SITE_A", 34.05, -118.24), ("SITE_B", 45.52, -122.68)],
                "2023-07-01",
                "2023-07-02",
                store,
            )

        # Only the uncovered site is fetched; the covered one is skipped.
        assert total == 2  # SITE_B's two days
        assert called == ["SITE_B"]
        assert "skip SITE_A" in capsys.readouterr().err
        assert {r.site_id for r in store.fetch_weather_daily()} == {"SITE_A", "SITE_B"}
    finally:
        store.close()


def test_ingest_weather_for_sites_continues_after_site_failure(tmp_path, capsys):
    store = AccuracyStore(tmp_path / "accuracy.db")
    failure = httpx.HTTPStatusError(
        "rate limited",
        request=httpx.Request("GET", WEATHER_ARCHIVE_URL),
        response=httpx.Response(429, request=httpx.Request("GET", WEATHER_ARCHIVE_URL)),
    )
    try:
        def fake_fetch(lat, lon, start_date, end_date, site_id=None):
            if site_id == "SITE_A":
                raise failure
            return _weather_records(site_id)

        with patch(
            "backend.eval.accuracy.ingest.weather.fetch_weather_daily",
            side_effect=fake_fetch,
        ), patch("backend.eval.accuracy.ingest.weather.time.sleep"):
            total = ingest_weather_for_sites(
                [("SITE_A", 34.05, -118.24), ("SITE_B", 45.52, -122.68)],
                "2023-07-01",
                "2023-07-02",
                store,
            )

        # SITE_A keeps failing but the batch completes with SITE_B's data.
        assert total == 2  # SITE_B's two days
        assert {r.site_id for r in store.fetch_weather_daily()} == {"SITE_B"}
        captured = capsys.readouterr()
        assert "FAILED SITE_A" in captured.err
        assert "failed: ['SITE_A']" in captured.out
    finally:
        store.close()


def test_ingest_weather_for_sites_concurrent_workers(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    call_log = []
    lock = threading.Lock()
    # All ``workers`` threads must rendezvous here before any fetch returns,
    # proving the fetches actually run concurrently (and not just submit).
    barrier = threading.Barrier(4)

    def fake_fetch(lat, lon, start_date, end_date, site_id=None):
        with lock:
            call_log.append(site_id)
        barrier.wait(timeout=5)
        return _weather_records(site_id)

    sites = [(f"SITE_{i:02d}", float(i), -float(i)) for i in range(8)]
    with patch(
        "backend.eval.accuracy.ingest.weather.fetch_weather_daily",
        side_effect=fake_fetch,
    ), patch("backend.eval.accuracy.ingest.weather.time.sleep"):
        total = ingest_weather_for_sites(
            sites, "2023-07-01", "2023-07-02", store, workers=4
        )

    assert total == 16  # 8 sites x 2 days
    assert sorted(call_log) == [s[0] for s in sites]
    assert store.count_weather_daily() == 16

    # No duplicates under the natural key (site_id, date_local).
    conn = sqlite3.connect(store.db_path)
    dupes = conn.execute(
        "SELECT site_id, date_local, COUNT(*) FROM weather_daily "
        "GROUP BY 1, 2 HAVING COUNT(*) > 1"
    ).fetchall()
    conn.close()
    assert dupes == []
    store.close()


def test_ingest_weather_for_sites_deterministic(tmp_path):
    sites = [
        ("SITE_A", 34.05, -118.24),
        ("SITE_B", 45.52, -122.68),
        ("SITE_C", 40.71, -74.00),
        ("SITE_D", 47.61, -122.33),
    ]

    def run(db_path, workers):
        store = AccuracyStore(db_path)
        try:
            with patch(
                "backend.eval.accuracy.ingest.weather.fetch_weather_daily",
                side_effect=lambda lat, lon, start_date, end_date, site_id=None: _weather_records(site_id),
            ), patch("backend.eval.accuracy.ingest.weather.time.sleep"):
                ingest_weather_for_sites(
                    sites, "2023-07-01", "2023-07-02", store, workers=workers
                )
            return store.fetch_weather_daily()
        finally:
            store.close()

    # Sequential and concurrent runs with the same inputs must produce the
    # same stored rows (ordered by site_id, date_local).
    sequential = run(tmp_path / "run-a.db", workers=1)
    concurrent = run(tmp_path / "run-b.db", workers=4)
    assert concurrent == sequential
    assert [(r.site_id, r.date_local) for r in sequential] == [
        (site_id, date_local)
        for site_id, _, _ in sites
        for date_local in ("2023-07-01", "2023-07-02")
    ]


# --------------------------------------------------------------------------
# FIRMS historical (area API)
# --------------------------------------------------------------------------

FIRMS_HEADER = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight"
)

# 3 valid rows + 1 row with an unparseable acq_time (skipped).
FIRMS_CSV = f"""{FIRMS_HEADER}
43.6719,-119.1263,297.1,0.6,0.4,2023-07-01,940,N,VIIRS,h,2.0URT,287.1,1.86,N
43.5800,-119.0500,300.0,0.6,0.4,2023-07-01,1000,N,VIIRS,n,2.0URT,290.0,5.20,D
43.5000,-119.3000,299.0,0.6,0.4,2023-07-02,200,NN,VIIRS,l,2.0URT,290.0,3.00,N
43.4000,-119.4000,299.0,0.6,0.4,2023-07-03,xxxx,NN,VIIRS,l,2.0URT,290.0,3.00,N
"""

BBOX = (-120.0, 43.0, -118.0, 44.0)


def _fetch_firms(start_date, end_date, csv_text, map_key="test-key"):
    calls = []

    def handler(url, params):
        calls.append(str(url))
        # Serve the fixture to the first source only; the other SP sources
        # respond with a header-only CSV (zero rows) so per-source requests
        # don't duplicate the fixture's rows.
        if "/VIIRS_SNPP_SP/" in url:
            return _mock_http_response(text=csv_text)
        return _mock_http_response(text=f"{FIRMS_HEADER}\n")

    client = _make_sync_client(handler)
    with patch("backend.eval.accuracy.ingest.firms_historical.httpx.Client", return_value=client), \
         patch("backend.eval.accuracy.ingest.firms_historical.FIRMS_MAP_KEY", map_key):
        records = fetch_firms_historical(start_date, end_date, BBOX)
    return records, calls


def test_date_windows_chunks_at_five_days():
    assert _date_windows("2023-07-01", "2023-07-01") == [("2023-07-01", "2023-07-01")]
    assert _date_windows("2023-07-01", "2023-07-05") == [("2023-07-01", "2023-07-05")]
    assert _date_windows("2023-07-01", "2023-07-06") == [
        ("2023-07-01", "2023-07-05"),
        ("2023-07-06", "2023-07-06"),
    ]
    assert _date_windows("2023-07-01", "2023-07-08") == [
        ("2023-07-01", "2023-07-05"),
        ("2023-07-06", "2023-07-08"),
    ]


def test_fetch_firms_historical_parses_records_and_url():
    records, calls = _fetch_firms("2023-07-01", "2023-07-03", FIRMS_CSV)

    # A 3-day window fits in one 5-day chunk, but each SP source is requested
    # separately (FIRMS rejects comma-joined source lists with HTTP 400).
    assert len(calls) == 2
    assert {u.split("csv/")[1].split("/")[1] for u in calls} == {
        "VIIRS_SNPP_SP", "VIIRS_NOAA20_SP",
    }
    for url in calls:
        # Single-source URLs, never a comma-joined source list.
        assert "," not in url.split("csv/")[1].split("/")[1]
        assert url.endswith("/-120.0,43.0,-118.0,44.0/3/2023-07-01")
    # The NRT product names (FIRMS_SOURCES, used by the LIVE path) are no
    # longer used by the historical path.
    assert all("_NRT" not in u for u in calls)

    # The malformed-acq_time row is skipped; 3 valid records remain.
    assert len(records) == 3
    first = records[0]
    assert isinstance(first, FirmsHotspotRecord)
    assert first.lat == 43.6719
    assert first.lon == -119.1263
    assert first.frp == 1.86
    assert first.acq_datetime == "2023-07-01T09:40:00+00:00"
    assert first.confidence == "high"
    assert first.satellite == "N"
    assert first.daynight == "N"
    assert first.natural_key == (
        43.6719, -119.1263, "2023-07-01T09:40:00+00:00", "N",
    )

    assert records[1].acq_datetime == "2023-07-01T10:00:00+00:00"
    assert records[1].confidence == "nominal"
    assert records[1].daynight == "D"
    # "200" -> 02:00 UTC.
    assert records[2].acq_datetime == "2023-07-02T02:00:00+00:00"
    assert records[2].confidence == "low"
    assert records[2].satellite == "NN"


def test_fetch_firms_historical_chunks_windows_over_5_days():
    # Header-only CSV: verify the URL loop, not the parsing.
    records, calls = _fetch_firms("2023-07-01", "2023-07-08", f"{FIRMS_HEADER}\n")

    assert records == []
    assert len(calls) == 4  # 2 SP sources x 2 windows
    assert all(u.endswith("/5/2023-07-01") for u in calls[:2])
    assert all(u.endswith("/3/2023-07-06") for u in calls[2:])


def test_fetch_firms_historical_without_key_returns_empty():
    with patch("backend.eval.accuracy.ingest.firms_historical.FIRMS_MAP_KEY", ""), \
         patch("backend.eval.accuracy.ingest.firms_historical.httpx.Client") as client_cls:
        assert fetch_firms_historical("2023-07-01", "2023-07-01", BBOX) == []
    client_cls.assert_not_called()


def test_firms_store_roundtrip_and_idempotent(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        records, _ = _fetch_firms("2023-07-01", "2023-07-03", FIRMS_CSV)
        assert store.insert_firms_hotspots(records) == 3
        assert store.count_firms_hotspots() == 3

        # Re-inserting replaces in place: still one row per natural key.
        assert store.insert_firms_hotspots(records) == 3
        assert store.count_firms_hotspots() == 3

        fetched = store.fetch_firms_hotspots()
        assert len(fetched) == 3
        by_key = {r.natural_key: r for r in fetched}
        for rec in records:
            assert by_key[rec.natural_key] == rec

        # Direct SQL check: exactly one row per natural key.
        conn = sqlite3.connect(store.db_path)
        dupes = conn.execute(
            "SELECT lat, lon, acq_datetime, satellite, COUNT(*) FROM firms_hotspot "
            "GROUP BY 1, 2, 3, 4 HAVING COUNT(*) > 1"
        ).fetchall()
        conn.close()
        assert dupes == []
    finally:
        store.close()


def test_firms_store_full_precision_keys_no_collision(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        rec = FirmsHotspotRecord(
            lat=43.67193, lon=-119.12631, frp=1.86,
            acq_datetime="2023-07-01T09:40:00+00:00",
            confidence="high", satellite="N", daynight="N",
        )
        assert store.insert_firms_hotspots([rec]) == 1
        fetched = store.fetch_firms_hotspots()[0]
        # Lat/lon are stored at FULL float precision (no 4-decimal rounding).
        assert fetched.lat == 43.67193
        assert fetched.lon == -119.12631
        assert store.count_firms_hotspots() == 1

        # A second detection whose lat differs only in the 5th decimal must be
        # a distinct row, not silently merged into the first.
        rec_adjacent = FirmsHotspotRecord(
            lat=43.67194, lon=-119.12631, frp=2.0,
            acq_datetime="2023-07-01T09:40:00+00:00",
            confidence="high", satellite="N", daynight="N",
        )
        assert store.insert_firms_hotspots([rec_adjacent]) == 1
        assert store.count_firms_hotspots() == 2
        lats = sorted(r.lat for r in store.fetch_firms_hotspots())
        assert lats == [43.67193, 43.67194]
    finally:
        store.close()


def test_ingest_firms_historical_writes_and_uses_config_key(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    calls = []

    def handler(url, params):
        calls.append(str(url))
        if "/VIIRS_SNPP_SP/" in url:
            return _mock_http_response(text=FIRMS_CSV)
        return _mock_http_response(text=f"{FIRMS_HEADER}\n")

    client = _make_sync_client(handler)
    with patch("backend.eval.accuracy.ingest.firms_historical.httpx.Client", return_value=client), \
         patch("backend.eval.accuracy.ingest.firms_historical.FIRMS_MAP_KEY", "cfg-key"):
        count = ingest_firms_historical("2023-07-01", "2023-07-01", BBOX, store)

    assert count == 3
    assert len(calls) == 2
    assert all("cfg-key" in c for c in calls)
    assert store.count_firms_hotspots() == 3


def test_fetch_firms_historical_merges_sources_and_tolerates_400():
    """Each source is requested separately; a 400 on one source is treated as
    'no data for that source' while the other sources' rows still merge in."""
    calls = []

    def handler(url, params):
        calls.append(str(url))
        if "/VIIRS_SNPP_SP/" in url:
            return _mock_http_response(text=FIRMS_CSV)  # 3 valid rows
        if "/VIIRS_NOAA20_SP/" in url:
            return _mock_http_response(status=400, text="Invalid source")
        raise AssertionError(f"unexpected source URL: {url}")

    client = _make_sync_client(handler)
    with patch("backend.eval.accuracy.ingest.firms_historical.httpx.Client", return_value=client), \
         patch("backend.eval.accuracy.ingest.firms_historical.FIRMS_MAP_KEY", "test-key"):
        records = fetch_firms_historical("2023-07-01", "2023-07-03", BBOX)

    # One request per source, each a single-source SP URL.
    assert len(calls) == 2
    assert {u.split("csv/")[1].split("/")[1] for u in calls} == {
        "VIIRS_SNPP_SP", "VIIRS_NOAA20_SP",
    }
    assert all("," not in u.split("csv/")[1].split("/")[1] for u in calls)
    # The NRT product names (FIRMS_SOURCES, used by the LIVE path) are no
    # longer used by the historical path.
    assert all("_NRT" not in u for u in calls)

    # NOAA-20's 400 does not abort the chunk: SNPP's 3 rows are merged in
    # (3 records total).
    assert len(records) == 3
    assert {round(r.lat, 4) for r in records} == {43.6719, 43.58, 43.5}
