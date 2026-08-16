"""Tests for the accuracy-eval AQS ingest adapter and store (Phase 1).

All fixtures are hand-written CSV text with the real EPA header; no network
access in these tests.
"""

import csv
import io
import sqlite3
import zipfile

import pytest

from backend.eval.accuracy.ingest.aqs import _extract_csvs, parse_aqs_daily_csv
from backend.eval.accuracy.records import AqsDailyRecord
from backend.eval.accuracy.store import AccuracyStore

# EPA daily summary header, in published order (29 columns).
_FIELDS = [
    "State Code", "County Code", "Site Num", "Parameter Code", "POC",
    "Latitude", "Longitude", "Datum", "Parameter Name", "Sample Duration",
    "Pollutant Standard", "Date Local", "Units of Measure", "Event Type",
    "Observation Count", "Observation Percent", "Arithmetic Mean",
    "1st Max Value", "1st Max Hour", "AQI", "Method Code", "Method Name",
    "Local Site Name", "Address", "State Name", "County Name", "City Name",
    "CBSA Name", "Date of Last Change",
]

_ROW_LA_PM25 = {
    "State Code": "6", "County Code": "37", "Site Num": "2",
    "Parameter Code": "88101", "POC": "1", "Latitude": "33.9372",
    "Longitude": "-118.1919", "Datum": "NAD83",
    "Parameter Name": "PM2.5 - Local Conditions", "Sample Duration": "24 HOUR",
    "Pollutant Standard": "PM25 24-hour 2012", "Date Local": "2023-07-01",
    "Units of Measure": "ug/m3 LC", "Event Type": "None",
    "Observation Count": "24", "Observation Percent": "100.0",
    "Arithmetic Mean": "15.5", "1st Max Value": "24.0", "1st Max Hour": "12",
    "AQI": "42", "Method Code": "145", "Method Name": "Met One SASS",
    "Local Site Name": "LA-North Main Street", "Address": "123 Main St, Suite 1",
    "State Name": "California", "County Name": "Los Angeles",
    "City Name": "Los Angeles", "CBSA Name": "Los Angeles-Long Beach-Anaheim, CA",
    "Date of Last Change": "2023-08-01",
}

_ROW_LA_PM25_DAY2 = {**_ROW_LA_PM25, "Date Local": "2023-07-02", "Arithmetic Mean": "35.2", "AQI": "95"}

_ROW_LA_OZONE = {**_ROW_LA_PM25, "Parameter Code": "44201",
                 "Parameter Name": "Ozone", "Date Local": "2023-07-01",
                 "Units of Measure": "ppm", "Arithmetic Mean": "0.055",
                 "AQI": "", "Method Code": "087"}

_ROW_OR_PM25 = {**_ROW_LA_PM25, "State Code": "41", "County Code": "3",
                "Site Num": "1", "POC": "2", "Latitude": "45.5181",
                "Longitude": "-122.6794", "Date Local": "2023-07-01",
                "Arithmetic Mean": "8.1", "AQI": "-999",
                "CBSA Name": "Portland-Vancouver-Hillsboro, OR-WA"}

# Skipped rows: missing concentration and missing coordinates respectively.
_ROW_NO_CONC = {**_ROW_LA_PM25, "Date Local": "2023-07-03", "Arithmetic Mean": ""}
_ROW_NO_COORDS = {**_ROW_LA_PM25, "Site Num": "5", "Latitude": "", "Longitude": ""}


def _build_csv(rows):
    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL)
    writer.writerow(_FIELDS)
    for row in rows:
        writer.writerow([row.get(field, "") for field in _FIELDS])
    return out.getvalue()


FIXTURE_CSV = _build_csv([
    _ROW_LA_PM25,
    _ROW_LA_PM25_DAY2,
    _ROW_LA_OZONE,
    _ROW_OR_PM25,
    _ROW_NO_CONC,
    _ROW_NO_COORDS,
])

# Only the first four rows carry usable coords + concentration.
EXPECTED_RECORD_COUNT = 4


def _drop_column(csv_text, col_name):
    """Rewrite CSV text without one named column (quoting preserved)."""
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    idx = rows[0].index(col_name)
    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL)
    for row in rows:
        writer.writerow([cell for i, cell in enumerate(row) if i != idx])
    return out.getvalue()


def test_parse_maps_fields_correctly():
    records = parse_aqs_daily_csv(FIXTURE_CSV)

    assert len(records) == EXPECTED_RECORD_COUNT

    la = records[0]
    assert isinstance(la, AqsDailyRecord)
    # site_id zero-padded to SS-CCC-NNNN from unpadded parts.
    assert la.site_id == "06-037-0002"
    assert (la.state_code, la.county_code, la.site_num) == ("6", "37", "2")
    assert la.parameter_code == "88101"
    assert la.parameter_name == "PM2.5 - Local Conditions"
    assert la.poc == 1
    assert la.lat == 33.9372
    assert la.lon == -118.1919
    assert la.date_local == "2023-07-01"
    assert la.concentration == 15.5
    assert la.units == "ug/m3 LC"
    assert la.aqi == 42
    assert la.method_code == "145"

    # Second day / same monitor keyed differently by date.
    assert records[1].date_local == "2023-07-02"
    assert records[1].concentration == 35.2
    assert records[1].aqi == 95

    # Empty AQI normalizes to None.
    ozone = records[2]
    assert ozone.parameter_code == "44201"
    assert ozone.units == "ppm"
    assert ozone.concentration == 0.055
    assert ozone.aqi is None

    # Negative AQI normalizes to None; site_id padded from 41/3/1.
    portland = records[3]
    assert portland.site_id == "41-003-0001"
    assert portland.concentration == 8.1
    assert portland.aqi is None


def test_parse_skips_rows_without_coords_or_concentration():
    records = parse_aqs_daily_csv(FIXTURE_CSV)
    assert len(records) == EXPECTED_RECORD_COUNT
    # The skipped rows would have keyed on 2023-07-03 (missing conc) and
    # 06-037-0005 (missing coords).
    assert {r.date_local for r in records} == {"2023-07-01", "2023-07-02"}


def test_fetch_aqs_daily_filters_by_site_and_date(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        records = parse_aqs_daily_csv(FIXTURE_CSV)
        store.insert_aqs_daily(records)
        all_keys = {r.natural_key for r in records}

        # site_id filter returns only that site's rows.
        la = store.fetch_aqs_daily(site_id="06-037-0002")
        assert {r.site_id for r in la} == {"06-037-0002"}
        assert {r.date_local for r in la} == {"2023-07-01", "2023-07-02"}
        assert {r.natural_key for r in la} == {
            k for k in all_keys if k[0] == "06-037-0002"
        }

        # date_local filter returns only that day's rows.
        day = store.fetch_aqs_daily(date_local="2023-07-01")
        assert {r.date_local for r in day} == {"2023-07-01"}
        assert {r.site_id for r in day} == {"06-037-0002", "41-003-0001"}
        assert {r.natural_key for r in day} == {
            k for k in all_keys if k[2] == "2023-07-01"
        }

        # Both filters together return exactly the matching rows.
        both = store.fetch_aqs_daily(site_id="06-037-0002", date_local="2023-07-01")
        assert len(both) == 2
        assert {r.parameter_code for r in both} == {"88101", "44201"}
        assert all(r.site_id == "06-037-0002" for r in both)
        assert all(r.date_local == "2023-07-01" for r in both)

        # A filter that matches nothing returns an empty list.
        assert store.fetch_aqs_daily(site_id="99-999-9999") == []
        assert store.fetch_aqs_daily(date_local="1999-01-01") == []
        assert store.fetch_aqs_daily(site_id="99-999-9999", date_local="1999-01-01") == []
    finally:
        store.close()


def test_fetch_aqs_daily_no_args_returns_all_rows(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        records = parse_aqs_daily_csv(FIXTURE_CSV)
        store.insert_aqs_daily(records)

        fetched = store.fetch_aqs_daily()
        assert len(fetched) == EXPECTED_RECORD_COUNT
        assert {r.natural_key for r in fetched} == {r.natural_key for r in records}
    finally:
        store.close()


def test_insert_aqs_daily_roundtrips_and_is_idempotent(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        records = parse_aqs_daily_csv(FIXTURE_CSV)
        assert store.insert_aqs_daily(records) == EXPECTED_RECORD_COUNT
        assert store.count_aqs_daily() == EXPECTED_RECORD_COUNT

        # Re-inserting the same records replaces in place: still one row per key.
        assert store.insert_aqs_daily(records) == EXPECTED_RECORD_COUNT
        assert store.count_aqs_daily() == EXPECTED_RECORD_COUNT

        fetched = store.fetch_aqs_daily()
        assert len(fetched) == EXPECTED_RECORD_COUNT
        by_key = {r.natural_key: r for r in fetched}
        for rec in records:
            assert by_key[rec.natural_key] == rec

        # Direct SQL check: exactly one row per natural key.
        conn = sqlite3.connect(store.db_path)
        rows = conn.execute(
            "SELECT site_id, parameter_code, date_local, poc, COUNT(*) "
            "FROM aqs_daily GROUP BY 1, 2, 3, 4 HAVING COUNT(*) > 1"
        ).fetchall()
        conn.close()
        assert rows == []
    finally:
        store.close()


def test_parse_tolerates_missing_column():
    # Dropping a trailing, unused column (CBSA Name) must not error or shift
    # the remaining column mapping.
    without_cbsa = _drop_column(FIXTURE_CSV, "CBSA Name")
    records = parse_aqs_daily_csv(without_cbsa)
    assert len(records) == EXPECTED_RECORD_COUNT
    assert records[0].site_id == "06-037-0002"
    assert records[0].concentration == 15.5
    assert records[0].aqi == 42
    assert records[0].date_local == "2023-07-01"


def test_parse_skips_rows_when_required_column_missing():
    # Without the Arithmetic Mean column no record has a concentration, so
    # everything is skipped rather than erroring.
    without_mean = _drop_column(FIXTURE_CSV, "Arithmetic Mean")
    assert parse_aqs_daily_csv(without_mean) == []


# ---------------------------------------------------------------------------
# POC sentinel round-trip
# ---------------------------------------------------------------------------


def test_aqs_poc_none_sentinel_roundtrip(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        rec = AqsDailyRecord(
            site_id="06-037-0002", state_code="6", county_code="37", site_num="2",
            parameter_code="88101", parameter_name="PM2.5 - Local Conditions",
            poc=None, lat=33.9372, lon=-118.1919, date_local="2023-07-01",
            concentration=15.5, units="ug/m3 LC", aqi=42, method_code="145",
        )
        # A missing POC is stored under the 0 sentinel...
        assert store.insert_aqs_daily([rec]) == 1
        # ...and read back as None, not 0.
        fetched = store.fetch_aqs_daily()
        assert len(fetched) == 1
        assert fetched[0].poc is None

        # The filtered fetch preserves the same sentinel read-back.
        filtered = store.fetch_aqs_daily(
            site_id="06-037-0002", date_local="2023-07-01"
        )
        assert len(filtered) == 1
        assert filtered[0].poc is None

        # The (site_id, parameter_code, date_local, poc=None) natural key still
        # upserts idempotently under the sentinel.
        assert store.insert_aqs_daily([rec]) == 1
        assert store.count_aqs_daily() == 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Zip extraction guards
# ---------------------------------------------------------------------------


def _write_zip_bytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members:
            zf.writestr(name, content)
    return buf.getvalue()


def test_extract_csvs_extracts_by_basename_with_traversal_guard(tmp_path):
    zip_path = tmp_path / "daily.zip"
    zip_path.write_bytes(_write_zip_bytes([
        ("daily_88101_2023.csv", "a,b\n1,2\n"),
        ("subdir/../daily_44201_2023.csv", "c,d\n3,4\n"),
        ("notes.txt", "not a csv"),
    ]))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    paths = _extract_csvs(zip_path, out_dir)
    # Only CSV members are extracted, flattened to their basename (no
    # subdirectory/.. traversal can escape out_dir).
    assert sorted(p.name for p in paths) == [
        "daily_44201_2023.csv",
        "daily_88101_2023.csv",
    ]
    assert (out_dir / "daily_88101_2023.csv").read_text() == "a,b\n1,2\n"
    assert (out_dir / "daily_44201_2023.csv").read_text() == "c,d\n3,4\n"


def test_extract_csvs_raises_when_size_limit_exceeded(tmp_path, monkeypatch):
    from backend.eval.accuracy.ingest import aqs as aqs_module

    zip_path = tmp_path / "daily.zip"
    zip_path.write_bytes(_write_zip_bytes([("daily_88101_2023.csv", "x" * 1024)]))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setattr(aqs_module, "MAX_EXTRACT_BYTES", 100)
    with pytest.raises(RuntimeError, match="decompressed size exceeds"):
        _extract_csvs(zip_path, out_dir)
    # Nothing was written before the cap tripped.
    assert list(out_dir.iterdir()) == []


def test_extract_csvs_raises_when_member_count_exceeded(tmp_path, monkeypatch):
    from backend.eval.accuracy.ingest import aqs as aqs_module

    zip_path = tmp_path / "daily.zip"
    zip_path.write_bytes(_write_zip_bytes([
        (f"daily_{i}.csv", "x") for i in range(3)
    ]))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setattr(aqs_module, "MAX_EXTRACT_MEMBERS", 2)
    with pytest.raises(RuntimeError, match="more than 2"):
        _extract_csvs(zip_path, out_dir)
