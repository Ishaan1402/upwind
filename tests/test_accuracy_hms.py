"""Tests for the accuracy-eval HMS historical smoke-polygon adapter (Phase 1b-B).

The KML fixture is hand-written: two polygon placemarks (light with a plain
outer ring, medium with an outer ring + a hole) and one point placemark that
must be skipped. All HTTP is mocked at the ``httpx.Client`` level (sync
client); no network access in these tests.
"""

import gzip
import io
import json
import sqlite3
import tarfile
import zipfile
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from backend.eval.accuracy.__main__ import main
from backend.eval.accuracy.ingest.hms import (
    _extract_zip,
    fetch_hms_smoke_daily,
    fetch_hms_smoke_year_tarball,
    ingest_hms_smoke_range,
    parse_hms_kml,
)
from backend.eval.accuracy.records import HmsSmokeRecord
from backend.eval.accuracy.store import AccuracyStore

# --------------------------------------------------------------------------
# Fixture KML (KML 2.2 default namespace, as NOAA publishes)
# --------------------------------------------------------------------------

FIXTURE_KML_NS = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <Placemark>
    <name>Smoke (Light)</name>
    <description>
      Smoke Attributes:
        Start Time: 2020005 1201
        Density: 5
        Satellite: GOES-EAST
    </description>
    <Polygon>
      <outerBoundaryIs>
        <LinearRing>
          <coordinates>
            -122.0,41.0,0
            -121.0,41.0,0
            -121.0,40.0,0
            -122.0,40.0,0
            -122.0,41.0,0
          </coordinates>
        </LinearRing>
      </outerBoundaryIs>
    </Polygon>
  </Placemark>
  <Placemark>
    <name>Smoke (Medium)</name>
    <description>
      Smoke Attributes:
        Start Time: 2020005 1500
        Density: 16
        Satellite: GOES-WEST
    </description>
    <Polygon>
      <outerBoundaryIs>
        <LinearRing>
          <coordinates>
            -120.0,42.0,0
            -119.0,42.0,0
            -119.0,41.0,0
            -120.0,41.0,0
            -120.0,42.0,0
          </coordinates>
        </LinearRing>
      </outerBoundaryIs>
      <innerBoundaryIs>
        <LinearRing>
          <coordinates>
            -119.7,41.7,0
            -119.3,41.7,0
            -119.3,41.3,0
            -119.7,41.3,0
            -119.7,41.7,0
          </coordinates>
        </LinearRing>
      </innerBoundaryIs>
    </Polygon>
  </Placemark>
  <Placemark>
    <name>Just a marker</name>
    <description>a point, not a polygon</description>
    <Point>
      <coordinates>-122.5,40.5,0</coordinates>
    </Point>
  </Placemark>
</Document>
</kml>
"""

# Same document without any namespace declaration (bare XML).
FIXTURE_KML_BARE = FIXTURE_KML_NS.replace(' xmlns="http://www.opengis.net/kml/2.2"', "", 1)

_MINIMAL_KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <Placemark>
    <description>no density info here</description>
    <Polygon>
      <outerBoundaryIs>
        <LinearRing>
          <coordinates>
            -100.0,40.0,0
            -99.0,40.0,0
            -99.0,39.0,0
            -100.0,39.0,0
            -100.0,40.0,0
          </coordinates>
        </LinearRing>
      </outerBoundaryIs>
    </Polygon>
  </Placemark>
</Document>
</kml>
"""


# --------------------------------------------------------------------------
# KML parsing
# --------------------------------------------------------------------------


def test_parse_hms_kml_namespaced():
    records = parse_hms_kml(FIXTURE_KML_NS, "2020-01-05")

    # Two polygon placemarks; the point placemark is skipped.
    assert len(records) == 2

    light = records[0]
    assert isinstance(light, HmsSmokeRecord)
    assert light.date_local == "2020-01-05"
    assert light.density == "light"
    assert light.natural_key[0] == "2020-01-05"
    assert light.natural_key[1] == "light"
    assert len(light.natural_key[2]) == 16  # 16-hex-char geometry hash

    geom = json.loads(light.geometry_json)
    assert geom["type"] == "Polygon"
    # Exterior ring only: [[lon, lat], ...].
    assert geom["coordinates"] == [[
        [-122.0, 41.0], [-121.0, 41.0], [-121.0, 40.0], [-122.0, 40.0], [-122.0, 41.0],
    ]]

    medium = records[1]
    assert medium.density == "medium"
    mgeom = json.loads(medium.geometry_json)
    # Outer ring + one hole.
    assert len(mgeom["coordinates"]) == 2
    assert mgeom["coordinates"][0][0] == [-120.0, 42.0]
    assert mgeom["coordinates"][1] == [
        [-119.7, 41.7], [-119.3, 41.7], [-119.3, 41.3], [-119.7, 41.3], [-119.7, 41.7],
    ]


def test_parse_hms_kml_unnamespaced():
    records = parse_hms_kml(FIXTURE_KML_BARE, "2020-01-05")
    assert len(records) == 2
    assert records[0].density == "light"
    assert records[1].density == "medium"
    assert json.loads(records[1].geometry_json)["type"] == "Polygon"


def test_parse_hms_kml_defaults_density_to_light():
    records = parse_hms_kml(_MINIMAL_KML, "2020-01-05")
    assert len(records) == 1
    assert records[0].density == "light"


def test_parse_hms_kml_tolerates_bad_input():
    assert parse_hms_kml("this is not xml at all", "2020-01-05") == []
    assert parse_hms_kml("<kml/>", "2020-01-05") == []


# --------------------------------------------------------------------------
# Store round-trip + idempotency
# --------------------------------------------------------------------------


def test_hms_store_roundtrip_and_idempotent(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    try:
        records = parse_hms_kml(FIXTURE_KML_NS, "2020-01-05")
        assert len(records) == 2

        assert store.insert_hms_smoke(records) == 2
        assert store.count_hms_smoke() == 2

        # Re-inserting the same records replaces in place: still one row per key.
        assert store.insert_hms_smoke(records) == 2
        assert store.count_hms_smoke() == 2

        fetched = store.fetch_hms_smoke()
        assert len(fetched) == 2
        by_key = {r.natural_key: r for r in fetched}
        for rec in records:
            assert by_key[rec.natural_key] == rec

        # Date filter narrows to the matching day.
        assert len(store.fetch_hms_smoke(date_local="2020-01-05")) == 2
        assert store.fetch_hms_smoke(date_local="1999-01-01") == []

        # Direct SQL check: exactly one row per natural key.
        conn = sqlite3.connect(store.db_path)
        dupes = conn.execute(
            "SELECT date_local, density, geometry_hash, COUNT(*) FROM hms_smoke "
            "GROUP BY 1, 2, 3 HAVING COUNT(*) > 1"
        ).fetchall()
        conn.close()
        assert dupes == []
    finally:
        store.close()


# --------------------------------------------------------------------------
# Daily fetch + range ingest (mocked HTTP)
# --------------------------------------------------------------------------


def _mock_http_response(status=200, content=b""):
    resp = Mock()
    resp.status_code = status
    resp.content = content
    return resp


def _make_sync_client(handler):
    """Sync ``httpx.Client`` stand-in: ``handler(url)`` -> resp."""
    client = Mock()
    client.get = Mock(side_effect=lambda url, **kwargs: handler(str(url)))
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    return client


def _tiny_zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hms_smoke20200105.shp", b"shp-bytes")
        zf.writestr("hms_smoke20200105.dbf", b"dbf-bytes")
        zf.writestr("hms_smoke20200105.shx", b"shx-bytes")
    return buf.getvalue()


def test_fetch_hms_smoke_daily_downloads_kml(tmp_path):
    kml_bytes = FIXTURE_KML_NS.encode("utf-8")
    requested = []

    def handler(url):
        requested.append(url)
        return _mock_http_response(status=200, content=kml_bytes)

    client = _make_sync_client(handler)
    with patch("backend.eval.accuracy.ingest.hms.httpx.Client", return_value=client):
        paths = fetch_hms_smoke_daily(datetime(2020, 1, 5), tmp_path)

    assert len(paths) == 1
    assert paths[0].name == "smoke20200105.kml"
    assert paths[0].read_bytes() == kml_bytes
    assert requested == [
        "https://satepsanone.nesdis.noaa.gov/pub/FIRE/HMS/hms_backup/2020/"
        "KML/smoke20200105.kml",
    ]


def test_fetch_hms_smoke_daily_falls_back_to_zip(tmp_path):
    requested = []

    def handler(url):
        requested.append(url)
        if "KML/smoke20200105.kml" in url:
            return _mock_http_response(status=404)
        if "GIS/SMOKE/hms_smoke20200105.zip" in url:
            return _mock_http_response(status=200, content=_tiny_zip_bytes())
        return _mock_http_response(status=404)

    client = _make_sync_client(handler)
    with patch("backend.eval.accuracy.ingest.hms.httpx.Client", return_value=client):
        paths = fetch_hms_smoke_daily(datetime(2020, 1, 5), tmp_path)

    # Zip fallback downloads + extracts every member.
    assert sorted(p.name for p in paths) == [
        "hms_smoke20200105.dbf",
        "hms_smoke20200105.shp",
        "hms_smoke20200105.shx",
    ]
    assert (tmp_path / "hms_smoke20200105.shp").read_bytes() == b"shp-bytes"
    assert len(requested) == 2


def test_fetch_hms_smoke_daily_missing_day_returns_empty(tmp_path):
    def handler(url):
        return _mock_http_response(status=404)

    client = _make_sync_client(handler)
    with patch("backend.eval.accuracy.ingest.hms.httpx.Client", return_value=client):
        paths = fetch_hms_smoke_daily(datetime(2020, 1, 5), tmp_path)

    assert paths == []
    assert list(tmp_path.iterdir()) == []  # nothing downloaded


def test_extract_zip_raises_when_size_limit_exceeded(tmp_path, monkeypatch):
    from backend.eval.accuracy.ingest import hms as hms_module

    zip_path = tmp_path / "hms_smoke20200105.zip"
    zip_path.write_bytes(_tiny_zip_bytes())
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setattr(hms_module, "MAX_EXTRACT_BYTES", 4)
    with pytest.raises(RuntimeError, match="decompressed size exceeds"):
        _extract_zip(zip_path, out_dir)
    # Nothing was written before the cap tripped.
    assert list(out_dir.iterdir()) == []


def test_ingest_hms_smoke_range_skips_missing_days(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    out_dir = tmp_path / "hms"
    requested = []

    def handler(url):
        requested.append(url)
        if "KML/smoke20200105.kml" in url:
            return _mock_http_response(status=200, content=FIXTURE_KML_NS.encode("utf-8"))
        return _mock_http_response(status=404)

    client = _make_sync_client(handler)
    try:
        with patch("backend.eval.accuracy.ingest.hms.httpx.Client", return_value=client), \
             patch("backend.eval.accuracy.ingest.hms.time.sleep") as sleep:
            count = ingest_hms_smoke_range("2020-01-05", "2020-01-06", store, out_dir)

        # Day 05 yields 2 records; day 06 404s on both URLs and is skipped.
        assert count == 2
        assert store.count_hms_smoke() == 2
        assert {r.date_local for r in store.fetch_hms_smoke()} == {"2020-01-05"}
        # Polite delay between days, none before the first request.
        assert sleep.call_count == 1
        # Day 06 tries KML then zip fallback; day 05 tries KML only.
        assert len(requested) == 3
    finally:
        store.close()


# --------------------------------------------------------------------------
# Yearly-tarball path (2003-2019 archives, per module docstring)
# --------------------------------------------------------------------------

_TARBALL_BASE = "https://satepsanone.nesdis.noaa.gov/pub/FIRE/HMS/hms_backup"


def _year_tar_bytes(members):
    """Build an in-memory ``.tar`` from ``(member_name, raw_bytes)`` pairs."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _mock_stream_response(status=200, content=b""):
    """Response stand-in for the yearly-tarball download (``client.stream``)."""
    resp = _mock_http_response(status=status, content=content)
    resp.iter_bytes = Mock(return_value=iter([content]))
    resp.raise_for_status = Mock()
    resp.__enter__ = Mock(return_value=resp)
    resp.__exit__ = Mock(return_value=False)
    return resp


def _make_stream_client(handler):
    """Sync ``httpx.Client`` stand-in covering both ``.get`` (per-day path) and
    ``.stream`` (tarball path): ``handler(url)`` -> resp."""
    client = Mock()
    client.get = Mock(side_effect=lambda url, **kwargs: handler(str(url)))
    client.stream = Mock(side_effect=lambda method, url, **kwargs: handler(str(url)))
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    return client


def test_fetch_hms_smoke_year_tarball_extracts_smoke_kmls(tmp_path):
    kml_bytes = FIXTURE_KML_NS.encode("utf-8")
    tar_bytes = _year_tar_bytes([
        ("2019/KML/smoke20190702.kml.gz", gzip.compress(kml_bytes)),
        ("2019/KML/smoke20190703.kml", kml_bytes),
        ("2019/KML/fire20190702.kml.gz", gzip.compress(b"<kml/>")),
    ])

    def handler(url):
        if url.endswith("/2019.tar"):
            return _mock_stream_response(status=200, content=tar_bytes)
        return _mock_stream_response(status=404)

    client = _make_stream_client(handler)
    with patch("backend.eval.accuracy.ingest.hms.httpx.Client", return_value=client):
        paths = fetch_hms_smoke_year_tarball(2019, tmp_path)

    # Only the daily smoke KMLs are extracted (gzipped and plain); the
    # fire/hysplit product KMLs are NOT smoke records and are skipped.
    assert sorted(p.name for p in paths) == ["smoke20190702.kml", "smoke20190703.kml"]
    assert (tmp_path / "smoke20190702.kml").read_text() == FIXTURE_KML_NS
    assert (tmp_path / "smoke20190703.kml").read_text() == FIXTURE_KML_NS
    # The ~1GB tarball is cached for resumable re-runs (AQS-style).
    assert (tmp_path / "2019.tar").read_bytes() == tar_bytes


def test_ingest_hms_smoke_range_uses_yearly_tarball_and_advances_watermark(tmp_path):
    db_path = tmp_path / "accuracy.db"
    with AccuracyStore(db_path):
        pass
    tar_bytes = _year_tar_bytes([
        ("2019/KML/smoke20190702.kml.gz", gzip.compress(FIXTURE_KML_NS.encode("utf-8"))),
        ("2019/KML/smoke20190703.kml.gz", gzip.compress(FIXTURE_KML_NS.encode("utf-8"))),
    ])
    requested = []

    def handler(url):
        requested.append(url)
        if url.endswith("/2019.tar"):
            return _mock_stream_response(status=200, content=tar_bytes)
        return _mock_stream_response(status=404)

    client = _make_stream_client(handler)
    with patch("backend.eval.accuracy.ingest.hms.httpx.Client", return_value=client), \
         patch("backend.eval.accuracy.ingest.hms.time.sleep"), \
         patch("backend.eval.accuracy.__main__.RAW_DATA_DIR", tmp_path / "raw"):
        rc = main([
            "ingest", "hms",
            "--start", "2019-07-02", "--end", "2019-07-03",
            "--db", str(db_path),
        ])

    assert rc == 0
    # The CLI watermark advances exactly as it does for the per-day path.
    with AccuracyStore(db_path) as store:
        assert store.get_watermark("hms") == "2019-07-03"
        records = store.fetch_hms_smoke()
        # 2 days x 2 polygon placemarks each from the tarball's KMLs.
        assert len(records) == 4
        assert {r.date_local for r in records} == {"2019-07-02", "2019-07-03"}
        for rec in records:
            assert json.loads(rec.geometry_json)["type"] == "Polygon"
    # One tarball fetch for the year; both days are served from it (no per-day
    # requests, no .tar.gz fallback).
    assert requested == [f"{_TARBALL_BASE}/2019.tar"]


def test_ingest_hms_smoke_range_falls_back_to_per_day_when_tarball_missing(tmp_path):
    store = AccuracyStore(tmp_path / "accuracy.db")
    out_dir = tmp_path / "hms"
    requested = []

    def handler(url):
        requested.append(url)
        if url.endswith(".tar") or url.endswith(".tar.gz"):
            return _mock_stream_response(status=404)
        if "KML/smoke20190702.kml" in url:
            return _mock_http_response(status=200, content=FIXTURE_KML_NS.encode("utf-8"))
        return _mock_http_response(status=404)

    client = _make_stream_client(handler)
    try:
        with patch("backend.eval.accuracy.ingest.hms.httpx.Client", return_value=client), \
             patch("backend.eval.accuracy.ingest.hms.time.sleep"):
            count = ingest_hms_smoke_range("2019-07-02", "2019-07-02", store, out_dir)

        assert count == 2
        assert store.count_hms_smoke() == 2
        # Both tarball candidates 404'd, then the per-day KML path served the day.
        assert requested[0] == f"{_TARBALL_BASE}/2019.tar"
        assert requested[1] == f"{_TARBALL_BASE}/2019.tar.gz"
        assert any("KML/smoke20190702.kml" in u for u in requested)
    finally:
        store.close()


def test_ingest_hms_smoke_range_skips_empty_tarball_year(tmp_path):
    """A tarball that exists but carries no smoke KMLs (e.g. 2016-2018)
    ingests 0 records and does not hammer the per-day URLs."""
    store = AccuracyStore(tmp_path / "accuracy.db")
    out_dir = tmp_path / "hms"
    requested = []
    tar_bytes = _year_tar_bytes([
        ("2019/KML/fire20190702.kml.gz", gzip.compress(b"<kml/>")),
    ])

    def handler(url):
        requested.append(url)
        if url.endswith("/2019.tar"):
            return _mock_stream_response(status=200, content=tar_bytes)
        return _mock_stream_response(status=404)

    client = _make_stream_client(handler)
    try:
        with patch("backend.eval.accuracy.ingest.hms.httpx.Client", return_value=client), \
             patch("backend.eval.accuracy.ingest.hms.time.sleep"):
            count = ingest_hms_smoke_range("2019-07-02", "2019-07-02", store, out_dir)

        assert count == 0
        assert store.count_hms_smoke() == 0
        # The tarball was fetched once and that is the only HTTP request.
        assert requested == [f"{_TARBALL_BASE}/2019.tar"]
    finally:
        store.close()
