"""SQLite persistence for the accuracy-evaluation pipeline.

A small stdlib ``sqlite3`` store keyed on each record's natural key so
re-ingesting a year is idempotent (INSERT OR REPLACE). The default database
lives under ``backend/eval/accuracy/data/accuracy.db`` and is created lazily.
"""

import os
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from backend.eval.accuracy.records import (
    AqsDailyRecord,
    FirmsHotspotRecord,
    HmsSmokeRecord,
    LabelRecord,
    PredictionRecord,
    WeatherDailyRecord,
)

# Default DB path for the accuracy eval pipeline. The data/ dir is created
# lazily on first open.
ACCURACY_DB_PATH = str(Path(__file__).resolve().parent / "data" / "accuracy.db")

_AQS_DAILY_SCHEMA = """
CREATE TABLE IF NOT EXISTS aqs_daily (
    site_id         TEXT NOT NULL,
    state_code      TEXT NOT NULL,
    county_code     TEXT NOT NULL,
    site_num        TEXT NOT NULL,
    parameter_code  TEXT NOT NULL,
    parameter_name  TEXT,
    poc             INTEGER NOT NULL DEFAULT 0,
    lat             REAL,
    lon             REAL,
    date_local      TEXT NOT NULL,
    concentration   REAL,
    units           TEXT,
    aqi             INTEGER,
    method_code     TEXT,
    ingested_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (site_id, parameter_code, date_local, poc)
)
"""

_AQS_DAILY_COLUMNS = (
    "site_id, state_code, county_code, site_num, parameter_code, "
    "parameter_name, poc, lat, lon, date_local, concentration, "
    "units, aqi, method_code"
)

# Sentinel for a missing POC so the (site_id, parameter_code, date_local, poc)
# primary key stays non-NULL and INSERT OR REPLACE stays idempotent. EPA POCs
# are positive integers, so 0 cannot collide with real data.
_UNSET_POC = 0

_WEATHER_DAILY_SCHEMA = """
CREATE TABLE IF NOT EXISTS weather_daily (
    site_id               TEXT NOT NULL,
    lat                   REAL,
    lon                   REAL,
    date_local            TEXT NOT NULL,
    tmax_f                REAL,
    tmin_f                REAL,
    wind_max_mph          REAL,
    wind_dir_dominant_deg INTEGER,
    ingested_at           TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (site_id, date_local)
)
"""

_WEATHER_DAILY_COLUMNS = (
    "site_id, lat, lon, date_local, tmax_f, tmin_f, wind_max_mph, "
    "wind_dir_dominant_deg"
)

_FIRMS_HOTSPOT_SCHEMA = """
CREATE TABLE IF NOT EXISTS firms_hotspot (
    lat           REAL NOT NULL,
    lon           REAL NOT NULL,
    frp           REAL NOT NULL,
    acq_datetime  TEXT NOT NULL,
    confidence    TEXT,
    satellite     TEXT NOT NULL DEFAULT '',
    daynight      TEXT,
    ingested_at   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (lat, lon, acq_datetime, satellite)
)
"""

_FIRMS_HOTSPOT_COLUMNS = "lat, lon, frp, acq_datetime, confidence, satellite, daynight"

# Sentinel for a missing satellite so the (lat, lon, acq_datetime, satellite)
# primary key stays non-NULL and INSERT OR REPLACE stays idempotent (SQLite
# treats NULLs as distinct in a UNIQUE/PK). Empty string round-trips back to
# None on read.
_UNSET_SATELLITE = ""

_HMS_SMOKE_SCHEMA = """
CREATE TABLE IF NOT EXISTS hms_smoke (
    date_local     TEXT NOT NULL,
    density        TEXT NOT NULL,
    geometry_hash  TEXT NOT NULL,
    geometry_json  TEXT NOT NULL,
    ingested_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (date_local, density, geometry_hash)
)
"""

_HMS_SMOKE_COLUMNS = "date_local, density, geometry_hash, geometry_json"

_LABELS_SCHEMA = """
CREATE TABLE IF NOT EXISTS labels (
    site_id           TEXT NOT NULL,
    date_local        TEXT NOT NULL,
    aqi               INTEGER,
    primary_pollutant TEXT NOT NULL,
    label             TEXT NOT NULL,
    precision_tier    TEXT NOT NULL,
    reasoning         TEXT NOT NULL,
    ingested_at       TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (site_id, date_local)
)
"""

_LABELS_COLUMNS = (
    "site_id, date_local, aqi, primary_pollutant, label, precision_tier, reasoning"
)

_PREDICTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    site_id          TEXT NOT NULL,
    date_local       TEXT NOT NULL,
    true_label       TEXT NOT NULL,
    predicted_label  TEXT NOT NULL,
    top_score        REAL,
    top_confidence   TEXT,
    ingested_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (site_id, date_local)
)
"""

_PREDICTIONS_COLUMNS = (
    "site_id, date_local, true_label, predicted_label, top_score, top_confidence"
)

# Per-source ingest watermarks. ``last_date`` is the last successfully
# ingested date (or the latest year's Dec 31 for the year-granular AQS
# source); ``meta`` is optional per-source bookkeeping kept out of the way.
_INGEST_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_state (
    source    TEXT PRIMARY KEY,
    last_date TEXT,
    meta      TEXT
)
"""


class AccuracyStore:
    """Persists canonical historical records for the accuracy pipeline."""

    def __init__(self, db_path: os.PathLike | str = ACCURACY_DB_PATH):
        self.db_path = str(db_path)
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(_AQS_DAILY_SCHEMA)
        self._conn.execute(_WEATHER_DAILY_SCHEMA)
        self._conn.execute(_FIRMS_HOTSPOT_SCHEMA)
        self._conn.execute(_HMS_SMOKE_SCHEMA)
        self._conn.execute(_LABELS_SCHEMA)
        self._conn.execute(_PREDICTIONS_SCHEMA)
        self._conn.execute(_INGEST_STATE_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AccuracyStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def insert_aqs_daily(self, records: Iterable[AqsDailyRecord]) -> int:
        """Insert daily AQS records, replacing any row already present under the
        natural key (site_id, parameter_code, date_local, poc). Commits once.

        Returns the number of records written.
        """
        rows = [
            (
                r.site_id,
                r.state_code,
                r.county_code,
                r.site_num,
                r.parameter_code,
                r.parameter_name,
                r.poc if r.poc is not None else _UNSET_POC,
                r.lat,
                r.lon,
                r.date_local,
                r.concentration,
                r.units,
                r.aqi,
                r.method_code,
            )
            for r in records
        ]
        if not rows:
            return 0
        self._conn.executemany(
            f"INSERT OR REPLACE INTO aqs_daily ({_AQS_DAILY_COLUMNS}) "
            f"VALUES ({', '.join('?' * 14)})",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def fetch_aqs_daily(
        self,
        site_id: Optional[str] = None,
        date_local: Optional[str] = None,
    ) -> List[AqsDailyRecord]:
        """Read stored daily AQS records, ordered by natural key, optionally
        filtered to one site and/or one day. The missing-POC sentinel (0) is
        read back as None."""
        clauses: List[str] = []
        params: List[str] = []
        if site_id is not None:
            clauses.append("site_id = ?")
            params.append(site_id)
        if date_local is not None:
            clauses.append("date_local = ?")
            params.append(date_local)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = self._conn.execute(
            f"SELECT {_AQS_DAILY_COLUMNS} FROM aqs_daily{where} "
            "ORDER BY site_id, parameter_code, date_local, poc",
            params,
        )
        return [
            AqsDailyRecord(
                site_id=row[0],
                state_code=row[1],
                county_code=row[2],
                site_num=row[3],
                parameter_code=row[4],
                parameter_name=row[5],
                poc=row[6] if row[6] != _UNSET_POC else None,
                lat=row[7],
                lon=row[8],
                date_local=row[9],
                concentration=row[10],
                units=row[11],
                aqi=row[12],
                method_code=row[13],
            )
            for row in cursor
        ]

    def count_aqs_daily(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM aqs_daily").fetchone()[0]

    def fetch_aqs_sites(self) -> List[Tuple[str, float, float]]:
        """Distinct ``(site_id, lat, lon)`` triples from ``aqs_daily``, ordered
        by ``site_id``.

        Rows without usable coordinates are excluded (the AQS adapter already
        skips them on ingest, so this is just a defensive filter). Used to
        derive the weather site list and to scope ``run``/``label``/``ingest``
        to a bounding box.
        """
        cursor = self._conn.execute(
            "SELECT DISTINCT site_id, lat, lon FROM aqs_daily "
            "WHERE lat IS NOT NULL AND lon IS NOT NULL ORDER BY site_id"
        )
        return [(row[0], row[1], row[2]) for row in cursor]

    def fetch_aqs_date_bounds(self) -> Optional[Tuple[str, str]]:
        """Inclusive ``(min, max)`` of ``date_local`` in ``aqs_daily``, or None
        when the table is empty.

        ``date_local`` is always stored as ``YYYY-MM-DD`` text, so SQLite's
        lexical collation orders the bounds correctly.
        """
        row = self._conn.execute(
            "SELECT MIN(date_local), MAX(date_local) FROM aqs_daily"
        ).fetchone()
        if row[0] is None or row[1] is None:
            return None
        return (row[0], row[1])

    def fetch_aqs_site_days(self, start_date: str, end_date: str) -> List[Tuple[str, str]]:
        """Distinct ``(site_id, date_local)`` pairs that have at least one
        ``aqs_daily`` row inside the inclusive ``[start_date, end_date]``
        window, ordered deterministically by ``(site_id, date_local)``.

        ``date_local`` is always stored as ``YYYY-MM-DD`` text, so SQLite's
        lexical collation compares the window bounds correctly.
        """
        cursor = self._conn.execute(
            "SELECT DISTINCT site_id, date_local FROM aqs_daily "
            "WHERE date_local BETWEEN ? AND ? "
            "ORDER BY site_id, date_local",
            (start_date, end_date),
        )
        return [(row[0], row[1]) for row in cursor]

    def insert_weather_daily(self, records: Iterable[WeatherDailyRecord]) -> int:
        """Insert daily Open-Meteo weather records, replacing any row already
        present under the natural key (site_id, date_local). Commits once.

        Returns the number of records written.
        """
        rows = [
            (
                r.site_id,
                r.lat,
                r.lon,
                r.date_local,
                r.tmax_f,
                r.tmin_f,
                r.wind_max_mph,
                r.wind_dir_dominant_deg,
            )
            for r in records
        ]
        if not rows:
            return 0
        self._conn.executemany(
            f"INSERT OR REPLACE INTO weather_daily ({_WEATHER_DAILY_COLUMNS}) "
            f"VALUES ({', '.join('?' * 8)})",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def fetch_weather_daily(self, site_id: Optional[str] = None) -> List[WeatherDailyRecord]:
        """Read weather_daily rows, optionally filtered to one site, ordered by
        natural key."""
        if site_id is None:
            cursor = self._conn.execute(
                f"SELECT {_WEATHER_DAILY_COLUMNS} FROM weather_daily "
                "ORDER BY site_id, date_local"
            )
        else:
            cursor = self._conn.execute(
                f"SELECT {_WEATHER_DAILY_COLUMNS} FROM weather_daily "
                "WHERE site_id = ? ORDER BY site_id, date_local",
                (site_id,),
            )
        return [
            WeatherDailyRecord(
                site_id=row[0],
                lat=row[1],
                lon=row[2],
                date_local=row[3],
                tmax_f=row[4],
                tmin_f=row[5],
                wind_max_mph=row[6],
                wind_dir_dominant_deg=row[7],
            )
            for row in cursor
        ]

    def count_weather_daily(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM weather_daily").fetchone()[0]

    def has_weather_coverage(self, site_id: str, start_date: str, end_date: str) -> bool:
        """True when ``weather_daily`` already has a row for ``site_id`` inside
        the inclusive ``[start_date, end_date]`` window.

        Lets ``ingest_weather_for_sites`` skip finished sites on a resumed run.
        ``date_local`` is always stored as ``YYYY-MM-DD`` text, so SQLite's
        lexical collation compares the window bounds correctly.
        """
        row = self._conn.execute(
            "SELECT 1 FROM weather_daily "
            "WHERE site_id = ? AND date_local BETWEEN ? AND ? LIMIT 1",
            (site_id, start_date, end_date),
        ).fetchone()
        return row is not None

    def insert_firms_hotspots(self, records: Iterable[FirmsHotspotRecord]) -> int:
        """Insert FIRMS hotspot records, replacing any row already present under
        the natural key (lat, lon, acq_datetime, satellite). Lat/lon are stored
        at FULL float precision so distinct detections that differ only past the
        4th decimal are not silently merged into one row. Commits once.

        Returns the number of records written.
        """
        rows = [
            (
                r.lat,
                r.lon,
                r.frp,
                r.acq_datetime,
                r.confidence,
                r.satellite if r.satellite is not None else _UNSET_SATELLITE,
                r.daynight,
            )
            for r in records
        ]
        if not rows:
            return 0
        self._conn.executemany(
            f"INSERT OR REPLACE INTO firms_hotspot ({_FIRMS_HOTSPOT_COLUMNS}) "
            f"VALUES ({', '.join('?' * 7)})",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def fetch_firms_hotspots(self) -> List[FirmsHotspotRecord]:
        """Read all stored FIRMS hotspot records, ordered by natural key. The
        empty-string satellite sentinel is read back as None."""
        cursor = self._conn.execute(
            f"SELECT {_FIRMS_HOTSPOT_COLUMNS} FROM firms_hotspot "
            "ORDER BY lat, lon, acq_datetime, satellite"
        )
        return [
            FirmsHotspotRecord(
                lat=row[0],
                lon=row[1],
                frp=row[2],
                acq_datetime=row[3],
                confidence=row[4],
                satellite=row[5] if row[5] != _UNSET_SATELLITE else None,
                daynight=row[6],
            )
            for row in cursor
        ]

    def fetch_firms_hotspots_in_bbox(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
        start_iso: str,
        end_iso: str,
    ) -> List[FirmsHotspotRecord]:
        """Read FIRMS hotspot records inside a coarse lat/lon bbox whose
        acquisition datetime falls within ``[start_iso, end_iso]``.

        ``acq_datetime`` is always stored as an ISO8601 UTC string
        (``YYYY-MM-DDTHH:MM:SS+00:00``), so SQLite's lexical collation compares
        the window bounds correctly. The empty-string satellite sentinel is
        read back as None.
        """
        cursor = self._conn.execute(
            f"SELECT {_FIRMS_HOTSPOT_COLUMNS} FROM firms_hotspot "
            "WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? "
            "AND acq_datetime BETWEEN ? AND ? "
            "ORDER BY lat, lon, acq_datetime, satellite",
            (south, north, west, east, start_iso, end_iso),
        )
        return [
            FirmsHotspotRecord(
                lat=row[0],
                lon=row[1],
                frp=row[2],
                acq_datetime=row[3],
                confidence=row[4],
                satellite=row[5] if row[5] != _UNSET_SATELLITE else None,
                daynight=row[6],
            )
            for row in cursor
        ]

    def count_firms_hotspots(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM firms_hotspot").fetchone()[0]

    def insert_hms_smoke(self, records: Iterable[HmsSmokeRecord]) -> int:
        """Insert daily HMS smoke polygons, replacing any row already present
        under the natural key (date_local, density, geometry_hash). The
        geometry hash is derived from the record and persisted so the primary
        key round-trips. Commits once.

        Returns the number of records written.
        """
        rows = [
            (r.date_local, r.density, r.natural_key[2], r.geometry_json)
            for r in records
        ]
        if not rows:
            return 0
        self._conn.executemany(
            f"INSERT OR REPLACE INTO hms_smoke ({_HMS_SMOKE_COLUMNS}) "
            f"VALUES ({', '.join('?' * 4)})",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def fetch_hms_smoke(self, date_local: Optional[str] = None) -> List[HmsSmokeRecord]:
        """Read hms_smoke rows, optionally filtered to one day, ordered by
        natural key."""
        if date_local is None:
            cursor = self._conn.execute(
                f"SELECT {_HMS_SMOKE_COLUMNS} FROM hms_smoke "
                "ORDER BY date_local, density, geometry_hash"
            )
        else:
            cursor = self._conn.execute(
                f"SELECT {_HMS_SMOKE_COLUMNS} FROM hms_smoke "
                "WHERE date_local = ? ORDER BY date_local, density, geometry_hash",
                (date_local,),
            )
        return [
            HmsSmokeRecord(date_local=row[0], density=row[1], geometry_json=row[3])
            for row in cursor
        ]

    def count_hms_smoke(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM hms_smoke").fetchone()[0]

    def insert_labels(self, records: Iterable[LabelRecord]) -> int:
        """Insert derived labels, replacing any row already present under the
        natural key (site_id, date_local). Commits once.

        Returns the number of records written.
        """
        rows = [
            (
                r.site_id,
                r.date_local,
                r.aqi,
                r.primary_pollutant,
                r.label,
                r.precision_tier,
                r.reasoning,
            )
            for r in records
        ]
        if not rows:
            return 0
        self._conn.executemany(
            f"INSERT OR REPLACE INTO labels ({_LABELS_COLUMNS}) "
            f"VALUES ({', '.join('?' * 7)})",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def fetch_labels(self, label: Optional[str] = None) -> List[LabelRecord]:
        """Read stored labels, optionally filtered to one label class, ordered
        by natural key."""
        if label is None:
            cursor = self._conn.execute(
                f"SELECT {_LABELS_COLUMNS} FROM labels "
                "ORDER BY site_id, date_local"
            )
        else:
            cursor = self._conn.execute(
                f"SELECT {_LABELS_COLUMNS} FROM labels "
                "WHERE label = ? ORDER BY site_id, date_local",
                (label,),
            )
        return [
            LabelRecord(
                site_id=row[0],
                date_local=row[1],
                aqi=row[2],
                primary_pollutant=row[3],
                label=row[4],
                precision_tier=row[5],
                reasoning=row[6],
            )
            for row in cursor
        ]

    def count_labels(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]

    def insert_predictions(self, records: Iterable[PredictionRecord]) -> int:
        """Insert scorer-vs-label evaluation outcomes, replacing any row already
        present under the natural key (site_id, date_local). Commits once.

        Returns the number of records written.
        """
        rows = [
            (
                r.site_id,
                r.date_local,
                r.true_label,
                r.predicted_label,
                r.top_score,
                r.top_confidence,
            )
            for r in records
        ]
        if not rows:
            return 0
        self._conn.executemany(
            f"INSERT OR REPLACE INTO predictions ({_PREDICTIONS_COLUMNS}) "
            f"VALUES ({', '.join('?' * 6)})",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def fetch_predictions(self) -> List[PredictionRecord]:
        """Read all stored predictions, ordered by natural key."""
        cursor = self._conn.execute(
            f"SELECT {_PREDICTIONS_COLUMNS} FROM predictions "
            "ORDER BY site_id, date_local"
        )
        return [
            PredictionRecord(
                site_id=row[0],
                date_local=row[1],
                true_label=row[2],
                predicted_label=row[3],
                top_score=row[4],
                top_confidence=row[5],
            )
            for row in cursor
        ]

    def count_predictions(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def get_watermark(self, source: str) -> Optional[str]:
        """The last successfully ingested date for ``source``, or None when the
        source has never been ingested."""
        row = self._conn.execute(
            "SELECT last_date FROM ingest_state WHERE source = ?", (source,)
        ).fetchone()
        return row[0] if row is not None else None

    def set_watermark(
        self, source: str, last_date: str, meta: Optional[str] = None
    ) -> None:
        """Record ``last_date`` as the source's ingest watermark (upsert).

        ``meta`` is preserved when None so a plain watermark advance never
        clobbers per-source metadata written by an earlier run.
        """
        if meta is None:
            self._conn.execute(
                "INSERT INTO ingest_state (source, last_date, meta) "
                "VALUES (?, ?, NULL) "
                "ON CONFLICT(source) DO UPDATE SET last_date = excluded.last_date",
                (source, last_date),
            )
        else:
            self._conn.execute(
                "INSERT INTO ingest_state (source, last_date, meta) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(source) DO UPDATE SET last_date = excluded.last_date, "
                "meta = excluded.meta",
                (source, last_date, meta),
            )
        self._conn.commit()
