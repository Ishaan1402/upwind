"""NASA FIRMS historical area-API ingest adapter.

FIRMS's area endpoint doubles as a historical archive when given a day range
and start date: ``.../area/csv/{map_key}/{source}/{bbox}/{day_range}/{YYYY-MM-DD}``
returns data for ``[date .. date + day_range - 1]`` (day_range 1-5). The
endpoint rejects comma-joined source lists, so each source is requested
individually per 5-day chunk and the parsed rows are merged. This module chunks
the requested window into 5-day requests, reuses the FIRMS CSV parser from
``backend.services.firms`` (no duplication), and persists the detections as
``FirmsHotspotRecord`` rows.

Availability split: the NRT products (``VIIRS_*_NRT``, the default for the
LIVE ``fetch_firms_hotspots`` path) only retain a rolling ~7 days via the area
API, while the SP (Standard Processing) products (``VIIRS_SNPP_SP``,
``VIIRS_NOAA20_SP``, ``MODIS_SP``) serve the full historical archive with a
~1-3 month processing lag. This adapter therefore defaults to the SP sources;
``MODIS_SP`` (1 km resolution) is a pre-2012 option for older windows.
"""

import time
from datetime import date, timedelta
from typing import Iterable, List, Optional, Tuple

import httpx

from backend.config import FIRMS_MAP_KEY
from backend.eval.accuracy.records import FirmsHotspotRecord
from backend.eval.accuracy.store import AccuracyStore
from backend.services.firms import (
    parse_firms_acq_datetime,
    parse_firms_csv_rows,
)

FIRMS_AREA_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# SP (Standard Processing) products: the historical archive with ~1-3 month
# processing lag, as opposed to the NRT products (FIRMS_SOURCES in
# backend.services.firms) which only retain a rolling ~7 days.
FIRMS_HISTORICAL_SOURCES = ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP")

# FIRMS area API supports day ranges of 1-5; longer windows are chunked.
_MAX_DAYS_PER_REQUEST = 5
# Be polite to the free API between chunk requests.
_REQUEST_DELAY_S = 0.3

_TIMEOUT_S = 60.0


def _date_windows(start_date: str, end_date: str) -> List[Tuple[str, str]]:
    """Split ``[start_date, end_date]`` into 5-day (max) ``(start, end)``
    ``YYYY-MM-DD`` windows, covering the range exactly once each."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    windows: List[Tuple[str, str]] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=_MAX_DAYS_PER_REQUEST - 1), end)
        windows.append((chunk_start.isoformat(), chunk_end.isoformat()))
        chunk_start = chunk_end + timedelta(days=1)
    return windows


def fetch_firms_historical(
    start_date: str,
    end_date: str,
    bbox: Tuple[float, float, float, float],
    map_key: Optional[str] = None,
    sources: Iterable[str] = FIRMS_HISTORICAL_SOURCES,
) -> List[FirmsHotspotRecord]:
    """Fetch historical FIRMS hotspots over ``[start_date, end_date]`` inside
    ``bbox`` (west, south, east, north).

    Windows longer than 5 days are requested in 5-day chunks and accumulated.
    Each source in ``sources`` (default: ``FIRMS_HISTORICAL_SOURCES``, the
    VIIRS SP / Standard Processing products with ~1-3 month processing lag) is
    requested with its own single-source URL per chunk, because FIRMS rejects
    comma-joined source lists; rows from all sources are merged. The NRT
    products used by the LIVE ``fetch_firms_hotspots`` path only retain a
    rolling ~7 days and are not appropriate here. A non-200 (or empty)
    response from one source is treated as "no data for that source" and does
    NOT abort the chunk. Rows whose acquisition datetime cannot be parsed are
    skipped. ``map_key`` falls back to ``FIRMS_MAP_KEY`` from config when
    omitted; without a key no records are returned.
    """
    key = map_key or FIRMS_MAP_KEY
    if not key:
        return []

    west, south, east, north = bbox
    windows = _date_windows(start_date, end_date)
    records: List[FirmsHotspotRecord] = []
    with httpx.Client(timeout=_TIMEOUT_S) as client:
        for idx, (window_start, window_end) in enumerate(windows):
            day_range = (
                date.fromisoformat(window_end) - date.fromisoformat(window_start)
            ).days + 1
            for source in sources:
                url = (
                    f"{FIRMS_AREA_BASE_URL}/{key}/{source}/"
                    f"{west},{south},{east},{north}/{day_range}/{window_start}"
                )
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                except httpx.HTTPError:
                    # No data for this source (e.g. HTTP 400 "Invalid source"):
                    # tolerate it and keep the other sources / chunks going.
                    continue
                for row in parse_firms_csv_rows(resp.text):
                    acq_dt = parse_firms_acq_datetime(
                        row.get("acq_date"), row.get("acq_time")
                    )
                    if acq_dt is None:
                        continue
                    records.append(FirmsHotspotRecord(
                        lat=row["lat"],
                        lon=row["lon"],
                        frp=row["frp"],
                        acq_datetime=acq_dt.isoformat(),
                        confidence=row.get("confidence"),
                        satellite=row.get("satellite"),
                        daynight=row.get("daynight"),
                    ))
            if idx < len(windows) - 1:
                time.sleep(_REQUEST_DELAY_S)
    return records


def ingest_firms_historical(
    start_date: str,
    end_date: str,
    bbox: Tuple[float, float, float, float],
    store: AccuracyStore,
    map_key: Optional[str] = None,
) -> int:
    """Fetch historical FIRMS hotspots and persist them to ``store``.

    ``map_key`` defaults to ``FIRMS_MAP_KEY`` from config. Returns the number
    of records written.
    """
    records = fetch_firms_historical(start_date, end_date, bbox, map_key=map_key)
    return store.insert_firms_hotspots(records)
