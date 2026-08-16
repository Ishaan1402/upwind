"""Open-Meteo historical weather archive ingest adapter.

Open-Meteo's free archive endpoint (``https://archive-api.open-meteo.com/v1/archive``)
serves ERA5 reanalysis weather for past date ranges. This module fetches daily
aggregates (tmax / tmin / 10m max wind speed / dominant wind direction) for a
set of sites and persists them as ``WeatherDailyRecord`` rows.

Raw values are stored as published in the requested units (Fahrenheit, mph);
days with no value for a variable carry None, and days with no values at all
are skipped.
"""

import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Optional, Tuple

import httpx

from backend.eval.accuracy.records import WeatherDailyRecord
from backend.eval.accuracy.store import AccuracyStore

WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Daily aggregates requested (temperature in Fahrenheit, wind in mph).
_DAILY_PARAMS = (
    "temperature_2m_max,temperature_2m_min,"
    "wind_speed_10m_max,wind_direction_10m_dominant"
)

_TIMEOUT_S = 30.0
# Be polite to the free archive API between per-site requests.
_SITE_DELAY_S = 0.5
# Attempts per site before giving up on 429 / 5xx / transport failures.
_MAX_ATTEMPTS = 4


def _opt_float(values: List[Optional[float]], idx: int) -> Optional[float]:
    """Numeric cell at ``idx``, or None when missing/null."""
    if idx >= len(values) or values[idx] is None:
        return None
    try:
        return float(values[idx])
    except (TypeError, ValueError):
        return None


def _opt_int(values: List[Optional[float]], idx: int) -> Optional[int]:
    """Integral cell at ``idx``, or None when missing/null."""
    value = _opt_float(values, idx)
    if value is None:
        return None
    return int(value)


def fetch_weather_daily(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    site_id: Optional[str] = None,
) -> List[WeatherDailyRecord]:
    """Fetch archived daily weather for one coordinate over
    ``[start_date, end_date]`` (both ``YYYY-MM-DD``).

    ``site_id`` defaults to a coordinate-derived id (``"<lat>,<lon>"``) when not
    given, so the adapter works standalone; the sites loop passes the canonical
    site id. Days whose four daily arrays are all null are skipped; a null in a
    single variable is kept as None on that field.

    Transient failures (HTTP 429, 5xx, and transport/connection errors) are
    retried up to ``_MAX_ATTEMPTS`` times with exponential backoff
    (``2 ** attempt`` seconds plus up to 0.5s of jitter). Non-retryable client
    errors (e.g. 400/404) are treated as "no data" and return ``[]`` rather
    than raising.
    """
    if site_id is None:
        site_id = f"{lat:.4f},{lon:.4f}"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": _DAILY_PARAMS,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "GMT",
    }
    last_error: Optional[Exception] = None
    with httpx.Client(timeout=_TIMEOUT_S) as client:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = client.get(WEATHER_ARCHIVE_URL, params=params)
                resp.raise_for_status()
                payload = resp.json()
                break
            except httpx.HTTPStatusError as exc:
                # Retry rate-limits (429) and server errors (5xx); any other
                # status (e.g. 400/404) means "no data for this site".
                if exc.response.status_code not in (429,) and exc.response.status_code < 500:
                    return []
                last_error = exc
            except httpx.TransportError as exc:
                last_error = exc
            if attempt < _MAX_ATTEMPTS - 1:
                # Exponential backoff (1s, 2s, 4s) plus jitter before retrying.
                time.sleep(2 ** attempt + random.uniform(0, 0.5))
        else:
            # Every attempt failed: surface the last 429/5xx/transport error.
            assert last_error is not None
            raise last_error

    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    wind_max = daily.get("wind_speed_10m_max") or []
    wind_dir = daily.get("wind_direction_10m_dominant") or []

    records: List[WeatherDailyRecord] = []
    for idx, date_local in enumerate(dates):
        values = (
            _opt_float(tmax, idx),
            _opt_float(tmin, idx),
            _opt_float(wind_max, idx),
            _opt_int(wind_dir, idx),
        )
        # A day with no values at all (missing row) is skipped entirely.
        if all(v is None for v in values):
            continue
        records.append(WeatherDailyRecord(
            site_id=site_id,
            lat=lat,
            lon=lon,
            date_local=date_local,
            tmax_f=values[0],
            tmin_f=values[1],
            wind_max_mph=values[2],
            wind_dir_dominant_deg=values[3],
        ))
    return records


def ingest_weather_for_sites(
    sites: Iterable[Tuple[str, float, float]],
    start_date: str,
    end_date: str,
    store: AccuracyStore,
    skip_existing: bool = True,
    workers: int = 4,
) -> int:
    """Fetch and persist archived daily weather for each ``(site_id, lat, lon)``.

    Returns the total number of records written.

    Resumable: when ``skip_existing`` is set (the default), any site that
    already has at least one ``weather_daily`` row inside
    ``[start_date, end_date]`` is skipped (a short ``skip <site>`` line is
    written to stderr), so a timed-out run can be re-invoked against the same
    store to finish the tail.

    Parallel: sites are fetched concurrently via a ``ThreadPoolExecutor`` with
    ``workers`` threads (default 4) — the sync ``fetch_weather_daily`` is
    I/O-bound, so threads are sufficient. Each worker sleeps ``_SITE_DELAY_S``
    (0.5s) before its request to keep the free archive API polite; the combined
    request rate is naturally capped at ``workers`` x the sequential rate rather
    than the raw thread rate. Per-site failures (e.g. a site that keeps getting
    429ed even after ``fetch_weather_daily``'s retries) do not abort the batch:
    the worker prints ``FAILED <site>`` to stderr, records the site id, and a
    final ``failed: [...]`` summary is printed after the run. Records are
    collected and inserted in ONE batched ``insert_weather_daily`` call at the
    end so SQLite is only written from the main thread, sorted by
    ``(site_id, date_local)`` so the stored row set is deterministic regardless
    of thread completion order (the insert is also idempotent under that
    natural key).
    """
    site_list = list(sites)
    if skip_existing:
        todo = []
        for site_id, lat, lon in site_list:
            if store.has_weather_coverage(site_id, start_date, end_date):
                print(f"skip {site_id}", file=sys.stderr)
            else:
                todo.append((site_id, lat, lon))
    else:
        todo = site_list
    if not todo:
        return 0

    failed: List[str] = []

    def _fetch_one(site: Tuple[str, float, float]) -> List[WeatherDailyRecord]:
        site_id, lat, lon = site
        # Polite throttle per worker, before the request.
        time.sleep(_SITE_DELAY_S)
        try:
            return fetch_weather_daily(lat, lon, start_date, end_date, site_id=site_id)
        except Exception as exc:
            # One persistently-failing site must not abort the whole batch.
            print(f"FAILED {site_id}: {exc}", file=sys.stderr)
            failed.append(site_id)
            return []

    all_records: List[WeatherDailyRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for records in executor.map(_fetch_one, todo):
            all_records.extend(records)

    if failed:
        print(f"failed: {failed}")

    # Deterministic persistence: sort by natural key so the stored rows do not
    # depend on thread completion order.
    all_records.sort(key=lambda r: (r.site_id, r.date_local))
    return store.insert_weather_daily(all_records)
