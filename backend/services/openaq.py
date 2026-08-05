"""
OpenAQ API v3 integration for Upwind.

Pulls raw pollutant concentrations from US reference-grade monitors only
(monitor=true, mobile=false, iso=US). These are often state/local monitors
rather than EPA-owned sites, so user-facing copy uses the provider name when
available and "air quality monitor" otherwise. OpenAQ serves physical
concentrations rather than AQI values, which supplements the attribution
scoring quality.

"""

import math
import time
from datetime import datetime, timedelta, timezone as dt_timezone
from statistics import median
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import httpx

from backend.config import OPENAQ_API_KEY

OPENAQ_BASE_URL = "https://api.openaq.org"
OPENAQ_TIMEOUT_S = 3.0

# Search radius in meters: prefer monitors close enough to be representative,
# widen to the OpenAQ API maximum only when nothing is found nearby.
OPENAQ_PREFERRED_RADIUS_M = 10_000
OPENAQ_RADIUS_M = 25_000

# Cached data TTLs (seconds).
CACHE_TTL_LOCATION_S = 24 * 3600
CACHE_TTL_LATEST_S = 15 * 60
CACHE_TTL_BASELINE_S = 24 * 3600

# Freshness gate: reference monitors report hourly; drop anything older.
MAX_READING_AGE_S = 3 * 3600

# Baseline window for "unusual for this location" context.
BASELINE_DAYS = 365
SAME_HOUR_WINDOW_DAYS = 30
SAME_HOUR_MIN_SAMPLES = 5

# Completeness gate for aggregated records (OpenAQ's documented threshold).
MIN_PERCENT_COMPLETE = 75.0

CANONICAL_PM_UNIT = "µg/m³"
CANONICAL_PPB_UNIT = "ppb"
CANONICAL_CO_UNIT = "ppm"

# OpenAQ owner strings that carry no useful source information.
_GENERIC_SOURCE_NAMES = {"", "unknown governmental organization"}

_CACHE_BUCKETS: Dict[str, Dict[Any, Tuple[float, Any]]] = {}


def _cache_get(bucket: str, key: Any) -> Optional[Any]:
    entry = _CACHE_BUCKETS.get(bucket, {}).get(key)
    if not entry:
        return None
    expires_at, value = entry
    if time.time() >= expires_at:
        _CACHE_BUCKETS.get(bucket, {}).pop(key, None)
        return None
    return value


def _cache_set(bucket: str, key: Any, value: Any, ttl_s: float) -> None:
    _CACHE_BUCKETS.setdefault(bucket, {})[key] = (time.time() + ttl_s, value)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two WGS84 points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(d_lambda / 2) ** 2
    )
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=OPENAQ_TIMEOUT_S,
        headers={"X-API-Key": OPENAQ_API_KEY},
    )


def _parse_utc(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def normalize_reading(parameter: str, value: Any, units: Optional[str]) -> Optional[Tuple[float, str]]:
    """
    Normalize a reading to canonical units.

    PM2.5/PM10 -> µg/m³, O3/NO2/SO2 -> ppb, CO -> ppm.
    ppm -> ppb is an exact scaling; µg/m³ gas readings are skipped rather than
    converted with temperature-dependent assumptions (no guessing).
    """
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    u = (units or "").strip().lower().replace(" ", "").replace("μ", "µ")
    param = (parameter or "").lower()

    if param in ("pm25", "pm10"):
        if u in ("µg/m³", "µg/m3", "ug/m3", "ug/m³"):
            return round(value, 2), CANONICAL_PM_UNIT
        return None
    if param in ("o3", "no2", "so2"):
        if u == "ppb":
            return round(value, 1), CANONICAL_PPB_UNIT
        if u == "ppm":
            return round(value * 1000.0, 1), CANONICAL_PPB_UNIT
        return None
    if param == "co":
        if u == "ppm":
            return round(value, 2), CANONICAL_CO_UNIT
        return None
    return None


def _percent_complete(record: Dict[str, Any]) -> Optional[float]:
    """Read percentComplete from either nested or flattened coverage fields."""
    coverage = record.get("coverage") or {}
    pct = coverage.get("percentComplete")
    if pct is None:
        pct = record.get("percent_complete")
    if pct is None:
        return None
    try:
        return float(pct)
    except (TypeError, ValueError):
        return None


def monitor_source_label(monitor: Dict[str, Any]) -> str:
    """
    Short human label for a monitor's operator: provider name when available,
    otherwise "air quality monitor". Skips generic owner strings.
    """
    provider = (monitor.get("provider") or "").strip()
    if provider and provider.lower() not in _GENERIC_SOURCE_NAMES:
        return f"{provider} monitor"
    owner = (monitor.get("owner") or "").strip()
    if owner and owner.lower() not in _GENERIC_SOURCE_NAMES:
        return f"{owner} monitor"
    return "air quality monitor"


async def _fetch_location_results(lat: float, lon: float, radius_m: int) -> Optional[List[Dict[str, Any]]]:
    """Raw reference-monitor location search; None on failure, [] on no matches."""
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": str(radius_m),
        "monitor": "true",
        "mobile": "false",
        "iso": "US",
        "limit": "1000",
    }
    try:
        async with _client() as client:
            resp = await client.get(f"{OPENAQ_BASE_URL}/v3/locations", params=params)
            if resp.status_code != 200:
                return None
            data = resp.json()
    except Exception:
        return None
    return data.get("results", []) if isinstance(data, dict) else []


async def discover_reference_monitors(
    lat: float, lon: float, limit: int = 3, radius_m: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Find the nearest US reference-grade monitors within radius_m (default: 10 km,
    widening to the API max when nothing is found), sorted by distance.

    Returns a list of candidate monitor metadata (nearest first), or an empty
    list when no monitor is nearby, the key is missing, or the request fails.
    """
    if not OPENAQ_API_KEY:
        return []

    results = await _fetch_location_results(lat, lon, radius_m or OPENAQ_PREFERRED_RADIUS_M)
    if results is None:
        return []
    if not results and radius_m is None:
        results = await _fetch_location_results(lat, lon, OPENAQ_RADIUS_M)
        if results is None:
            return []

    candidates = []
    for loc in results:
        coords = loc.get("coordinates") or {}
        lat2, lon2 = coords.get("latitude"), coords.get("longitude")
        if lat2 is None or lon2 is None:
            continue
        dist = haversine_km(lat, lon, float(lat2), float(lon2))
        provider = loc.get("provider") or {}
        owner = loc.get("owner") or {}
        candidates.append({
            "location_id": loc.get("id"),
            "name": loc.get("name"),
            "distance_km": round(dist, 2),
            "timezone": loc.get("timezone"),
            "provider": provider.get("name") if isinstance(provider, dict) else loc.get("provider_name"),
            "owner": owner.get("name") if isinstance(owner, dict) else loc.get("owner_name"),
        })

    candidates.sort(key=lambda m: m["distance_km"])
    return candidates[:limit]


async def discover_reference_monitor(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Find the single nearest US reference-grade monitor (10 km preferred, 25 km max)."""
    monitors = await discover_reference_monitors(lat, lon, limit=1)
    return monitors[0] if monitors else None


async def fetch_location_sensors(location_id: int) -> Dict[int, Dict[str, Any]]:
    """
    Map sensor id -> {name, units} for a location, cached 24h.
    """
    cached = _cache_get("location", location_id)
    if cached is not None:
        return cached

    sensors_map: Dict[int, Dict[str, Any]] = {}
    if not OPENAQ_API_KEY:
        return sensors_map
    try:
        async with _client() as client:
            resp = await client.get(f"{OPENAQ_BASE_URL}/v3/locations/{location_id}")
            if resp.status_code != 200:
                return sensors_map
            data = resp.json()
    except Exception:
        return sensors_map

    results = data.get("results") or []
    if results:
        for sensor in results[0].get("sensors") or []:
            param = sensor.get("parameter") or {}
            name = (param.get("name") or "").lower()
            sid = sensor.get("id")
            if sid is not None and name:
                sensors_map[sid] = {
                    "name": name,
                    "units": param.get("units"),
                }
    _cache_set("location", location_id, sensors_map, CACHE_TTL_LOCATION_S)
    return sensors_map


async def fetch_latest(location_id: int) -> Dict[str, Dict[str, Any]]:
    """
    Latest fresh readings for a location, keyed by canonical parameter name.

    Readings are normalized to canonical units and dropped if older than 3h.
    Cached 15 minutes.
    """
    cached = _cache_get("latest", location_id)
    if cached is not None:
        return cached

    readings: Dict[str, Dict[str, Any]] = {}
    if not OPENAQ_API_KEY:
        return readings
    try:
        async with _client() as client:
            resp = await client.get(f"{OPENAQ_BASE_URL}/v3/locations/{location_id}/latest")
            if resp.status_code != 200:
                return readings
            data = resp.json()
    except Exception:
        return readings

    sensors_map = await fetch_location_sensors(location_id)
    now = datetime.now(dt_timezone.utc)
    results = data.get("results") or []
    for item in results:
        sensor_info = sensors_map.get(item.get("sensorsId"))
        if not sensor_info:
            continue
        normalized = normalize_reading(
            sensor_info["name"], item.get("value"), sensor_info.get("units")
        )
        if normalized is None:
            continue
        dt_utc = _parse_utc((item.get("datetime") or {}).get("utc"))
        if dt_utc is None or (now - dt_utc).total_seconds() > MAX_READING_AGE_S:
            continue
        value, unit = normalized
        readings[sensor_info["name"]] = {
            "value": value,
            "unit": unit,
            "as_of": dt_utc.isoformat(),
        }

    _cache_set("latest", location_id, readings, CACHE_TTL_LATEST_S)
    return readings


async def _fetch_aggregate_series(sensor_id: int, resource: str, datetime_from: datetime) -> List[Dict[str, Any]]:
    """Fetch and completeness-filter an aggregate series (days or hours)."""
    cache_key = (resource, sensor_id)
    cached = _cache_get("baseline", cache_key)
    if cached is not None:
        return cached

    if not OPENAQ_API_KEY:
        return []
    params = {
        "datetime_from": datetime_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": "1000",
    }
    try:
        async with _client() as client:
            resp = await client.get(
                f"{OPENAQ_BASE_URL}/v3/sensors/{sensor_id}/{resource}",
                params=params,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
    except Exception:
        return []

    results = data.get("results") or []
    filtered = []
    for record in results:
        pct = _percent_complete(record)
        # Strict gate: only use aggregates with verified completeness.
        if pct is None or pct < MIN_PERCENT_COMPLETE:
            continue
        if record.get("value") is None:
            continue
        filtered.append(record)
    _cache_set("baseline", cache_key, filtered, CACHE_TTL_BASELINE_S)
    return filtered


def _percentile_of(value: float, distribution: List[float]) -> float:
    if not distribution:
        return 0.0
    below = sum(1 for v in distribution if v <= value)
    return round(below / len(distribution) * 100.0, 1)


async def fetch_daily_baseline(sensor_id: int) -> Optional[Dict[str, Any]]:
    """
    Percentile of the most recent daily mean vs the past 365 days of daily
    means (all >=75% complete). Cached 24h.
    """
    now = datetime.now(dt_timezone.utc)
    records = await _fetch_aggregate_series(
        sensor_id, "days", now - timedelta(days=BASELINE_DAYS)
    )
    if not records:
        return None

    dated = []
    for record in records:
        dt_to = _parse_utc((record.get("period") or {}).get("datetimeTo", {}).get("utc"))
        if dt_to is None:
            continue
        dated.append((dt_to, float(record["value"])))
    if not dated:
        return None

    dated.sort(key=lambda pair: pair[0])
    today_value = dated[-1][1]
    distribution = [v for _, v in dated]
    return {
        "percentile": _percentile_of(today_value, distribution),
        "today_value": today_value,
        "count": len(distribution),
    }


async def fetch_same_hour_baseline(
    sensor_id: int,
    timezone_str: Optional[str],
    current_value: Optional[float],
) -> Optional[Dict[str, Any]]:
    """
    Same-hour-of-day context from the past 30 days of hourly means.

    Returns the percentile of the current reading among readings for the same
    local hour, plus the median of that same-hour distribution. Requires at
    least SAME_HOUR_MIN_SAMPLES to avoid noisy votes.
    """
    if current_value is None:
        return None

    now = datetime.now(dt_timezone.utc)
    records = await _fetch_aggregate_series(
        sensor_id, "hours", now - timedelta(days=SAME_HOUR_WINDOW_DAYS)
    )
    if not records:
        return None

    try:
        tz = ZoneInfo(timezone_str) if timezone_str else dt_timezone.utc
    except Exception:
        tz = dt_timezone.utc

    by_hour: Dict[int, List[float]] = {}
    for record in records:
        period = record.get("period") or {}
        local_str = (period.get("datetimeTo") or {}).get("local") or (period.get("datetimeFrom") or {}).get("local")
        local_dt = _parse_utc(local_str)
        if local_dt is None:
            continue
        by_hour.setdefault(local_dt.hour, []).append(float(record["value"]))

    current_hour = datetime.now(tz).hour
    same_hour_values = by_hour.get(current_hour, [])
    if len(same_hour_values) < SAME_HOUR_MIN_SAMPLES:
        return None

    return {
        "percentile": _percentile_of(current_value, same_hour_values),
        "median": round(median(same_hour_values), 2),
        "count": len(same_hour_values),
    }


def sensor_id_for_parameter(sensors_map: Dict[int, Dict[str, Any]], parameter: str) -> Optional[int]:
    for sid, info in sensors_map.items():
        if info.get("name") == parameter:
            return sid
    return None
