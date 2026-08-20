from typing import Optional, Dict, Any, List
import math
import httpx
from datetime import datetime, timedelta, timezone as dt_timezone
from backend.config import AIRNOW_KEY, get_aqi_category
from backend.engine.params import get_params

AIRNOW_DATA_URL = "https://www.airnowapi.org/aq/data/"
AIRNOW_TIMEOUT_S = 3.0

# AirNow /aq/data/ conventions: -999 (and null) mean "no data"; the raw
# concentration is an unvalidated fallback. Concentrations are always UG/M3.
AIRNOW_MISSING = -999
AIRNOW_LOOKBACK_HOURS = 3
AIRNOW_BBOX_DEG = 0.25

async def fetch_airnow_observation(lat: float, lon: float, distance_miles: int = 25) -> Optional[Dict[str, Any]]:
    """
    Fetch current AQI observation from the AirNow ``/aq/data/`` feed.

    Queries the raw hourly endpoint (``dataType=A``) over a 3h lookback inside a
    +/-0.25 deg BBOX and returns the max-AQI row in the latest UTC hour as the
    standardized observation dictionary. This replaces the retired
    ``/aq/observation/latLong/current/`` endpoint (sunset Sept 30, 2026).

    Returns None if the key is missing, the request fails, or no valid AQI rows
    exist. ``distance_miles`` is retained for call compatibility but is no longer
    used: the query window is the fixed +/-0.25 deg BBOX.
    """
    if not AIRNOW_KEY:
        return None

    now = datetime.now(dt_timezone.utc)
    start = now - timedelta(hours=AIRNOW_LOOKBACK_HOURS)
    params = {
        "startDate": start.strftime("%Y-%m-%dT%H"),
        "endDate": now.strftime("%Y-%m-%dT%H"),
        "parameters": "PM25,PM10,OZONE,CO,NO2,SO2",
        "BBOX": (
            f"{lon - AIRNOW_BBOX_DEG},{lat - AIRNOW_BBOX_DEG},"
            f"{lon + AIRNOW_BBOX_DEG},{lat + AIRNOW_BBOX_DEG}"
        ),
        "dataType": "A",
        "format": "application/json",
        "verbose": 1,
        "monitorType": 0,
        "includerawconcentrations": 0,
        "API_KEY": AIRNOW_KEY,
    }
    try:
        async with httpx.AsyncClient(timeout=AIRNOW_TIMEOUT_S) as client:
            response = await client.get(AIRNOW_DATA_URL, params=params)
            if response.status_code != 200:
                return None
            data = response.json()
    except Exception:
        return None

    if not isinstance(data, list) or len(data) == 0:
        return None

    return _build_aqi_observation(data)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two WGS84 points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(d_lambda / 2) ** 2
    )
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _is_missing(value: Any) -> bool:
    """True when an AirNow value is the numeric -999 missing sentinel."""
    try:
        return float(value) == AIRNOW_MISSING
    except (TypeError, ValueError):
        return False


def _parse_airnow_utc(value: Any) -> Optional[datetime]:
    """Parse an AirNow UTC timestamp; naive strings are treated as UTC."""
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=dt_timezone.utc)
    return dt.astimezone(dt_timezone.utc)


def _hour_key(dt: datetime) -> datetime:
    """Bucket a timestamp to its UTC hour for cross-site comparability."""
    return dt.replace(minute=0, second=0, microsecond=0)


def _build_aqi_observation(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Standardized AQI observation from raw ``/aq/data/`` rows.

    Rows are PascalCase with ``Parameter`` values PM2.5/PM10/O3/CO/NO2/SO2 and
    ``AQI`` -999/null marking no data. The max-AQI row in the latest UTC hour
    becomes the primary pollutant; every valid AQI in that hour populates the
    pollutants map. Returns None when there are no valid AQI rows.
    """
    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        aqi = row.get("AQI")
        if aqi is None or _is_missing(aqi):
            continue
        utc = _parse_airnow_utc(row.get("UTC"))
        if utc is None:
            continue
        try:
            aqi = int(aqi)
        except (TypeError, ValueError):
            continue
        if aqi < 0:
            continue
        parsed.append((utc, aqi, row))

    if not parsed:
        return None

    # Only rows in the latest UTC hour count toward the observation.
    latest_hour = max(_hour_key(utc) for utc, _, _ in parsed)
    hour_rows = [(u, a, r) for u, a, r in parsed if _hour_key(u) == latest_hour]
    latest_utc = max(utc for utc, _, _ in hour_rows)

    max_aqi = -1
    max_row = None
    pollutant_readings: Dict[str, int] = {}
    for utc, aqi, row in hour_rows:
        parameter = (row.get("Parameter") or "").strip()
        pollutant_readings[parameter] = aqi
        if aqi > max_aqi:
            max_aqi = aqi
            max_row = row

    if max_row is None:
        return None

    cat = get_aqi_category(max_aqi)

    return {
        "source": "AirNow",
        "aqi": max_aqi,
        "primary_pollutant": (max_row.get("Parameter") or "PM2.5").strip(),
        "category": cat["label"],
        "category_color": cat["color"],
        "category_text_color": cat["textColor"],
        "category_description": cat["description"],
        "reporting_area": max_row.get("SiteName") or "Unknown Area",
        "pollutants": pollutant_readings,
        "as_of": latest_utc.isoformat(),
    }


def _row_concentration(row: Dict[str, Any]) -> Optional[float]:
    """
    Concentration from a raw AirNow row, or None when the row has no data.

    A -999/null ``Value`` falls back to the unvalidated ``RawConcentration``
    when present; rows with neither are dropped. Negative concentrations are
    clamped to 0.
    """
    value = row.get("Value")
    if value is None or _is_missing(value):
        value = row.get("RawConcentration")
    if value is None or _is_missing(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, value)


def _site_key(row: Dict[str, Any]) -> Optional[Any]:
    """Stable identity for a monitoring site: AQSID, else FullAQSCode, else coords."""
    aqsid = row.get("AQSID")
    if aqsid is not None:
        return ("aqsid", aqsid)
    full_code = row.get("FullAQSCode")
    if full_code is not None:
        return ("aqs_code", full_code)
    lat2, lon2 = row.get("Latitude"), row.get("Longitude")
    if lat2 is None or lon2 is None:
        return None
    try:
        return ("coords", float(lat2), float(lon2))
    except (TypeError, ValueError):
        return None


def _select_airnow_site(
    rows: List[Dict[str, Any]], lat: float, lon: float
) -> Optional[Dict[str, Any]]:
    """Nearest site with valid PM2.5 and PM10 at its latest UTC hour, or None."""
    sites: Dict[Any, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        parameter = (row.get("Parameter") or "").strip()
        if parameter not in ("PM2.5", "PM10"):
            continue
        value = _row_concentration(row)
        if value is None:
            continue
        utc = _parse_airnow_utc(row.get("UTC"))
        if utc is None:
            continue
        key = _site_key(row)
        if key is None:
            continue
        site = sites.get(key)
        if site is None:
            try:
                site_lat = float(row.get("Latitude"))
                site_lon = float(row.get("Longitude"))
            except (TypeError, ValueError):
                continue
            site = {
                "lat": site_lat,
                "lon": site_lon,
                "name": row.get("SiteName") or row.get("ReportingArea"),
                "aqsid": row.get("AQSID"),
                "pm25": None,
                "pm10": None,
            }
            sites[key] = site
        field = "pm25" if parameter == "PM2.5" else "pm10"
        current = site[field]
        if current is None or utc > current[1]:
            site[field] = (value, utc)

    best = None
    for site in sites.values():
        pm25_entry, pm10_entry = site["pm25"], site["pm10"]
        if pm25_entry is None or pm10_entry is None:
            continue
        distance_km = _haversine_km(lat, lon, site["lat"], site["lon"])
        if best is None or distance_km < best["distance_km"]:
            best = {
                "pm25": pm25_entry[0],
                "pm10": pm10_entry[0],
                "as_of": max(pm25_entry[1], pm10_entry[1]),
                "name": site["name"],
                "aqsid": site["aqsid"],
                "distance_km": distance_km,
            }

    if best is None:
        return None

    if best["distance_km"] > get_params().airnow_ratio_max_distance_km:
        # A distant monitor is not representative of a local plume: treat the
        # ratio as missing (honest "unknown") rather than reporting a far-site
        # ratio. The ceiling is a tunable active param.
        return None

    pm25, pm10 = best["pm25"], best["pm10"]
    return {
        "pm25": pm25,
        "pm10": pm10,
        "pm25_pm10_ratio": round(pm25 / pm10, 2) if pm10 > 0 else None,
        "site": {
            "name": best["name"],
            "aqsid": best["aqsid"],
            "distance_km": round(best["distance_km"], 2),
            "as_of": best["as_of"].isoformat(),
        },
    }


async def fetch_airnow_concentrations(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Recover the PM2.5/PM10 ratio from the nearest AirNow monitoring site.

    Queries the raw hourly ``/aq/data/`` endpoint over a 3h lookback inside a
    +/-0.25 deg BBOX and picks the nearest site with valid PM2.5 and PM10
    concentrations in its latest UTC hour. Returns None on any failure or when
    no qualifying site exists. Used as a fallback when OpenAQ's nearest monitor
    has PM2.5 but no co-located PM10.
    """
    if not AIRNOW_KEY:
        return None

    now = datetime.now(dt_timezone.utc)
    start = now - timedelta(hours=AIRNOW_LOOKBACK_HOURS)
    params = {
        "startDate": start.strftime("%Y-%m-%dT%H"),
        "endDate": now.strftime("%Y-%m-%dT%H"),
        "parameters": "PM25,PM10",
        "BBOX": (
            f"{lon - AIRNOW_BBOX_DEG},{lat - AIRNOW_BBOX_DEG},"
            f"{lon + AIRNOW_BBOX_DEG},{lat + AIRNOW_BBOX_DEG}"
        ),
        "dataType": "B",
        "format": "application/json",
        "verbose": 1,
        "monitorType": 0,
        "includerawconcentrations": 1,
        "API_KEY": AIRNOW_KEY,
    }
    try:
        async with httpx.AsyncClient(timeout=AIRNOW_TIMEOUT_S) as client:
            response = await client.get(AIRNOW_DATA_URL, params=params)
            if response.status_code != 200:
                return None
            data = response.json()
    except Exception:
        return None

    if not isinstance(data, list) or len(data) == 0:
        return None

    return _select_airnow_site(data, lat, lon)
