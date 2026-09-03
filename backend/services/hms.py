import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import time
import httpx

# NOAA Hazard Mapping System (HMS) Smoke Detection Feature Service.
# The legacy NESDIS pub/FIRE/HMS/GIS/GEOJSON static feed was decommissioned,
# so we query the current ArcGIS Feature Service with a server-side point
# intersection filter and only download the smoke polygons overhead.
HMS_URLS = [
    "https://services2.arcgis.com/C8EMgrsFcRFL6LrL/arcgis/rest/services/NOAA_Satellite_Smoke_Detection_(v1)/FeatureServer/0/query",
]

HMS_USER_AGENT = "UpwindAQI/1.0 (https://github.com; contact@upwind.app)"

# Density codes map to canonical labels
_DENSITY_MAP = {
    "5": "light", "light": "light",
    "16": "medium", "medium": "medium",
    "27": "heavy", "heavy": "heavy",
}

def point_in_polygon(x: float, y: float, poly: List[List[float]]) -> bool:
    """
    Ray-casting algorithm to test if point (x=lon, y=lat) is inside polygon coordinates.
    poly is a list of [lon, lat] pairs.
    """
    n = len(poly)
    inside = False
    p1x, p1y = poly[0][0], poly[0][1]
    for i in range(n + 1):
        p2x, p2y = poly[i % n][0], poly[i % n][1]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def _point_in_ring_set(x: float, y: float, rings: List[List[List[float]]]) -> bool:
    """
    True when (x, y) is inside the exterior ring (rings[0]) and outside every
    interior ring/hole (rings[1:]), per GeoJSON (RFC 7946 §3.1.6).
    """
    if not rings or not rings[0] or len(rings[0]) <= 2:
        return False
    if not point_in_polygon(x, y, rings[0]):
        return False
    return not any(point_in_polygon(x, y, hole) for hole in rings[1:] if hole and len(hole) > 2)


def _parse_hms_analysis_time(value: Any) -> Optional[datetime]:
    """
    Parse the ArcGIS smoke-analysis timestamp ('YYYYDDD HHMM', e.g. '2026232 1500'
    = 2026 day-of-year 232 at 15:00 UTC) into an aware UTC datetime. None when
    the value is missing or malformed.
    """
    if not value:
        return None
    m = re.match(r"^(\d{4})(\d{3}) (\d{2})(\d{2})$", str(value).strip())
    if not m:
        return None
    year, day_of_year, hour, minute = (int(g) for g in m.groups())
    if not (1 <= day_of_year <= 366 and 0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    try:
        return datetime(year, 1, 1, hour, minute, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1)
    except ValueError:
        return None


def check_hms_smoke_plume(lat: float, lon: float, geojson_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if (lon, lat) falls inside any HMS smoke polygon.
    Density categories: 5 = Light, 16 = Medium, 27 = Heavy (or text string 'Light', 'Medium', 'Heavy').
    """
    features = geojson_data.get("features", [])
    if not features:
        return {"status": "absent", "density": None, "details": "No HMS smoke features in feed"}

    # Most recent smoke-analysis time across the returned polygons, used as the
    # feed's as_of timestamp for staleness tracking.
    analysis_times = [
        dt
        for feature in features
        for dt in [_parse_hms_analysis_time((feature.get("properties") or {}).get("Start"))]
        if dt is not None
    ]
    as_of = max(analysis_times).isoformat() if analysis_times else None

    found_densities = []

    for feature in features:
        geom = feature.get("geometry", {})
        props = feature.get("properties", {})
        geom_type = geom.get("type", "")
        coords = geom.get("coordinates", [])

        density_val = props.get("Density", props.get("density", "unknown"))

        is_inside = False
        if geom_type == "Polygon":
            is_inside = _point_in_ring_set(lon, lat, coords)
        elif geom_type == "MultiPolygon":
            for poly in coords:
                if _point_in_ring_set(lon, lat, poly):
                    is_inside = True
                    break

        if is_inside:
            found_densities.append(str(density_val))

    if found_densities:
        # Determine the strongest density level matched
        dens_str = ", ".join(found_densities)
        levels = {_DENSITY_MAP.get(d.strip().lower()) for d in found_densities}
        density_label = "heavy" if "heavy" in levels else ("medium" if "medium" in levels else "light")
        result = {
            "status": "present",
            "density": density_label,
            "raw_density": dens_str,
            "details": f"Location is inside HMS overhead smoke plume ({density_label} density)",
        }
        if as_of is not None:
            result["as_of"] = as_of
        return result

    return {
        "status": "absent",
        "density": None,
        "details": "No overhead HMS smoke plume detected at this location"
    }

# Cache HMS query results per rounded location (30 minute TTL)
_HMS_CACHE_TTL_SECONDS = 30 * 60
_HMS_CACHE_MAX_ENTRIES = 256
_hms_cache: Dict[str, Any] = {}


def _hms_cache_key(lat: float, lon: float) -> str:
    # Round to ~1 km so tiny coordinate jitter reuses the cached plume set
    return f"{round(lat, 2)},{round(lon, 2)}"


async def _fetch_hms_geojson(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Query the HMS smoke polygons overhead for (lat, lon); None if unreachable."""
    key = _hms_cache_key(lat, lon)
    cached = _hms_cache.get(key)
    if cached is not None and (time.monotonic() - cached["fetched_at"]) < _HMS_CACHE_TTL_SECONDS:
        return cached["geojson"]

    params = {
        "where": "1=1",
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": 4326,
        "outFields": "Density,Start,End_",
        "f": "geojson",
        "resultRecordCount": 100,
    }

    async with httpx.AsyncClient(timeout=15.0, headers={"User-Agent": HMS_USER_AGENT}) as client:
        for url in HMS_URLS:
            try:
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    continue
                geojson = resp.json()
                # Treat ArcGIS error payloads as an outage rather than absence
                if not isinstance(geojson, dict) or "error" in geojson or "features" not in geojson:
                    continue
                _hms_cache[key] = {"fetched_at": time.monotonic(), "geojson": geojson}
                if len(_hms_cache) > _HMS_CACHE_MAX_ENTRIES:
                    _hms_cache.clear()
                return geojson
            except Exception:
                continue

    return None


async def fetch_hms_smoke(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch NOAA HMS smoke polygons overhead and evaluate status for lat/lon.
    The raw feed is never returned; only the plume status/density at the point.
    """
    geojson = await _fetch_hms_geojson(lat, lon)
    if geojson is None:
        return {
            "status": "unavailable",
            "density": None,
            "details": "NOAA HMS smoke feed unreachable or offline"
        }
    return check_hms_smoke_plume(lat, lon, geojson)
