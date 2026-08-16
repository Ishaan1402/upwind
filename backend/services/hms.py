from typing import Dict, Any, List, Optional
import time
import httpx

# NOAA Hazard Mapping System (HMS) Smoke GeoJSON URLs
HMS_URLS = [
    "https://satepsanone.nesdis.noaa.gov/pub/FIRE/HMS/GIS/GEOJSON/hms_smoke_latest.json",
    "https://webapps.doughty.noaa.gov/hms/data/geojson/latest/hms_smoke_latest.geojson"
]

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

def check_hms_smoke_plume(lat: float, lon: float, geojson_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if (lon, lat) falls inside any HMS smoke polygon.
    Density categories: 5 = Light, 16 = Medium, 27 = Heavy (or text string 'Light', 'Medium', 'Heavy').
    """
    features = geojson_data.get("features", [])
    if not features:
        return {"status": "absent", "density": None, "details": "No HMS smoke features in feed"}

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
        return {
            "status": "present",
            "density": density_label,
            "raw_density": dens_str,
            "details": f"Location is inside HMS overhead smoke plume ({density_label} density)"
        }

    return {
        "status": "absent",
        "density": None,
        "details": "No overhead HMS smoke plume detected at this location"
    }

# Cache HMS GeoJSON response in memory (30 minute TTL)
_HMS_CACHE_TTL_SECONDS = 30 * 60
_hms_cache: Dict[str, Any] = {"fetched_at": 0.0, "geojson": None}


async def _fetch_hms_geojson() -> Optional[Dict[str, Any]]:
    """Download the HMS latest GeoJSON once per TTL window; None if unreachable."""
    now = time.monotonic()
    if _hms_cache["geojson"] is not None and (now - _hms_cache["fetched_at"]) < _HMS_CACHE_TTL_SECONDS:
        return _hms_cache["geojson"]

    async with httpx.AsyncClient(timeout=8.0) as client:
        for url in HMS_URLS:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    geojson = resp.json()
                    _hms_cache["fetched_at"] = time.monotonic()
                    _hms_cache["geojson"] = geojson
                    return geojson
            except Exception:
                continue

    return None


async def fetch_hms_smoke(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch NOAA HMS smoke GeoJSON and evaluate status for lat/lon.
    The raw feed is never returned; only the plume status/density at the point.
    """
    geojson = await _fetch_hms_geojson()
    if geojson is None:
        return {
            "status": "unavailable",
            "density": None,
            "details": "NOAA HMS smoke feed unreachable or offline"
        }
    return check_hms_smoke_plume(lat, lon, geojson)
