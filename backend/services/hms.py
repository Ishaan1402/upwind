from typing import Dict, Any, List, Optional
import httpx

# NOAA Hazard Mapping System (HMS) Smoke GeoJSON URLs
HMS_URLS = [
    "https://satepsanone.nesdis.noaa.gov/pub/FIRE/HMS/GIS/GEOJSON/hms_smoke_latest.json",
    "https://webapps.doughty.noaa.gov/hms/data/geojson/latest/hms_smoke_latest.geojson"
]

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
            # Outer ring is coords[0]
            if coords and len(coords[0]) > 2:
                if point_in_polygon(lon, lat, coords[0]):
                    is_inside = True
        elif geom_type == "MultiPolygon":
            for poly in coords:
                if poly and len(poly[0]) > 2:
                    if point_in_polygon(lon, lat, poly[0]):
                        is_inside = True
                        break

        if is_inside:
            found_densities.append(str(density_val))

    if found_densities:
        # Determine highest density matched
        dens_str = ", ".join(found_densities)
        is_heavy = any("27" in d or "heavy" in d.lower() for d in found_densities)
        is_med = any("16" in d or "med" in d.lower() for d in found_densities)
        
        density_label = "heavy" if is_heavy else ("medium" if is_med else "light")
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

async def fetch_hms_smoke(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch NOAA HMS smoke GeoJSON and evaluate status for lat/lon.
    """
    async with httpx.AsyncClient(timeout=8.0) as client:
        for url in HMS_URLS:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    geojson = resp.json()
                    res = check_hms_smoke_plume(lat, lon, geojson)
                    res["raw_geojson"] = geojson
                    return res
            except Exception:
                continue

    return {
        "status": "unavailable",
        "density": None,
        "details": "NOAA HMS smoke feed unreachable or offline"
    }
