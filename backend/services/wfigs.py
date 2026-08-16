from typing import Dict, Any, List, Optional
import httpx

from backend.services.firms import (
    calculate_haversine_distance,
    calculate_bearing_degrees,
    bearing_degrees_to_compass,
    angular_difference,
    UPWIND_SECTOR_WIDTH_DEG,
)

# NIFC WFIGS current incident locations endpoint
WFIGS_URLS = [
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Incident_Locations_Current/FeatureServer/0/query",
]

WFIGS_MAX_RADIUS_MILES = 300.0
# Skip incidents above 90% containment
WFIGS_MAX_CONTAINMENT_PCT = 90.0
# Maximum records to retrieve per spatial query
WFIGS_RECORD_CAP = 200

WFIGS_OUT_FIELDS = (
    "IncidentName,IncidentSize,PercentContained,POOState,POOCounty,"
    "FireDiscoveryDateTime,ModifiedOnDateTime_dt,IrwinID,GlobalID"
)


def _parse_optional_float(value: Any) -> Optional[float]:
    """Parse a numeric ArcGIS field defensively: tolerates int/float, string
    numbers, and percentage strings like '95%'."""
    if value is None or value == "":
        return None
    try:
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None


def _select_nearest_wildfire(
    features: List[Dict[str, Any]],
    target_lat: float,
    target_lon: float,
    wind_dir_deg: Optional[float],
) -> List[Dict[str, Any]]:
    """
    Convert raw WFIGS features into incident dicts and keep only those that
    qualify as potential smoke sources: a named wildfire within
    WFIGS_MAX_RADIUS_MILES that is not >90% contained. Rows with missing or
    invalid coordinates are ignored. Distance/bearing are computed from the
    target (bearing = direction the incident sits from the target); upwind
    means within UPWIND_SECTOR_WIDTH_DEG of the wind's FROM direction.
    """
    incidents: List[Dict[str, Any]] = []
    for feature in features:
        props = feature.get("properties", {}) or {}
        geom = feature.get("geometry", {}) or {}
        coords = geom.get("coordinates")
        if geom.get("type") != "Point" or not coords or len(coords) < 2:
            continue
        try:
            fire_lon, fire_lat = float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            continue

        name = str(props.get("IncidentName") or "").strip()
        if not name:
            continue

        contained_pct = _parse_optional_float(props.get("PercentContained"))
        if contained_pct is not None and contained_pct >= WFIGS_MAX_CONTAINMENT_PCT:
            continue

        dist_km, dist_mi = calculate_haversine_distance(target_lat, target_lon, fire_lat, fire_lon)
        if dist_mi > WFIGS_MAX_RADIUS_MILES:
            continue

        bearing_deg = calculate_bearing_degrees(target_lat, target_lon, fire_lat, fire_lon)
        is_upwind = wind_dir_deg is None or (
            angular_difference(bearing_deg, wind_dir_deg % 360) <= UPWIND_SECTOR_WIDTH_DEG
        )

        state = str(props.get("POOState") or "")
        if state.upper().startswith("US-"):
            state = state[3:]

        incidents.append({
            "name": name,
            "size_acres": _parse_optional_float(props.get("IncidentSize")),
            "percent_contained": contained_pct,
            "state": state or None,
            "county": str(props.get("POOCounty") or "") or None,
            "discovery_ms": props.get("FireDiscoveryDateTime"),
            "modified_ms": props.get("ModifiedOnDateTime_dt"),
            "distance_km": round(dist_km, 1),
            "distance_miles": round(dist_mi, 1),
            "bearing": bearing_degrees_to_compass(bearing_deg),
            "bearing_deg": round(bearing_deg, 1),
            "is_upwind": is_upwind,
        })

    incidents.sort(key=lambda i: i["distance_miles"])
    return incidents


async def fetch_wfigs_incident(
    lat: float,
    lon: float,
    wind_dir_deg: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Query the NIFC WFIGS current-incident feed for the nearest active wildfire
    that could carry smoke toward the target. Never raises: feed errors return
    status "unavailable"; healthy query with no qualifying incidents return
    "absent".
    """
    params = {
        "f": "geojson",
        "where": "IncidentTypeCategory='WF'",
        "outFields": WFIGS_OUT_FIELDS,
        "outSR": 4326,
        # Server-side point distance filter within 300 miles
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "spatialRel": "esriSpatialRelIntersects",
        "distance": WFIGS_MAX_RADIUS_MILES,
        "units": "esriSRUnit_StatuteMile",
        "inSR": 4326,
        "orderByFields": "ModifiedOnDateTime_dt DESC",
        "resultRecordCount": WFIGS_RECORD_CAP,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        for url in WFIGS_URLS:
            try:
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                # Treat ArcGIS error payloads as feed outage rather than absence
                if not isinstance(data, dict) or "error" in data:
                    continue
                incidents = _select_nearest_wildfire(
                    data.get("features", []), lat, lon, wind_dir_deg
                )
                if not incidents:
                    return {
                        "status": "absent",
                        "incident": None,
                        "count": 0,
                        "alignment": None,
                        "details": "No active federal wildfire incidents within 300 mi",
                    }

                nearest = incidents[0]
                alignment = "upwind" if nearest["is_upwind"] else "nearby"
                size_txt = f"{nearest['size_acres']:,.0f} acres" if nearest["size_acres"] is not None else "size unknown"
                cont_txt = f"{nearest['percent_contained']:.0f}% contained" if nearest["percent_contained"] is not None else "containment unknown"
                return {
                    "status": "present",
                    "incident": nearest,
                    "count": len(incidents),
                    "alignment": alignment,
                    "details": (
                        f"Federal incident registry lists '{nearest['name']}' "
                        f"({size_txt}, {cont_txt}) "
                        f"{nearest['distance_miles']} mi {nearest['bearing']}"
                    ),
                }
            except Exception:
                continue

    return {
        "status": "unavailable",
        "incident": None,
        "count": 0,
        "alignment": None,
        "details": "NIFC WFIGS incident feed unreachable or offline",
    }
