from typing import Dict, Any, List, Optional
import httpx

from backend.services.firms import (
    calculate_haversine_distance,
    calculate_bearing_degrees,
    bearing_degrees_to_compass,
    angular_difference,
    UPWIND_SECTOR_WIDTH_DEG,
)
from backend.engine.params import (
    WFIGS_ACTIVITY_FLOOR,
    WFIGS_DEFAULT_SIZE_ACRES,
    WFIGS_MAX_CONTAINMENT_PCT,
    WFIGS_MAX_RADIUS_MILES,
    WFIGS_RELEVANCE_EPS_MILES,
    WFIGS_UPWIND_BONUS,
)

# NIFC WFIGS current incident locations endpoint
WFIGS_URLS = [
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Incident_Locations_Current/FeatureServer/0/query",
]

# Maximum records to retrieve per spatial query
WFIGS_RECORD_CAP = 200

WFIGS_OUT_FIELDS = (
    "IncidentName,IncidentSize,PercentContained,POOState,POOCounty,"
    "FireDiscoveryDateTime,ModifiedOnDateTime_dt,IrwinID,GlobalID,"
    "IsCpxChild,CpxName,CpxID"
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


def _parse_optional_bool(value: Any) -> bool:
    """Parse an ArcGIS boolean field defensively: ArcGIS may return real JSON
    booleans or string forms like 'true'/'false'."""
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    return str(value).strip().lower() in ("true", "1", "yes")


def _complex_group_key(incident: Dict[str, Any]) -> Optional[str]:
    """Group key for fire-complex collapse: a shared CpxID, or - when the
    child has no id but is flagged as a complex child with a name - the shared
    CpxName. Lone (non-complex) incidents return None and pass through."""
    if incident.get("cpx_id"):
        return f"id:{incident['cpx_id']}"
    if incident.get("is_cpx_child") and incident.get("cpx_name"):
        return f"name:{incident['cpx_name']}"
    return None


def _merge_complex_children(children: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge WFIGS complex children into one representative incident dict.

    The representative takes its name from the complex name when present (else
    the largest child's name), size = SUM of child sizes, containment = MIN of
    child containment (a complex is only as contained as its least-contained
    child), distance/bearing from the nearest child point, and is_upwind True
    if any child is upwind. All other fields come from the nearest child."""
    nearest = min(children, key=lambda c: c["distance_miles"])
    sizes = [c["size_acres"] for c in children if c["size_acres"] is not None]
    containments = [
        c["percent_contained"] for c in children if c["percent_contained"] is not None
    ]

    cpx_name = next((c["cpx_name"] for c in children if c.get("cpx_name")), None)
    if cpx_name:
        name = cpx_name
    else:
        name = max(children, key=lambda c: c["size_acres"] or 0)["name"]

    merged = dict(nearest)
    merged["name"] = name
    merged["size_acres"] = sum(sizes) if sizes else None
    merged["percent_contained"] = min(containments) if containments else None
    merged["is_upwind"] = any(c["is_upwind"] for c in children)
    merged["is_cpx_child"] = True
    return merged


def _collapse_complexes(incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse fire-complex children into a single representative incident each.

    Children that share the same non-empty cpx_id (or, when the id is missing
    but the incident is flagged as a complex child with a name, the same
    cpx_name) merge into one entry via _merge_complex_children. Lone
    (non-complex) incidents pass through unchanged."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for incident in incidents:
        key = _complex_group_key(incident)
        if key is not None:
            grouped.setdefault(key, []).append(incident)

    collapsed = [i for i in incidents if _complex_group_key(i) is None]
    collapsed.extend(_merge_complex_children(group) for group in grouped.values())
    return collapsed


def _select_relevant_wildfires(
    features: List[Dict[str, Any]],
    target_lat: float,
    target_lon: float,
    wind_dir_deg: Optional[float],
) -> List[Dict[str, Any]]:
    """
    Convert raw WFIGS features into incident dicts, collapse fire complexes,
    and rank the survivors by smoke relevance (size x activity x upwind
    alignment x distance decay) so the largest, most-active, upwind fire wins
    instead of simply the nearest one.

    Only named wildfires within WFIGS_MAX_RADIUS_MILES that are not >90%
    contained qualify. Rows with missing or invalid coordinates are ignored.
    Distance/bearing are computed from the target (bearing = direction the
    incident sits from the target); upwind means within
    UPWIND_SECTOR_WIDTH_DEG of the wind's FROM direction. Each returned
    incident carries a rounded `relevance` score.
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
            "cpx_id": str(props.get("CpxID") or "").strip() or None,
            "cpx_name": str(props.get("CpxName") or "").strip() or None,
            "is_cpx_child": _parse_optional_bool(props.get("IsCpxChild")),
        })

    incidents = _collapse_complexes(incidents)

    for incident in incidents:
        activity = max(
            1.0 - (incident["percent_contained"] or 0.0) / 100.0,
            WFIGS_ACTIVITY_FLOOR,
        )
        size = (
            incident["size_acres"]
            if incident["size_acres"] is not None
            else WFIGS_DEFAULT_SIZE_ACRES
        )
        upwind = WFIGS_UPWIND_BONUS if incident["is_upwind"] else 1.0
        decay = 1.0 / (incident["distance_miles"] + WFIGS_RELEVANCE_EPS_MILES)
        incident["relevance"] = round(size * activity * upwind * decay, 1)

    incidents.sort(key=lambda i: i["relevance"], reverse=True)
    return incidents


async def fetch_wfigs_incident(
    lat: float,
    lon: float,
    wind_dir_deg: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Query the NIFC WFIGS current-incident feed for the most relevant active
    wildfire that could carry smoke toward the target (largest, least-contained,
    upwind fire rather than merely the nearest). Never raises: feed errors
    return status "unavailable"; healthy query with no qualifying incidents
    return "absent".
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
                incidents = _select_relevant_wildfires(
                    data.get("features", []), lat, lon, wind_dir_deg
                )
                if not incidents:
                    return {
                        "status": "absent",
                        "incident": None,
                        "count": 0,
                        "alignment": None,
                        "candidates": [],
                        "details": "No active federal wildfire incidents within 300 mi",
                    }

                top = incidents[0]
                alignment = "upwind" if top["is_upwind"] else "nearby"
                size_txt = f"{top['size_acres']:,.0f} acres" if top["size_acres"] is not None else "size unknown"
                cont_txt = f"{top['percent_contained']:.0f}% contained" if top["percent_contained"] is not None else "containment unknown"
                return {
                    "status": "present",
                    "incident": top,
                    "count": len(incidents),
                    "alignment": alignment,
                    "candidates": incidents[:3],
                    "details": (
                        f"Federal incident registry lists '{top['name']}' "
                        f"({size_txt}, {cont_txt}) "
                        f"{top['distance_miles']} mi {top['bearing']}"
                    ),
                }
            except Exception:
                continue

    return {
        "status": "unavailable",
        "incident": None,
        "count": 0,
        "alignment": None,
        "candidates": [],
        "details": "NIFC WFIGS incident feed unreachable or offline",
    }
