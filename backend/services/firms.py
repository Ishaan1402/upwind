import asyncio
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import httpx
from backend.config import FIRMS_MAP_KEY
from backend.engine.params import Params, get_params

# Active VIIRS NRT instruments. FIRMS's area/csv endpoint does NOT accept a
# comma-joined source list (it returns HTTP 400 "Invalid source"), so each
# instrument is queried with its own single-source request and the CSV payloads
# are merged (see fetch_firms_hotspots).
FIRMS_SOURCES = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")


def firms_search_radius_miles(
    wind_speed_mph: Optional[float] = None,
    params: Optional[Params] = None,
) -> float:
    """Search radius for FIRMS bbox - floored so calm conditions still cover nearby fires."""
    p = params if params is not None else get_params()
    return min(p.firms_max_radius_miles, max(p.firms_min_radius_miles, (wind_speed_mph or p.firms_default_wind_mph) * p.firms_radius_wind_factor))

def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
    """
    Calculate distance between two coordinates in kilometers and miles.
    """
    R_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    dist_km = R_km * c
    dist_miles = dist_km * 0.621371
    return dist_km, dist_miles

def calculate_bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Numeric compass bearing (0-360) from point 1 to point 2."""
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360

def bearing_degrees_to_compass(bearing_deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((bearing_deg + 22.5) / 45) % 8
    return dirs[idx]

def calculate_compass_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    return bearing_degrees_to_compass(calculate_bearing_degrees(lat1, lon1, lat2, lon2))

def angular_difference(deg_a: float, deg_b: float) -> float:
    """Smallest difference between two compass bearings, 0-180."""
    diff = abs(deg_a - deg_b) % 360
    return min(diff, 360 - diff)

def angular_upwind_factor(angular_diff_deg: float, upwind_bonus: float) -> float:
    """Graded upwind relevance multiplier.

    Replaces the old hard binary (``upwind_bonus if is_upwind else 1.0``) with a
    cosine decay over the angular difference from the true upwind bearing: an
    on-axis source keeps the full ``upwind_bonus``, a 45-degree-off source gets
    roughly ``1 + (bonus - 1) * cos(45)``, and anything >= 90 degrees off stays
    neutral at 1.0 (never below). The boolean ``is_upwind`` gate is computed
    separately and is unaffected by this factor.
    """
    return 1.0 + (upwind_bonus - 1.0) * max(math.cos(math.radians(angular_diff_deg)), 0.0)

def filter_upwind_hotspots(hotspots: List[Dict[str, Any]], wind_dir_deg: Optional[float]) -> List[Dict[str, Any]]:
    """
    Given hotspots (each with a 'bearing_deg' float field), return only those
    within ``upwind_sector_width_deg`` of the true upwind bearing.
    If wind_dir_deg is None (wind unknown), return all hotspots unfiltered.
    """
    if wind_dir_deg is None:
        return hotspots
    p = get_params()
    # Upwind bearing from target is wind_dir_deg itself
    upwind_target_deg = wind_dir_deg % 360
    return [h for h in hotspots if angular_difference(h["bearing_deg"], upwind_target_deg) <= p.upwind_sector_width_deg]

def build_upwind_bbox(lat: float, lon: float, wind_dir_deg: float, radius_miles: float = 100.0) -> Tuple[float, float, float, float]:
    """
    Build a bounding box around target location.
    Returns (west, south, east, north).
    """
    lat_delta = radius_miles / 69.0
    lon_delta = radius_miles / (69.0 * math.cos(math.radians(lat)))

    west = max(-180.0, lon - lon_delta)
    east = min(180.0, lon + lon_delta)
    south = max(-90.0, lat - lat_delta)
    north = min(90.0, lat + lat_delta)

    return (round(west, 3), round(south, 3), round(east, 3), round(north, 3))


def parse_firms_acq_datetime(acq_date: Optional[str], acq_time: Optional[str]) -> Optional[datetime]:
    """
    Parse NASA FIRMS acq_date (YYYY-MM-DD) + acq_time (HHMM UTC, leading zero
    optional, e.g. "940" == 09:40) into an aware UTC datetime. Returns None when
    either field is missing or malformed so callers can degrade gracefully.
    """
    if not acq_date or not acq_time:
        return None
    try:
        day = datetime.strptime(acq_date.strip(), "%Y-%m-%d")
    except ValueError:
        return None
    digits = acq_time.strip()
    if not digits.isdigit():
        return None
    if len(digits) == 4:
        hour, minute = int(digits[:2]), int(digits[2:])
    elif 1 <= len(digits) <= 3:
        # e.g. "940" -> 09:40, "9" -> 00:09
        hour = int(digits[:-2]) if len(digits) >= 3 else 0
        minute = int(digits[-2:])
    else:
        return None
    try:
        return datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc)
    except ValueError:
        return None


def normalize_firms_confidence(value: Optional[str]) -> Optional[str]:
    """Normalize FIRMS confidence labels, accepting both full words and the
    abbreviated NRT forms ("l"/"n"/"h"). Returns None when unknown/missing."""
    if value is None:
        return None
    v = value.strip().lower()
    aliases = {"l": "low", "n": "nominal", "h": "high"}
    return aliases.get(v, v if v in ("low", "nominal", "high") else None)


def parse_firms_csv_rows(csv_text: str) -> List[Dict[str, Any]]:
    """
    Parse NASA FIRMS area/csv response into hotspot dicts.
    Uses header names (not fixed column indices) so schema shifts don't silently drop all rows.
    """
    lines = csv_text.strip().split("\n")
    if len(lines) <= 1:
        return []

    header = [col.strip().lower() for col in lines[0].split(",")]
    try:
        lat_idx = header.index("latitude")
        lon_idx = header.index("longitude")
    except ValueError:
        # Fallback for unexpected formats
        lat_idx, lon_idx = 0, 1

    frp_idx = header.index("frp") if "frp" in header else None
    acq_date_idx = header.index("acq_date") if "acq_date" in header else None
    acq_time_idx = header.index("acq_time") if "acq_time" in header else None
    confidence_idx = header.index("confidence") if "confidence" in header else None
    satellite_idx = header.index("satellite") if "satellite" in header else None
    daynight_idx = header.index("daynight") if "daynight" in header else None

    def _cell(idx: Optional[int], parts: List[str]) -> Optional[str]:
        if idx is not None and idx < len(parts) and parts[idx]:
            return parts[idx].strip()
        return None

    hotspots: List[Dict[str, Any]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) <= max(lat_idx, lon_idx):
            continue
        try:
            h_lat = float(parts[lat_idx])
            h_lon = float(parts[lon_idx])
            frp = 0.0
            if frp_idx is not None and frp_idx < len(parts) and parts[frp_idx]:
                frp = float(parts[frp_idx])
            hotspot = {
                "lat": h_lat,
                "lon": h_lon,
                "frp": frp,
                "acq_date": _cell(acq_date_idx, parts),
                "acq_time": _cell(acq_time_idx, parts),
                "confidence": normalize_firms_confidence(_cell(confidence_idx, parts)),
                "satellite": _cell(satellite_idx, parts),
                "daynight": _cell(daynight_idx, parts),
            }
            hotspots.append(hotspot)
        except (ValueError, IndexError):
            continue

    return hotspots


def cluster_firms_hotspots(
    hotspots: List[Dict[str, Any]],
    target_lat: float,
    target_lon: float,
    cluster_radius_km: Optional[float] = None,
    upwind_target_deg: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Greedy centroid clustering over surviving FIRMS pixels (pure Python).

    Pixels within cluster_radius_km of an existing cluster centroid (or seed)
    merge into that cluster; stronger pixels (higher FRP) seed first so the
    dominant detection anchors each cluster. Cluster intensity uses the PEAK
    single-member FRP (the most intense single detection) so repeated overpasses
    don't double-count persistence as intensity; summed FRP is kept as
    informational. Confidence weight is the max member weight, and the detection
    count feeds a bounded persistence multiplier. Returns clusters sorted by
    relevance (desc) after dropping clusters whose summed FRP falls below
    ``firms_min_cluster_frp``.
    """
    if not hotspots:
        return []

    p = get_params()
    if cluster_radius_km is None:
        cluster_radius_km = p.firms_cluster_radius_km
    ordered = sorted(hotspots, key=lambda h: h["frp"], reverse=True)
    member_groups: List[List[Dict[str, Any]]] = []
    centroids: List[Tuple[float, float]] = []

    # Spatial grid so the neighbor search is ~O(1) instead of O(hotspots x
    # clusters): every cluster is keyed by its CURRENT centroid's cell, and a
    # hotspot only tests the 3x3 cell neighborhood around its own cell. Cell size
    # is 2x the merge radius (times a 1.1 margin) in degrees, so any cluster
    # centroid within cluster_radius_km of a hotspot is guaranteed to sit inside
    # that 3x3 window (worst case the hotspot and centroid sit at opposite
    # corners of diagonally adjacent cells, i.e. < 2x cell size = 2.2x radius
    # apart). Because the grid is re-keyed whenever the running-mean centroid
    # drifts into a new cell, the guarantee tracks the live centroid position.
    cell_size_km = 2.0 * cluster_radius_km * 1.1
    lat_cell = cell_size_km / 111.0
    # lon deg/km shrinks with |lat|; size lon cells from the extreme (highest
    # |lat|) hotspot so coverage holds at every latitude in the dataset.
    ref_lat = min(max(abs(h["lat"]) for h in hotspots), 89.0)
    lon_cell = cell_size_km / (111.0 * math.cos(math.radians(ref_lat)))
    grid: Dict[Tuple[int, int], List[int]] = {}

    def _cell_of(lat: float, lon: float) -> Tuple[int, int]:
        return (int(math.floor(lat / lat_cell)), int(math.floor(lon / lon_cell)))

    for h in ordered:
        h_lat = h["lat"]
        h_lon = h["lon"]
        cell = _cell_of(h_lat, h_lon)
        joined = False
        # Candidate cluster indices from the 3x3 neighborhood, tested in
        # creation order so the FIRST within radius still wins (identical to a
        # full linear scan: any in-range cluster is provably in this window).
        candidates: List[int] = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                bucket = grid.get((cell[0] + di, cell[1] + dj))
                if bucket:
                    candidates.extend(bucket)
        candidates.sort()
        for idx in candidates:
            clat, clon = centroids[idx]
            dist_km, _ = calculate_haversine_distance(clat, clon, h_lat, h_lon)
            if dist_km <= cluster_radius_km:
                member_groups[idx].append(h)
                n = len(member_groups[idx])
                new_centroid = (
                    (clat * (n - 1) + h_lat) / n,
                    (clon * (n - 1) + h_lon) / n,
                )
                centroids[idx] = new_centroid
                # Re-key the grid if the running-mean centroid drifted cells.
                new_cell = _cell_of(new_centroid[0], new_centroid[1])
                old_cell = _cell_of(clat, clon)
                if new_cell != old_cell:
                    grid[old_cell].remove(idx)
                    grid.setdefault(new_cell, []).append(idx)
                joined = True
                break
        if not joined:
            member_groups.append([h])
            centroids.append((h_lat, h_lon))
            grid.setdefault(cell, []).append(len(centroids) - 1)

    clusters = []
    for members in member_groups:
        n = len(members)
        clat = sum(m["lat"] for m in members) / n
        clon = sum(m["lon"] for m in members) / n
        frp = sum(m["frp"] for m in members)
        if frp < p.firms_min_cluster_frp:
            continue

        age_hours = min(m["age_hours"] for m in members)  # most recent detection drives recency
        is_upwind = any(m["is_upwind"] for m in members)
        best_member = max(members, key=lambda m: m.get("confidence_weight", 1.0))
        confidence_weight = best_member.get("confidence_weight", 1.0)
        peak_frp = max(m["frp"] for m in members)
        persistence = min(1.0 + p.firms_persistence_step * (n - 1), p.firms_persistence_cap)

        dist_km, dist_mi = calculate_haversine_distance(target_lat, target_lon, clat, clon)
        bearing_deg = calculate_bearing_degrees(target_lat, target_lon, clat, clon)
        recency = max(p.firms_recency_floor, 2 ** (-age_hours / p.firms_recency_half_life_hours))
        # Graded upwind multiplier (full bonus on-axis, cosine decay to 1.0 at
        # >= 90 deg off); the boolean is_upwind above is the unchanged gate.
        # Without a wind bearing (upwind_target_deg None) the angular term is
        # undefined, so keep the historical binary factor exactly - callers
        # that predate the angular term (accuracy reconstruction) depend on it.
        if upwind_target_deg is None:
            upwind = p.firms_upwind_bonus if is_upwind else 1.0
        else:
            upwind_angular_diff = angular_difference(bearing_deg, upwind_target_deg)
            upwind = angular_upwind_factor(upwind_angular_diff, p.firms_upwind_bonus)
        decay = 1.0 / (dist_mi + p.firms_relevance_eps_miles)

        cluster = {
            "lat": round(clat, 4),
            "lon": round(clon, 4),
            "frp": round(frp, 1),
            "peak_frp": round(peak_frp, 1),
            "detections": n,
            "age_hours": round(age_hours, 1),
            "distance_km": round(dist_km, 1),
            "distance_miles": round(dist_mi, 1),
            "bearing": bearing_degrees_to_compass(bearing_deg),
            "bearing_deg": round(bearing_deg, 1),
            "is_upwind": is_upwind,
            "confidence": best_member.get("confidence"),
            "confidence_weight": round(confidence_weight, 2),
            "pixels": [{"lat": m["lat"], "lon": m["lon"]} for m in members],
            "relevance": round(peak_frp * confidence_weight * persistence * upwind * decay * recency, 1),
        }
        clusters.append(cluster)

    clusters.sort(key=lambda c: c["relevance"], reverse=True)
    return clusters


async def fetch_firms_hotspots(
    target_lat: float,
    target_lon: float,
    wind_dir_deg: Optional[float] = None,
    wind_speed_mph: Optional[float] = None,
    reference_utc: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Query NASA FIRMS hotspots in target bbox.

    Prefer upwind-aligned hotspot clusters; if none are upwind, still return
    nearby clusters as weaker regional evidence (alignment='nearby'). Each
    detection is weighted by recency (exponential decay from reference_utc)
    and confidence (nominal/high scaled; low-confidence detections dropped),
    and spatially clustered so cluster intensity is peak member FRP with a
    bounded persistence count.
    """
    if not FIRMS_MAP_KEY:
        return {
            "status": "unavailable",
            "hotspots": [],
            "details": "NASA FIRMS API key not configured"
        }

    p = get_params()
    # Floor search radius at 75 mi to cover nearby fires under calm winds
    radius_miles = firms_search_radius_miles(wind_speed_mph)
    wind_dir = wind_dir_deg if wind_dir_deg is not None else 180.0
    west, south, east, north = build_upwind_bbox(target_lat, target_lon, wind_dir, radius_miles)

    # Query all active VIIRS NRT instruments across a 48 hour window. FIRMS
    # rejects comma-joined source lists, so issue ONE request per source
    # (concurrently) and merge the per-source CSV payloads below.
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async def _get_source(source: str) -> Optional[str]:
                url = (
                    f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/"
                    f"{source}/{west},{south},{east},{north}/2"
                )
                try:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        return None
                    return resp.text
                except Exception:
                    return None

            results = await asyncio.gather(
                *(_get_source(src) for src in FIRMS_SOURCES),
                return_exceptions=True,
            )
            texts = [t for t in results if isinstance(t, str)]
            if not texts:
                return {
                    "status": "unavailable",
                    "hotspots": [],
                    "details": "FIRMS all sources failed (HTTP errors)"
                }

            # Merge the successful per-source payloads, keeping each source's
            # header line only once.
            merged_lines: List[str] = []
            for text in texts:
                lines = text.strip().split("\n")
                if not lines or not lines[0].strip():
                    continue
                if merged_lines:
                    lines = lines[1:]
                merged_lines.extend(lines)
            text = "\n".join(merged_lines)

            # Treat unexpected response as an outage rather than absence
            first_line = text.strip().split("\n", 1)[0] if text.strip() else ""
            if "latitude" not in first_line.lower():
                return {
                    "status": "unavailable",
                    "hotspots": [],
                    "details": f"FIRMS API error response: {first_line[:80]}"
                }

            parsed = parse_firms_csv_rows(text)
            if not parsed:
                return {
                    "status": "absent",
                    "hotspots": [],
                    "count": 0,
                    "total_count": 0,
                    "nearest": None,
                    "alignment": None,
                    "details": "No active thermal hotspots detected nearby"
                }

            reference = reference_utc or datetime.now(timezone.utc)
            upwind_target_deg = wind_dir_deg % 360 if wind_dir_deg is not None else None

            # Build pixel records: age (recency), confidence weight, distance/bearing.
            pixels = []
            # Most recent acquisition datetime among the surviving detections,
            # used as the feed's as_of timestamp for staleness tracking.
            latest_acq_dt: Optional[datetime] = None
            for row in parsed:
                acq_dt = parse_firms_acq_datetime(row.get("acq_date"), row.get("acq_time"))
                # Missing/malformed acq fields degrade gracefully: treat as fresh.
                age_hours = 0.0 if acq_dt is None else max(0.0, (reference - acq_dt).total_seconds() / 3600.0)
                if age_hours > p.firms_max_age_hours:
                    continue
                if acq_dt is not None and (latest_acq_dt is None or acq_dt > latest_acq_dt):
                    latest_acq_dt = acq_dt

                confidence = row.get("confidence")
                # Unknown/missing confidence keeps a neutral weight of 1.0.
                confidence_weight = p.firms_confidence_weight.get(confidence, 1.0)
                if confidence_weight == 0.0:
                    continue  # "low" confidence detections (sun glint / false positives) dropped

                h_lat = row["lat"]
                h_lon = row["lon"]
                frp = row["frp"]

                dist_km, dist_mi = calculate_haversine_distance(target_lat, target_lon, h_lat, h_lon)
                bearing_deg = calculate_bearing_degrees(target_lat, target_lon, h_lat, h_lon)
                is_upwind = wind_dir_deg is None or angular_difference(bearing_deg, upwind_target_deg) <= p.upwind_sector_width_deg
                recency = max(p.firms_recency_floor, 2 ** (-age_hours / p.firms_recency_half_life_hours))
                # Graded upwind multiplier (full bonus on-axis, cosine decay to
                # 1.0 at >= 90 deg off); the boolean is_upwind gate is unchanged.
                upwind_angular_diff = 0.0 if wind_dir_deg is None else angular_difference(bearing_deg, upwind_target_deg)
                upwind = angular_upwind_factor(upwind_angular_diff, p.firms_upwind_bonus)
                decay = 1.0 / (dist_mi + p.firms_relevance_eps_miles)

                pixels.append({
                    "lat": h_lat,
                    "lon": h_lon,
                    "frp": frp,
                    "age_hours": round(age_hours, 1),
                    "confidence": confidence,
                    "confidence_weight": confidence_weight,
                    "distance_km": round(dist_km, 1),
                    "distance_miles": round(dist_mi, 1),
                    "bearing": bearing_degrees_to_compass(bearing_deg),
                    "bearing_deg": round(bearing_deg, 1),
                    "is_upwind": is_upwind,
                    "relevance": round(frp * confidence_weight * upwind * decay * recency, 1),
                })

            if not pixels:
                return {
                    "status": "absent",
                    "hotspots": [],
                    "count": 0,
                    "total_count": 0,
                    "nearest": None,
                    "alignment": None,
                    "details": "No valid hotspot coordinates found nearby"
                }

            # Rank raw pixels by per-pixel relevance (cap 40 for payload size).
            pixels.sort(key=lambda x: x["relevance"], reverse=True)

            # Spatial clustering: summed FRP per cluster + persistence (detections).
            clusters = cluster_firms_hotspots(pixels, target_lat, target_lon, upwind_target_deg=upwind_target_deg)
            if not clusters:
                return {
                    "status": "absent",
                    "hotspots": pixels[:40],
                    "count": 0,
                    "total_count": 0,
                    "nearest": None,
                    "alignment": None,
                    "details": "Detected hotspots are too weak to register (below FRP floor)"
                }

            upwind_clusters = [c for c in clusters if c["is_upwind"]]
            total_count = len(clusters)
            count = len(upwind_clusters)
            # 'nearest' is the top cluster by relevance across ALL clusters, so
            # a far larger downwind fire can claim the named slot instead of
            # being outranked by a weak upwind cluster; alignment and count
            # still report the upwind subset (smoke corroboration unchanged).
            nearest = clusters[0]

            if upwind_clusters:
                alignment = "upwind"
                details = (
                    f"{count} upwind hotspot cluster(s) found "
                    f"(strongest cluster overall {nearest['distance_miles']} mi {nearest['bearing']} "
                    f"(FRP {nearest['frp']:.0f} MW, {nearest['detections']} detections)); "
                    f"{total_count} total detected nearby"
                )
            else:
                # Nearby non-upwind hotspots
                alignment = "nearby"
                details = (
                    f"{total_count} hotspot cluster(s) within ~{int(radius_miles)} mi "
                    f"(strongest cluster {nearest['distance_miles']} mi {nearest['bearing']} "
                    f"(FRP {nearest['frp']:.0f} MW, {nearest['detections']} detections)), "
                    f"but none aligned upwind of current wind"
                )

            result = {
                "status": "present",
                "hotspots": pixels[:40],
                "clusters": clusters[:10],
                "count": count,
                "total_count": total_count,
                "nearest": nearest,
                "alignment": alignment,
                "details": details,
            }
            if latest_acq_dt is not None:
                result["as_of"] = latest_acq_dt.isoformat()
            return result

    except Exception as e:
        print(f"[FIRMS Service Error]: {e}")
        return {
            "status": "unavailable",
            "hotspots": [],
            "details": "Error reaching NASA FIRMS service"
        }
