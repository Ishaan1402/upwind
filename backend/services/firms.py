import math
from typing import Dict, Any, List, Optional, Tuple
import httpx
from backend.config import FIRMS_MAP_KEY

UPWIND_SECTOR_WIDTH_DEG = 90.0  # hotspots within +/-90 deg of the true upwind bearing count as "upwind"
# Floor at 75 mi so calm winds still catch fires a town/county over (Burns-class misses).
FIRMS_MIN_RADIUS_MILES = 75.0
FIRMS_MAX_RADIUS_MILES = 150.0


def firms_search_radius_miles(wind_speed_mph: Optional[float] = None) -> float:
    """Search radius for FIRMS bbox — floored so calm conditions still cover nearby fires."""
    return min(FIRMS_MAX_RADIUS_MILES, max(FIRMS_MIN_RADIUS_MILES, (wind_speed_mph or 10.0) * 5.0))

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

def filter_upwind_hotspots(hotspots: List[Dict[str, Any]], wind_dir_deg: Optional[float]) -> List[Dict[str, Any]]:
    """
    Given hotspots (each with a 'bearing_deg' float field), return only those
    within UPWIND_SECTOR_WIDTH_DEG of the true upwind bearing.
    If wind_dir_deg is None (wind unknown), return all hotspots unfiltered.
    """
    if wind_dir_deg is None:
        return hotspots
    upwind_target_deg = (wind_dir_deg + 180) % 360
    return [h for h in hotspots if angular_difference(h["bearing_deg"], upwind_target_deg) <= UPWIND_SECTOR_WIDTH_DEG]

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
            hotspots.append({"lat": h_lat, "lon": h_lon, "frp": frp})
        except (ValueError, IndexError):
            continue

    return hotspots


async def fetch_firms_hotspots(
    target_lat: float,
    target_lon: float,
    wind_dir_deg: Optional[float] = None,
    wind_speed_mph: Optional[float] = None
) -> Dict[str, Any]:
    """
    Query NASA FIRMS hotspots in target bbox.
    Prefer upwind-aligned hotspots; if none are upwind, still return nearby
    hotspots as weaker regional evidence (alignment='nearby').
    """
    if not FIRMS_MAP_KEY:
        return {
            "status": "unavailable",
            "hotspots": [],
            "details": "NASA FIRMS API key not configured"
        }

    # Floor at 75 mi so calm winds still catch fires a town/county over (Burns-class misses).
    radius_miles = firms_search_radius_miles(wind_speed_mph)
    wind_dir = wind_dir_deg if wind_dir_deg is not None else 180.0
    west, south, east, north = build_upwind_bbox(target_lat, target_lon, wind_dir, radius_miles)

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/VIIRS_SNPP_NRT/{west},{south},{east},{north}/1"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return {
                    "status": "unavailable",
                    "hotspots": [],
                    "details": f"FIRMS API HTTP {resp.status_code}"
                }

            parsed = parse_firms_csv_rows(resp.text)
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

            hotspots = []

            for row in parsed:
                h_lat = row["lat"]
                h_lon = row["lon"]
                frp = row["frp"]

                dist_km, dist_mi = calculate_haversine_distance(target_lat, target_lon, h_lat, h_lon)
                bearing_deg = calculate_bearing_degrees(target_lat, target_lon, h_lat, h_lon)
                bearing = bearing_degrees_to_compass(bearing_deg)

                hotspots.append({
                    "lat": h_lat,
                    "lon": h_lon,
                    "frp": frp,
                    "distance_km": round(dist_km, 1),
                    "distance_miles": round(dist_mi, 1),
                    "bearing": bearing,
                    "bearing_deg": round(bearing_deg, 1)
                })

            if not hotspots:
                return {
                    "status": "absent",
                    "hotspots": [],
                    "count": 0,
                    "total_count": 0,
                    "nearest": None,
                    "alignment": None,
                    "details": "No valid hotspot coordinates found nearby"
                }

            hotspots.sort(key=lambda x: x["distance_km"])

            upwind_target_deg = (wind_dir_deg + 180) % 360 if wind_dir_deg is not None else None
            for h in hotspots:
                h["is_upwind"] = wind_dir_deg is None or angular_difference(h["bearing_deg"], upwind_target_deg) <= UPWIND_SECTOR_WIDTH_DEG

            upwind_hotspots = filter_upwind_hotspots(hotspots, wind_dir_deg)
            total_count = len(hotspots)

            if upwind_hotspots:
                nearest = upwind_hotspots[0]
                return {
                    "status": "present",
                    "hotspots": hotspots[:10],
                    "count": len(upwind_hotspots),
                    "total_count": total_count,
                    "nearest": nearest,
                    "alignment": "upwind",
                    "details": (
                        f"{len(upwind_hotspots)} upwind hotspot cluster(s) found "
                        f"(nearest: {nearest['distance_miles']} mi {nearest['bearing']}); "
                        f"{total_count} total detected nearby"
                    )
                }

            # Nearby but not upwind — still useful regional fire evidence (weaker than upwind).
            nearest = hotspots[0]
            return {
                "status": "present",
                "hotspots": hotspots[:10],
                "count": total_count,
                "total_count": total_count,
                "nearest": nearest,
                "alignment": "nearby",
                "details": (
                    f"{total_count} hotspot(s) within ~{int(radius_miles)} mi "
                    f"(nearest: {nearest['distance_miles']} mi {nearest['bearing']}), "
                    f"but none aligned upwind of current wind"
                )
            }

    except Exception as e:
        print(f"[FIRMS Service Error]: {e}")
        return {
            "status": "unavailable",
            "hotspots": [],
            "details": "Error reaching NASA FIRMS service"
        }
