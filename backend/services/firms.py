import math
from typing import Dict, Any, List, Optional, Tuple
import httpx
from backend.config import FIRMS_MAP_KEY

UPWIND_SECTOR_WIDTH_DEG = 90.0  # hotspots within +/-90 deg of the true upwind bearing count as "upwind"

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

async def fetch_firms_hotspots(
    target_lat: float,
    target_lon: float,
    wind_dir_deg: Optional[float] = None,
    wind_speed_mph: Optional[float] = None
) -> Dict[str, Any]:
    """
    Query NASA FIRMS hotspots in target bbox and filter upwind-aligned hotspots.
    """
    if not FIRMS_MAP_KEY:
        return {
            "status": "unavailable",
            "hotspots": [],
            "details": "NASA FIRMS API key not configured"
        }

    radius_miles = min(150.0, max(30.0, (wind_speed_mph or 10.0) * 5.0))
    wind_dir = wind_dir_deg if wind_dir_deg is not None else 180.0
    west, south, east, north = build_upwind_bbox(target_lat, target_lon, wind_dir, radius_miles)

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/VIIRS_SNPP_NRT/{west},{south},{east},{north}/1"

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return {
                    "status": "unavailable",
                    "hotspots": [],
                    "details": f"FIRMS API HTTP {resp.status_code}"
                }

            lines = resp.text.strip().split("\n")
            if len(lines) <= 1:
                return {
                    "status": "absent",
                    "hotspots": [],
                    "count": 0,
                    "total_count": 0,
                    "nearest": None,
                    "details": "No active thermal hotspots detected upwind"
                }

            hotspots = []
            
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        h_lat = float(parts[0])
                        h_lon = float(parts[1])
                        frp = float(parts[5]) if len(parts) > 5 and parts[5] != '' else 0.0
                        
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
                    except Exception:
                        continue

            if not hotspots:
                return {
                    "status": "absent",
                    "hotspots": [],
                    "count": 0,
                    "total_count": 0,
                    "nearest": None,
                    "details": "No valid hotspot coordinates found in upwind area"
                }

            # Sort all hotspots by distance
            hotspots.sort(key=lambda x: x["distance_km"])
            
            # Tag each hotspot with is_upwind flag
            upwind_target_deg = (wind_dir_deg + 180) % 360 if wind_dir_deg is not None else None
            for h in hotspots:
                h["is_upwind"] = wind_dir_deg is None or angular_difference(h["bearing_deg"], upwind_target_deg) <= UPWIND_SECTOR_WIDTH_DEG

            upwind_hotspots = filter_upwind_hotspots(hotspots, wind_dir_deg)
            total_count = len(hotspots)

            if not upwind_hotspots:
                return {
                    "status": "absent",
                    "hotspots": hotspots[:10],
                    "count": 0,
                    "total_count": total_count,
                    "nearest": None,
                    "details": f"{total_count} hotspot(s) detected within radius, but none are upwind of this location" if total_count > 0 else "No active thermal hotspots detected upwind"
                }

            nearest = upwind_hotspots[0]
            return {
                "status": "present",
                "hotspots": hotspots[:10],
                "count": len(upwind_hotspots),
                "total_count": total_count,
                "nearest": nearest,
                "details": f"{len(upwind_hotspots)} upwind hotspot cluster(s) found (nearest: {nearest['distance_miles']} mi {nearest['bearing']}); {total_count} total detected nearby"
            }

    except Exception as e:
        print(f"[FIRMS Service Error]: {e}")
        return {
            "status": "unavailable",
            "hotspots": [],
            "details": "Error reaching NASA FIRMS service"
        }
