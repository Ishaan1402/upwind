import re
from typing import Dict, Any, Optional
import pgeocode
from geopy.geocoders import Nominatim
from backend.db import get_cached_geocode, set_cached_geocode

_nom_geocodes = Nominatim(user_agent="upwind-aqi-why")
_nomi_us = pgeocode.Nominatim('us')

def geocode_location(query: str) -> Optional[Dict[str, Any]]:
    """
    Geocode an input query which can be:
    - 5-digit US ZIP code
    - 'lat,lon' string
    - City, State, or global location string
    Returns dict with lat, lon, name, zip_code.
    """
    query = query.strip()
    if not query:
        return None

    # Check lat,lon pattern
    latlon_match = re.match(r'^([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)$', query)
    if latlon_match:
        lat = float(latlon_match.group(1))
        lon = float(latlon_match.group(2))
        reverse = _reverse_geocode(lat, lon)
        return {
            "lat": lat,
            "lon": lon,
            "name": f"{lat:.4f}, {lon:.4f}",
            "zip_code": None,
            "state": reverse.get("state"),
            "city": reverse.get("city"),
            "country_code": reverse.get("country_code"),
            "country": reverse.get("country"),
        }

    # Check 5-digit US ZIP code
    zip_match = re.match(r'^\d{5}$', query)
    if zip_match:
        res = _nomi_us.query_postal_code(query)
        if res is not None and not res.empty and not math_isnan(res.latitude):
            city = str(res.place_name) if res.place_name and str(res.place_name) != 'nan' else ""
            state = str(res.state_code) if res.state_code and str(res.state_code) != 'nan' else ""
            name = f"{city}, {state} {query}".strip()
            return {
                "lat": float(res.latitude),
                "lon": float(res.longitude),
                "name": name,
                "zip_code": query,
                "state": state,
                "city": city,
                "country_code": "US",
                "country": "United States"
            }

    # General Nominatim lookup with 7-day SQLite caching
    query_norm = query.lower().strip()
    cached = get_cached_geocode(query_norm)
    if cached:
        return cached

    try:
        location = _nom_geocodes.geocode(query, timeout=5)
        if location:
            address = (location.raw or {}).get("address") or {}
            result = {
                "lat": float(location.latitude),
                "lon": float(location.longitude),
                "name": location.address,
                "zip_code": query if zip_match else None,
                "state": None,
                "city": None,
                "country_code": address.get("country_code"),
                "country": address.get("country"),
            }
            set_cached_geocode(query_norm, result)
            return result
    except Exception as e:
        print(f"[Nominatim Geocode Error]: {e}")
        pass

    return None


def _reverse_geocode(lat: float, lon: float) -> Dict[str, Any]:
    """Best-effort reverse geocode for direct lat/lon lookups (cached 7 days)."""
    query_key = f"reverse_{lat:.4f}_{lon:.4f}"
    cached = get_cached_geocode(query_key)
    if cached:
        return cached
    result = {
        "state": None,
        "city": None,
        "country_code": None,
        "country": None,
    }
    try:
        location = _nom_geocodes.reverse((lat, lon), timeout=5)
        if location:
            address = (location.raw or {}).get("address") or {}
            result.update({
                "state": address.get("state"),
                "city": address.get("city") or address.get("town") or address.get("village"),
                "country_code": address.get("country_code"),
                "country": address.get("country"),
            })
            set_cached_geocode(query_key, result)
    except Exception as e:
        print(f"[Nominatim Reverse Geocode Error]: {e}")
        pass
    return result

def math_isnan(val) -> bool:
    try:
        import math
        return math.isnan(val)
    except Exception:
        return False
