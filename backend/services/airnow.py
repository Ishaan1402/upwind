from typing import Optional, Dict, Any, List
import httpx
from datetime import datetime
from backend.config import AIRNOW_KEY, get_aqi_category

AIRNOW_LATLON_URL = "https://www.airnowapi.org/aq/observation/latLong/current/"

async def fetch_airnow_observation(lat: float, lon: float, distance_miles: int = 25) -> Optional[Dict[str, Any]]:
    """
    Fetch current AQI observation from AirNow API for given lat/lon.
    Returns standardized observation dictionary or None if key is missing or request fails.
    """
    if not AIRNOW_KEY:
        return None

    params = {
        "format": "application/json",
        "latitude": lat,
        "longitude": lon,
        "distance": distance_miles,
        "API_KEY": AIRNOW_KEY
    }

    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            response = await client.get(AIRNOW_LATLON_URL, params=params)
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not isinstance(data, list) or len(data) == 0:
                return None

            # Find max AQI entry and collect all pollutant readings
            max_aqi = -1
            primary_pollutant = "PM2.5"
            pollutant_readings = {}
            reporting_area = "Unknown Area"

            for item in data:
                pollutant = item.get("ParameterName", "")
                aqi = item.get("AQI", -1)
                reporting_area = item.get("ReportingArea", reporting_area)
                
                if aqi >= 0:
                    pollutant_readings[pollutant] = aqi
                    if aqi > max_aqi:
                        max_aqi = aqi
                        primary_pollutant = pollutant

            if max_aqi < 0:
                return None

            cat = get_aqi_category(max_aqi)

            return {
                "source": "AirNow",
                "aqi": max_aqi,
                "primary_pollutant": primary_pollutant,
                "category": cat["label"],
                "category_color": cat["color"],
                "category_text_color": cat["textColor"],
                "category_description": cat["description"],
                "reporting_area": reporting_area,
                "pollutants": pollutant_readings,
                "as_of": datetime.utcnow().isoformat() + "Z"
            }
    except Exception:
        return None
