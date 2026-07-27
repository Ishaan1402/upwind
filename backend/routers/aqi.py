from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from backend.services.geocode import geocode_location
from backend.services.airnow import fetch_airnow_observation
from backend.services.openmeteo import fetch_openmeteo_aqi

router = APIRouter(prefix="/api", tags=["AQI"])

@router.get("/aqi")
async def get_aqi(
    query: Optional[str] = Query(default=None, description="ZIP code, lat,lon string, or location name"),
    zip: Optional[str] = Query(default=None, description="5-digit US ZIP code"),
    lat: Optional[float] = Query(default=None, description="Latitude"),
    lon: Optional[float] = Query(default=None, description="Longitude")
):
    # Determine target location
    location_info = None

    if isinstance(lat, (float, int)) and isinstance(lon, (float, int)):
        location_info = {
            "lat": float(lat),
            "lon": float(lon),
            "name": f"{float(lat):.4f}, {float(lon):.4f}",
            "zip_code": zip if isinstance(zip, str) else None,
            "state": None,
            "city": None
        }
    else:
        search_term = (zip if isinstance(zip, str) and zip.strip() else None) or \
                      (query if isinstance(query, str) and query.strip() else None) or \
                      "90210" # default Beverly Hills if blank
        location_info = geocode_location(search_term)

    if not location_info:
        raise HTTPException(status_code=400, detail="Could not geocode location. Please provide a valid location or coordinates.")

    target_lat = location_info["lat"]
    target_lon = location_info["lon"]

    # 1. Try AirNow API
    observation = await fetch_airnow_observation(target_lat, target_lon)

    # 2. Fallback to Open-Meteo Air Quality
    if not observation:
        observation = await fetch_openmeteo_aqi(target_lat, target_lon)

    if not observation:
        raise HTTPException(status_code=503, detail="Air quality data is currently unavailable for this location.")

    return {
        "location": location_info,
        "observation": observation
    }
