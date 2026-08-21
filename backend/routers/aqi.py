import asyncio
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from backend.services.geocode import geocode_location
from backend.services.airnow import fetch_airnow_observation
from backend.services.openmeteo import fetch_openmeteo_aqi
from backend.services.coverage import coverage_for_location
from backend.config import OBSERVATION_TOKEN_MAX_AGE_SECONDS, OBSERVATION_TOKEN_SECRET
from backend.observation_token import sign_observation

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
        location_info = await asyncio.to_thread(geocode_location, f"{float(lat):.6f},{float(lon):.6f}")
    else:
        search_term = (zip if isinstance(zip, str) and zip.strip() else None) or \
                      (query if isinstance(query, str) and query.strip() else None) or \
                      "90210" # default Beverly Hills if blank
        location_info = await asyncio.to_thread(geocode_location, search_term)

    if not location_info:
        raise HTTPException(status_code=400, detail="Could not geocode location. Please provide a valid location or coordinates.")

    target_lat = location_info["lat"]
    target_lon = location_info["lon"]

    country_code = (location_info.get("country_code") or "").lower()
    use_airnow = country_code in ("", "us")

    # Fetch AirNow (primary, US only) and Open-Meteo (global fallback)
    airnow_res, openmeteo_res = await asyncio.gather(
        fetch_airnow_observation(target_lat, target_lon) if use_airnow else _empty(),
        fetch_openmeteo_aqi(target_lat, target_lon),
        return_exceptions=True,
    )

    observation = airnow_res if (isinstance(airnow_res, dict) and airnow_res) else (openmeteo_res if (isinstance(openmeteo_res, dict) and openmeteo_res) else None)

    if not observation:
        raise HTTPException(status_code=503, detail="Air quality data is currently unavailable for this location.")

    coverage = coverage_for_location(location_info, observation)
    return {
        "location": location_info,
        "observation": observation,
        "coverage": coverage,
        "observation_token": sign_observation(
            location_info,
            observation,
            OBSERVATION_TOKEN_SECRET,
            max_age_seconds=OBSERVATION_TOKEN_MAX_AGE_SECONDS,
        ),
    }


async def _empty():
    return None
