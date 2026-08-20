from typing import Optional, Dict, Any
import httpx
from datetime import datetime, timezone
from backend.config import get_aqi_category

OPENMETEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPENMETEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

async def fetch_openmeteo_aqi(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Fallback AQI fetch using Open-Meteo Air Quality API, including Aerosol Optical Depth (AOD).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "us_aqi,pm2_5,pm10,ozone,aerosol_optical_depth"
    }

    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            resp = await client.get(OPENMETEO_AQ_URL, params=params)
            if resp.status_code != 200:
                print(f"[Open-Meteo Service Error]: HTTP {resp.status_code} - {resp.text}")
                return None
            
            data = resp.json()
            current = data.get("current", {})
            
            us_aqi = current.get("us_aqi")
            if us_aqi is None:
                return None

            us_aqi = int(us_aqi)
            pm25 = float(current.get("pm2_5") or 0)
            pm10 = float(current.get("pm10") or 0)
            o3 = float(current.get("ozone") or 0)
            aod = current.get("aerosol_optical_depth")

            primary_pollutant = "PM2.5"
            if o3 > pm25 and o3 > 80:
                primary_pollutant = "O3"

            cat = get_aqi_category(us_aqi)

            return {
                "source": "Open-Meteo",
                "aqi": us_aqi,
                "primary_pollutant": primary_pollutant,
                "category": cat["label"],
                "category_color": cat["color"],
                "category_text_color": cat["textColor"],
                "category_description": cat["description"],
                "reporting_area": f"Coordinates ({lat:.2f}, {lon:.2f})",
                "pollutants": {
                    "PM2.5": round(pm25, 1) if pm25 > 0 else None,
                    "PM10": round(pm10, 1) if pm10 > 0 else None,
                    "OZONE": round(o3, 1) if o3 > 0 else None
                },
                "aerosol_optical_depth": round(float(aod), 2) if aod is not None else None,
                "as_of": datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        print(f"[Open-Meteo Service Error]: {e}")
        return None

async def fetch_openmeteo_weather(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Fetch current temperature (°F), 10m wind direction & speed (mph), 10m wind
    gust (mph), boundary layer height (m), and the past-30-day precipitation
    sum (inches) from Open-Meteo in one call.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,boundary_layer_height",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "daily": "precipitation_sum",
        "past_days": 30,
    }

    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            resp = await client.get(OPENMETEO_WEATHER_URL, params=params)
            if resp.status_code != 200:
                print(f"[Open-Meteo Weather Error]: HTTP {resp.status_code} - {resp.text}")
                return None
            
            data = resp.json()
            current = data.get("current", {})

            # Daily precipitation (mm) for today plus the past 30 days. The last
            # 30 entries are today + 29 prior days, matching the Lamar rule's
            # "30-day antecedent precipitation" window; missing days count as 0.
            daily = data.get("daily", {})
            precip_mm = daily.get("precipitation_sum")
            precip_30d_in = None
            if precip_mm:
                total_mm = sum(v for v in precip_mm[-30:] if v is not None)
                precip_30d_in = round(total_mm / 25.4, 2)

            return {
                "temperature_f": current.get("temperature_2m"),
                "wind_speed_mph": current.get("wind_speed_10m"),
                "wind_direction_deg": current.get("wind_direction_10m"),
                "wind_gust_mph": current.get("wind_gusts_10m"),
                "boundary_layer_height_m": current.get("boundary_layer_height"),
                "precip_30d_in": precip_30d_in,
            }
    except Exception as e:
        print(f"[Open-Meteo Weather Error]: {e}")
        return None
