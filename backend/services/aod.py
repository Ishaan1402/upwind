from datetime import datetime, timezone
from typing import Dict, Any, Optional
import httpx

OPENMETEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

def _current_time_as_of(current: Dict[str, Any]) -> Optional[str]:
    """
    Open-Meteo current.time (the CAMS-derived forecast's current valid time,
    ISO-8601 without a suffix; the API's default timezone is GMT) as a UTC
    ISO-8601 string. None when missing/unparseable, so as_of is only attached
    when the response exposed a real timestamp.
    """
    raw = current.get("time")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

def classify_aod(aod_value: float) -> Dict[str, Any]:
    """
    Classify Aerosol Optical Depth (AOD):
    - AOD >= 0.8: Heavy plume
    - AOD >= 0.4: Medium plume
    - AOD >= 0.2: Light haze
    - AOD < 0.2: Clear sky
    """
    if aod_value >= 0.8:
        density = "heavy"
        desc = "Dense atmospheric particle plume detected"
    elif aod_value >= 0.4:
        density = "medium"
        desc = "Moderate atmospheric particle plume detected"
    elif aod_value >= 0.2:
        density = "light"
        desc = "Light atmospheric haze detected"
    else:
        density = "clear"
        desc = "Clear atmospheric column"

    return {
        "status": "present" if aod_value >= 0.2 else "absent",
        "aod_value": round(aod_value, 2),
        "density": density,
        "details": f"{desc} (AOD {aod_value:.2f})"
    }

async def fetch_aod_signal(lat: float, lon: float, existing_aod: Optional[float] = None) -> Dict[str, Any]:
    """
    Fetch or evaluate Aerosol Optical Depth (AOD) signal for target location.
    """
    if existing_aod is not None:
        return classify_aod(existing_aod)

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "aerosol_optical_depth"
    }

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(OPENMETEO_AQ_URL, params=params)
            if resp.status_code == 200:
                data = resp.json()
                current = data.get("current", {})
                aod_val = current.get("aerosol_optical_depth")
                if aod_val is not None:
                    result = classify_aod(float(aod_val))
                    as_of = _current_time_as_of(current)
                    if as_of is not None:
                        result["as_of"] = as_of
                    return result
    except Exception as e:
        print(f"[AOD Service Error]: {e}")

    return {
        "status": "unavailable",
        "aod_value": None,
        "density": None,
        "details": "Aerosol Optical Depth (AOD) data unavailable"
    }
