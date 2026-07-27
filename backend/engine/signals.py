import time
from typing import Dict, Any, List, Tuple
from backend.services.openmeteo import fetch_openmeteo_weather
from backend.services.aod import fetch_aod_signal
from backend.services.firms import fetch_firms_hotspots
from backend.services.incident_search import search_fire_incident_name

TOOL_STEPS = {
    "weather_vector": "Calculating Open-Meteo wind trajectory & temperature",
    "aod_density": "Reading CAMS Aerosol Optical Depth (AOD) column density",
    "firms_scan": "Scanning NASA FIRMS thermal hotspot clusters upwind",
    "web_search": "Searching public news & active incident feeds (Web Search)",
    "score_hypotheses": "Scoring attribution hypotheses (Evidence Matrix)"
}

def create_trace_step(step: str, duration_ms: float, status: str = "done") -> Dict[str, Any]:
    return {
        "step": step,
        "label": TOOL_STEPS.get(step, step),
        "duration_ms": max(0.1, round(duration_ms, 1)),
        "status": status
    }

async def assemble_evidence_signals(
    location: Dict[str, Any],
    observation: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Assemble evidence signals package for a given location and observation.
    Measures exact real execution timing for each backend tool step.
    Returns (signals, execution_trace).
    """
    lat = location["lat"]
    lon = location["lon"]
    state = location.get("state")
    city = location.get("city")

    execution_trace = []

    # Step 1: Open-Meteo Weather (Wind, Temp & Boundary Layer Height)
    t0 = time.perf_counter()
    weather = await fetch_openmeteo_weather(lat, lon)
    t1 = time.perf_counter()
    execution_trace.append(create_trace_step("weather_vector", (t1 - t0) * 1000, "done" if weather else "warning"))

    wind_speed = weather.get("wind_speed_mph") if weather else None
    wind_dir = weather.get("wind_direction_deg") if weather else None
    temp_f = weather.get("temperature_f") if weather else None
    boundary_layer_height_m = weather.get("boundary_layer_height_m") if weather else None

    # Step 2: CAMS Aerosol Optical Depth (AOD)
    t0 = time.perf_counter()
    existing_aod = observation.get("aerosol_optical_depth")
    aod_res = await fetch_aod_signal(lat, lon, existing_aod)
    t1 = time.perf_counter()
    execution_trace.append(create_trace_step("aod_density", (t1 - t0) * 1000, aod_res.get("status", "done")))

    # Step 3: NASA FIRMS Upwind Hotspots
    t0 = time.perf_counter()
    firms_res = await fetch_firms_hotspots(lat, lon, wind_dir, wind_speed)
    t1 = time.perf_counter()
    execution_trace.append(create_trace_step("firms_scan", (t1 - t0) * 1000, firms_res.get("status", "done")))

    # Extract observation fields
    aqi_val = observation.get("aqi", 0)
    primary_pollutant = observation.get("primary_pollutant", "").upper()
    pollutants = observation.get("pollutants", {})

    pm25_val = pollutants.get("PM2.5") or pollutants.get("PM25") or 0
    pm10_val = pollutants.get("PM10") or 0
    o3_val = pollutants.get("O3") or pollutants.get("OZONE") or 0

    is_pm_primary = "PM" in primary_pollutant
    is_pm10_primary = "PM10" in primary_pollutant
    is_pm25_primary = is_pm_primary and not is_pm10_primary
    is_pm_elevated = aqi_val > 50 and is_pm_primary

    # Step 4: Web Search for active fire incidents
    t0 = time.perf_counter()
    incident_name = None
    if is_pm_elevated or firms_res["status"] == "present" or aod_res["status"] == "present":
        incident_name = await search_fire_incident_name(state, city, lat, lon)
    t1 = time.perf_counter()

    execution_trace.append(create_trace_step("web_search", (t1 - t0) * 1000, "done" if incident_name else "absent"))

    signals = []

    # Signal 1: Aerosol Optical Depth Plume
    signals.append({
        "id": "aerosol_plume",
        "label": "Atmospheric Column Particle Density (AOD)",
        "status": aod_res["status"],
        "density": aod_res.get("density"),
        "aod_value": aod_res.get("aod_value"),
        "details": aod_res.get("details", "")
    })

    # Signal 2: FIRMS Upwind Hotspots
    firms_status = firms_res["status"]
    firms_details = firms_res.get("details", "")
    aod_value = float(aod_res.get("aod_value") or 0.0)
    firms_present = firms_status == "present"
    has_corroboration = firms_present or (aod_res.get("status") == "present" and aod_value >= 0.2)

    signals.append({
        "id": "firms_upwind",
        "label": "Upwind Thermal Hotspots (NASA FIRMS)",
        "status": firms_status,
        "count": firms_res.get("count", 0),
        "total_count": firms_res.get("total_count", 0),
        "nearest": firms_res.get("nearest"),
        "hotspots": firms_res.get("hotspots", []),
        "alignment": firms_res.get("alignment"),
        "incident_name": incident_name if has_corroboration else None,
        "unverified_news_incident": incident_name if not has_corroboration else None,
        "details": firms_details
    })

    # Signal 3: Wind Field & Boundary Layer Height
    if weather and wind_speed is not None and wind_dir is not None:
        from backend.services.firms import calculate_compass_bearing
        upwind_dir = (wind_dir + 180) % 360
        
        signals.append({
            "id": "wind",
            "label": "Surface Wind Vector",
            "status": "present",
            "speed_mph": wind_speed,
            "direction_deg": wind_dir,
            "upwind_deg": round(upwind_dir, 1),
            "boundary_layer_height_m": boundary_layer_height_m,
            "details": f"{wind_speed} mph from {wind_dir}°"
        })
    else:
        signals.append({
            "id": "wind",
            "label": "Surface Wind Vector",
            "status": "unavailable",
            "speed_mph": None,
            "direction_deg": None,
            "upwind_deg": None,
            "boundary_layer_height_m": None,
            "details": "Wind vector data unavailable"
        })

    # Signal 4: Surface PM Level
    signals.append({
        "id": "surface_pm_level",
        "label": "Surface Particulate Matter (PM2.5 / PM10)",
        "status": "present" if (pm25_val > 0 or is_pm_primary) else "absent",
        "primary": is_pm_primary,
        "pm10_primary": is_pm10_primary,
        "pm25_primary": is_pm25_primary,
        "elevated": is_pm_elevated,
        "pm25_value": pm25_val,
        "details": f"Primary pollutant: {primary_pollutant} (AQI {aqi_val})"
    })

    # Signal 5: Ozone & Heat
    is_o3_primary = "O3" in primary_pollutant or "OZONE" in primary_pollutant
    is_hot = (temp_f is not None) and (temp_f >= 85.0)
    
    signals.append({
        "id": "ozone_heat",
        "label": "Ozone & Atmospheric Heat",
        "status": "present" if (is_o3_primary or is_hot) else "absent",
        "primary": is_o3_primary,
        "hot_day": is_hot,
        "temperature_f": temp_f,
        "details": f"O3 Primary: {is_o3_primary}, Temperature: {f'{temp_f}°F' if temp_f else 'N/A'}"
    })

    return signals, execution_trace
