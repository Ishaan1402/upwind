import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from backend.services.openmeteo import fetch_openmeteo_weather
from backend.services.aod import fetch_aod_signal
from backend.services.firms import fetch_firms_hotspots
from backend.services.incident_search import search_fire_incident_name
from backend.services.openaq import (
    discover_reference_monitors,
    fetch_latest,
    fetch_location_sensors,
    fetch_daily_baseline,
    fetch_same_hour_baseline,
    sensor_id_for_parameter,
    monitor_source_label,
    OPENAQ_RADIUS_M,
)

TOOL_STEPS = {
    "weather_vector": "Calculating Open-Meteo wind trajectory & temperature",
    "aod_density": "Reading CAMS Aerosol Optical Depth (AOD) column density",
    "firms_scan": "Scanning NASA FIRMS thermal hotspot clusters upwind",
    "web_search": "Searching public news & active incident feeds (Web Search)",
    "openaq_monitors": "Reading local monitor concentrations (OpenAQ)",
    "score_hypotheses": "Scoring attribution hypotheses (Evidence Matrix)"
}

def create_trace_step(step: str, duration_ms: float, status: str = "done") -> Dict[str, Any]:
    return {
        "step": step,
        "label": TOOL_STEPS.get(step, step),
        "duration_ms": max(0.1, round(duration_ms, 1)),
        "status": status
    }


def _openaq_unavailable_signal(detail: str) -> Dict[str, Any]:
    return {
        "id": "openaq_concentrations",
        "label": "Local Monitor Concentrations (OpenAQ)",
        "status": "unavailable",
        "details": detail,
    }


async def collect_openaq_signal(
    lat: float, lon: float, include_baselines: bool = True
) -> Dict[str, Any]:
    """
    Gather OpenAQ reference-monitor concentrations for attribution.

    Only fresh (<=3h), US reference-monitor readings are included; anything
    else results in an "unavailable" signal so scoring behaves as before.
    Baseline percentiles are skipped when include_baselines is False (Good-AQI
    briefings never use them, so the extra calls are wasted latency).
    Never raises.
    """
    try:
        candidates = await discover_reference_monitors(lat, lon, limit=3)
        if not candidates:
            return _openaq_unavailable_signal(
                "No air quality monitor within 25 km (or OpenAQ key not configured)"
            )

        # Fetch fresh readings for all nearby candidates concurrently.
        latest_by_id: Dict[Any, Dict[str, Any]] = {}
        results = await asyncio.gather(
            *(fetch_latest(c["location_id"]) for c in candidates),
            return_exceptions=True,
        )
        for candidate, candidate_readings in zip(candidates, results):
            if isinstance(candidate_readings, dict) and candidate_readings:
                latest_by_id[candidate["location_id"]] = candidate_readings

        # A dead feed on the nearest monitor must not block a live one further
        # out: widen to the max radius when nothing nearby has fresh readings.
        if not latest_by_id:
            wider = await discover_reference_monitors(lat, lon, limit=10, radius_m=OPENAQ_RADIUS_M)
            if wider:
                new_ids = [c["location_id"] for c in candidates]
                wider_results = await asyncio.gather(
                    *(fetch_latest(c["location_id"]) for c in wider if c["location_id"] not in new_ids),
                    return_exceptions=True,
                )
                wider = [c for c in wider if c["location_id"] not in new_ids]
                for candidate, candidate_readings in zip(wider, wider_results):
                    if isinstance(candidate_readings, dict) and candidate_readings:
                        latest_by_id[candidate["location_id"]] = candidate_readings
                candidates = candidates + wider

        # Freshness-aware selection among candidates (nearest-first):
        # 1) nearest with BOTH live PM2.5 and PM10 (enables the dust/smoke ratio),
        # 2) else nearest with live PM2.5,
        # 3) else nearest with any fresh readings.
        monitor = None
        readings: Dict[str, Any] = {}
        pm25_monitor, pm25_readings = None, {}
        fallback_monitor, fallback_readings = None, {}
        for candidate in candidates:
            candidate_readings = latest_by_id.get(candidate["location_id"])
            if not candidate_readings:
                continue
            if fallback_monitor is None:
                fallback_monitor, fallback_readings = candidate, candidate_readings
            if "pm25" in candidate_readings and "pm10" in candidate_readings:
                monitor, readings = candidate, candidate_readings
                break
            if "pm25" in candidate_readings and pm25_monitor is None:
                pm25_monitor, pm25_readings = candidate, candidate_readings
        if monitor is None and pm25_monitor is not None:
            monitor, readings = pm25_monitor, pm25_readings
        if monitor is None:
            monitor, readings = fallback_monitor, fallback_readings
        if monitor is None:
            return _openaq_unavailable_signal("No fresh readings from nearby air quality monitors")

        location_id = monitor["location_id"]

        pm25 = readings.get("pm25")
        pm10 = readings.get("pm10")
        as_of = pm25["as_of"] if pm25 else next(iter(readings.values()))["as_of"]

        signal: Dict[str, Any] = {
            "id": "openaq_concentrations",
            "label": "Local Monitor Concentrations (OpenAQ)",
            "status": "present",
            "pm25": pm25["value"] if pm25 else None,
            "pm10": pm10["value"] if pm10 else None,
            "o3_ppb": None,
            "no2_ppb": None,
            "co_ppm": None,
            "so2_ppb": None,
            "pm25_pm10_ratio": None,
            "monitor": {
                "location_id": monitor.get("location_id"),
                "name": monitor.get("name"),
                "distance_km": monitor.get("distance_km"),
                "provider": monitor.get("provider"),
                "owner": monitor.get("owner"),
            },
            "as_of": as_of,
            "daily_percentile": None,
            "same_hour_percentile": None,
            "same_hour_median": None,
            "details": (
                f"Nearest {monitor_source_label(monitor)} with live readings: {monitor.get('name')} "
                f"({monitor.get('distance_km')} km away)"
            ),
        }

        if pm25 and pm10 and pm10["value"] > 0:
            signal["pm25_pm10_ratio"] = round(pm25["value"] / pm10["value"], 2)

        for key, param in (("o3_ppb", "o3"), ("no2_ppb", "no2"), ("co_ppm", "co"), ("so2_ppb", "so2")):
            reading = readings.get(param)
            if reading:
                signal[key] = reading["value"]

        sensors_map = await fetch_location_sensors(location_id)
        pm25_sensor_id = sensor_id_for_parameter(sensors_map, "pm25")
        if include_baselines and pm25_sensor_id is not None and pm25:
            daily = await fetch_daily_baseline(pm25_sensor_id)
            if daily:
                signal["daily_percentile"] = daily["percentile"]
            same_hour = await fetch_same_hour_baseline(
                pm25_sensor_id, monitor.get("timezone"), pm25["value"]
            )
            if same_hour:
                signal["same_hour_percentile"] = same_hour["percentile"]
                signal["same_hour_median"] = same_hour["median"]

        return signal
    except Exception as e:
        print(f"[OpenAQ Service Error]: {e}")
        return _openaq_unavailable_signal("OpenAQ concentration feed unavailable")


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

    # Step 4: Web Search for active fire incidents
    t0 = time.perf_counter()
    incident_name = None
    primary_pollutant = observation.get("primary_pollutant", "").upper()
    is_pm_elevated = observation.get("aqi", 0) > 50 and "PM" in primary_pollutant
    if is_pm_elevated or firms_res["status"] == "present" or aod_res["status"] == "present":
        incident_name = await search_fire_incident_name(state, city, lat, lon)
    t1 = time.perf_counter()

    execution_trace.append(create_trace_step("web_search", (t1 - t0) * 1000, "done" if incident_name else "absent"))

    # Step 5: OpenAQ Reference Monitor Concentrations
    t0 = time.perf_counter()
    openaq_sig = await collect_openaq_signal(
        lat, lon, include_baselines=observation.get("aqi", 0) > 50
    )
    t1 = time.perf_counter()
    execution_trace.append(create_trace_step(
        "openaq_monitors",
        (t1 - t0) * 1000,
        "done" if openaq_sig.get("status") == "present" else "warning"
    ))

    signals = build_evidence_signals(
        observation, weather, aod_res, firms_res, incident_name, openaq_sig
    )

    return signals, execution_trace


def build_evidence_signals(
    observation: Dict[str, Any],
    weather: Optional[Dict[str, Any]],
    aod_res: Dict[str, Any],
    firms_res: Dict[str, Any],
    incident_name: Optional[str],
    openaq_sig: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build the shared evidence signals list for both /api/why paths.

    The non-streaming and streaming (SSE) endpoints must present identical
    evidence to scoring and the LLM. Incident names are passed through raw;
    scoring decides whether they are corroborated by hotspots (fire vote) or
    become unverified-news open questions.
    """
    aqi_val = observation.get("aqi", 0)
    primary_pollutant = observation.get("primary_pollutant", "").upper()
    pollutants = observation.get("pollutants", {})

    pm25_val = pollutants.get("PM2.5") or pollutants.get("PM25") or 0
    pm10_val = pollutants.get("PM10") or 0

    is_pm_primary = "PM" in primary_pollutant
    is_pm10_primary = "PM10" in primary_pollutant
    is_pm25_primary = is_pm_primary and not is_pm10_primary
    is_pm_elevated = aqi_val > 50 and is_pm_primary

    wind_speed = weather.get("wind_speed_mph") if weather else None
    wind_dir = weather.get("wind_direction_deg") if weather else None
    temp_f = weather.get("temperature_f") if weather else None
    boundary_layer_height_m = weather.get("boundary_layer_height_m") if weather else None

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
        "incident_name": incident_name,
        "unverified_news_incident": incident_name if not has_corroboration else None,
        "details": firms_details
    })

    # Signal 3: Wind Field & Boundary Layer Height
    if weather and wind_speed is not None and wind_dir is not None:
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
    pm25_conc = openaq_sig.get("pm25")
    pm10_conc = openaq_sig.get("pm10")
    signals.append({
        "id": "surface_pm_level",
        "label": "Surface Particulate Matter (PM2.5 / PM10)",
        "status": "present" if (pm25_conc is not None or pm10_conc is not None or is_pm_primary) else "absent",
        "primary": is_pm_primary,
        "pm10_primary": is_pm10_primary,
        "pm25_primary": is_pm25_primary,
        "elevated": is_pm_elevated,
        "pm25_value": pm25_val,
        "pm10_value": pm10_val,
        "pm25_conc": pm25_conc,
        "pm10_conc": pm10_conc,
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

    signals.append(openaq_sig)

    return signals
