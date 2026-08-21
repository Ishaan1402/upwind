import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
from backend.services.openmeteo import fetch_openmeteo_weather
from backend.services.aod import fetch_aod_signal
from backend.services.firms import fetch_firms_hotspots
from backend.services.hms import fetch_hms_smoke
from backend.services.wfigs import fetch_wfigs_incident
from backend.services.nws import fetch_dust_alert
from backend.services.metar import fetch_metar_dust
from backend.services.place_context import fetch_place_context
from backend.services.incident_search import search_fire_incident_name
from backend.services.openaq import (
    discover_reference_monitors,
    fetch_latest,
    fetch_location_sensors,
    fetch_daily_baseline,
    fetch_same_hour_baseline,
    sensor_id_for_parameter,
    monitor_source_label,
)
from backend.engine.params import get_params
from backend.services.airnow import fetch_airnow_concentrations

TOOL_STEPS = {
    "weather_vector": "Calculating Open-Meteo wind trajectory & temperature",
    "aod_density": "Reading modeled Aerosol Optical Depth (AOD) column density",
    "hms_scan": "Checking NOAA smoke-plume analysis overhead",
    "wfigs_scan": "Checking federal wildfire incident registry",
    "nws_dust_scan": "Checking NWS dust warnings/advisories",
    "metar_dust_scan": "Checking nearby METAR stations for dust",
    "firms_scan": "Scanning NASA FIRMS thermal hotspot clusters upwind",
    "web_search": "Searching public news & active incident feeds (Web Search)",
    "openaq_monitors": "Reading local monitor concentrations (OpenAQ)",
    "place_context": "Looking up local population context (Census)",
    "score_hypotheses": "Scoring attribution hypotheses (Evidence Matrix)"
}

def create_trace_step(step: str, duration_ms: float, status: str = "done", as_of: Optional[str] = None) -> Dict[str, Any]:
    trace: Dict[str, Any] = {
        "step": step,
        "label": TOOL_STEPS.get(step, step),
        "duration_ms": max(0.1, round(duration_ms, 1)),
        "status": status
    }
    if as_of is not None:
        trace["as_of"] = as_of
    return trace


def _openaq_unavailable_signal(detail: str) -> Dict[str, Any]:
    return {
        "id": "openaq_concentrations",
        "label": "Local Monitor Concentrations (OpenAQ)",
        "status": "unavailable",
        "details": detail,
    }


async def collect_openaq_signal(
    lat: float,
    lon: float,
    include_baselines: bool = True,
    country_code: Optional[str] = "US",
) -> Dict[str, Any]:
    """
    Gather OpenAQ reference-monitor concentrations for attribution.

    Only fresh (<=3h) reference-monitor readings for the target country are
    included; anything else results in an "unavailable" signal so scoring
    behaves as before. A country_code of None searches without a country filter.
    Baseline percentiles are skipped when include_baselines is False (Good-AQI
    briefings never use them, so the extra calls are wasted latency).
    Never raises.
    """
    p = get_params()
    try:
        candidates = await discover_reference_monitors(lat, lon, limit=3, country_code=country_code)
        if not candidates:
            return _openaq_unavailable_signal(
                "No air quality monitor within 25 km (or OpenAQ key not configured)"
            )

        # Fetch fresh readings for all nearby candidates concurrently
        latest_by_id: Dict[Any, Dict[str, Any]] = {}
        results = await asyncio.gather(
            *(fetch_latest(c["location_id"]) for c in candidates),
            return_exceptions=True,
        )
        for candidate, candidate_readings in zip(candidates, results):
            if isinstance(candidate_readings, dict) and candidate_readings:
                latest_by_id[candidate["location_id"]] = candidate_readings

        # Widen radius when nearest monitor lacks fresh readings
        if not latest_by_id:
            wider = await discover_reference_monitors(
                lat, lon, limit=10, radius_m=p.openaq_radius_m, country_code=country_code
            )
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

        # Select nearest candidate with live PM2.5 and PM10 readings
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
            "ratio_source": None,
            "ratio_monitor_distance_km": None,
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
            signal["ratio_source"] = "openaq"

        # OpenAQ's nearest monitor lacks a co-located PM10: recover the
        # fine/coarse discriminator from the AirNow raw hourly feed instead.
        # AirNow serves real monitor data, so no "modeled" flag is needed.
        if signal["pm25_pm10_ratio"] is None:
            airnow = await fetch_airnow_concentrations(lat, lon)
            if airnow and airnow.get("pm25_pm10_ratio") is not None:
                signal["pm25_pm10_ratio"] = airnow["pm25_pm10_ratio"]
                signal["ratio_source"] = "airnow"
                site = airnow.get("site") or {}
                signal["ratio_monitor_distance_km"] = site.get("distance_km")
                signal["details"] += (
                    f" PM2.5/PM10 ratio from AirNow monitor "
                    f"{site.get('name') or site.get('aqsid')} {site.get('distance_km')} km away"
                )

        for key, param in (("o3_ppb", "o3"), ("no2_ppb", "no2"), ("co_ppm", "co"), ("so2_ppb", "so2")):
            reading = readings.get(param)
            if reading:
                signal[key] = reading["value"]

        sensors_map = await fetch_location_sensors(location_id)
        pm25_sensor_id = sensor_id_for_parameter(sensors_map, "pm25")
        if include_baselines and pm25_sensor_id is not None and pm25:
            daily, same_hour = await asyncio.gather(
                fetch_daily_baseline(pm25_sensor_id),
                fetch_same_hour_baseline(pm25_sensor_id, monitor.get("timezone"), pm25["value"]),
                return_exceptions=True,
            )
            if isinstance(daily, dict) and daily:
                signal["daily_percentile"] = daily["percentile"]
            if isinstance(same_hour, dict) and same_hour:
                signal["same_hour_percentile"] = same_hour["percentile"]
                signal["same_hour_median"] = same_hour["median"]

        return signal
    except Exception as e:
        print(f"[OpenAQ Service Error]: {e}")
        return _openaq_unavailable_signal("OpenAQ concentration feed unavailable")


def _unavailable_feed_result(fallback: Dict[str, Any], detail: str) -> Dict[str, Any]:
    """Normalize a raised or empty tool result into an unavailable signal payload."""
    if isinstance(fallback, BaseException):
        return {"status": "unavailable", "details": detail}
    return fallback if fallback else {"status": "unavailable", "details": detail}


async def iter_evidence_signals(
    location: Dict[str, Any],
    observation: Dict[str, Any],
) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
    """
    DAG implementation to concurrently '_run' tasks. Applied conditional branching because FIRMS and WFIGS tasks require wind
    to be calculated first. Also web search is gated by AOD and FIRMS. All concurrent tasks are executed in the TaskGroup and 
    output in the subsequent 'assemble_evidence_signals' function. 

    TODO: Clean up all calls with single shared API client 
    """
    lat = location["lat"]
    lon = location["lon"]
    state = location.get("state")
    city = location.get("city")

    events: asyncio.Queue = asyncio.Queue()

    # Map backend feed status to trace step status
    _STEP_STATUS = {"present": "done", "absent": "absent", "unavailable": "warning"}

    def _start(step: str) -> Tuple[str, Dict[str, Any]]:
        return ("tool_start", {"step": step, "label": TOOL_STEPS[step]})

    async def _run_feed(step, coro, unavailable_detail):
        """Run a feed coroutine, emitting tool_start/tool_done with its own duration."""
        events.put_nowait(_start(step))
        t0 = time.perf_counter()
        try:
            result = await coro
        except Exception:
            result = None
        dur = (time.perf_counter() - t0) * 1000
        result = _unavailable_feed_result(result, unavailable_detail)
        status = _STEP_STATUS.get(result.get("status"), "done")
        as_of = result["as_of"] if isinstance(result, dict) and result.get("as_of") else None
        events.put_nowait(("tool_done", create_trace_step(step, dur, status, as_of=as_of)))
        return result

    async def _run_weather():
        events.put_nowait(_start("weather_vector"))
        t0 = time.perf_counter()
        try:
            weather = await fetch_openmeteo_weather(lat, lon)
        except Exception:
            weather = None
        dur = (time.perf_counter() - t0) * 1000
        as_of = weather.get("as_of") if isinstance(weather, dict) else None
        events.put_nowait(("tool_done", create_trace_step("weather_vector", dur, "done" if weather else "warning", as_of=as_of)))
        return weather

    async def _run_web_search():
        events.put_nowait(_start("web_search"))
        t0 = time.perf_counter()
        try:
            name = await search_fire_incident_name(state, city, lat, lon)
        except Exception:
            name = None
        dur = (time.perf_counter() - t0) * 1000
        events.put_nowait(("tool_done", create_trace_step("web_search", dur, "done" if name else "absent")))
        return name

    async def _orchestrate():
        p = get_params()
        aqi_val = observation.get("aqi")
        try:
            async with asyncio.TaskGroup() as tg:
                is_pm_elevated = (
                    aqi_val is not None and aqi_val > p.aqi_elevated and "PM" in observation.get("primary_pollutant", "").upper()
                )

                # T=0: independent position-only tasks
                weather_t = tg.create_task(_run_weather())
                aod_t = tg.create_task(_run_feed(
                    "aod_density", fetch_aod_signal(lat, lon, observation.get("aerosol_optical_depth")), "AOD feed unavailable"
                ))
                hms_t = tg.create_task(_run_feed("hms_scan", fetch_hms_smoke(lat, lon), "NOAA HMS smoke feed unavailable"))
                nws_t = tg.create_task(_run_feed(
                    "nws_dust_scan", fetch_dust_alert(lat, lon), "NWS dust alerts unavailable"
                ))
                metar_t = tg.create_task(_run_feed(
                    "metar_dust_scan", fetch_metar_dust(lat, lon), "METAR dust observations unavailable"
                ))
                openaq_t = tg.create_task(_run_feed(
                    "openaq_monitors", collect_openaq_signal(lat, lon, include_baselines=(aqi_val is not None and aqi_val > p.aqi_elevated), country_code=location.get("country_code") or "US"),
                    "OpenAQ concentration feed unavailable",
                ))
                place_t = tg.create_task(_run_feed(
                    "place_context", fetch_place_context(location.get("zip_code")), "Census place context unavailable",
                ))
                web_t = tg.create_task(_run_web_search()) if is_pm_elevated else None

                # Start wind-dependent feeds once weather resolves
                weather = await weather_t
                wind_speed = weather.get("wind_speed_mph") if weather else None
                wind_dir = weather.get("wind_direction_deg") if weather else None
                firms_t = tg.create_task(_run_feed("firms_scan", fetch_firms_hotspots(lat, lon, wind_dir, wind_speed), "NASA FIRMS unavailable"))
                wfigs_t = tg.create_task(_run_feed("wfigs_scan", fetch_wfigs_incident(lat, lon, wind_dir), "NIFC WFIGS incident feed unavailable"))

                aod = await aod_t
                firms = await firms_t

                # Run web search for non-elevated PM only if fire evidence exists
                if web_t is None and (firms.get("status") == "present" or aod.get("status") == "present"):
                    web_t = tg.create_task(_run_web_search())
                if web_t is None:
                    # Resolve skipped search step with absent status
                    events.put_nowait(("tool_start", {"step": "web_search", "label": TOOL_STEPS["web_search"]}))
                    events.put_nowait(("tool_done", create_trace_step("web_search", 0.0, "absent")))

                hms = await hms_t
                wfigs = await wfigs_t
                openaq = await openaq_t
                place = await place_t
                incident_name = await web_t if web_t else None
                nws = await nws_t
                metar = await metar_t

            signals = build_evidence_signals(
                observation, weather, aod, firms, incident_name, openaq, hms, wfigs, place,
                nws_res=nws,
                metar_res=metar,
            )
        except Exception as exc:
            # Fall back to empty signals on orchestration exception
            print(f"[Evidence Pipeline Error]: {exc}")
            signals = []
        events.put_nowait(("signals", signals))

    orch_task = asyncio.create_task(_orchestrate())

    try:
        while True:
            kind, payload = await events.get()
            yield (kind, payload)
            if kind == "signals":
                return
    finally:
        # Cancel orchestrator and active child tasks on client disconnect
        orch_task.cancel()


async def assemble_evidence_signals(
    location: Dict[str, Any],
    observation: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Assemble evidence signals package for a given location and observation.
    Measures exact real execution timing for each backend tool step.
    Returns (signals, execution_trace).
    """
    execution_trace: List[Dict[str, Any]] = []
    signals: Optional[List[Dict[str, Any]]] = None
    async for kind, payload in iter_evidence_signals(location, observation):
        if kind == "tool_done":
            execution_trace.append(payload)
        elif kind == "signals":
            signals = payload
    return signals or [], execution_trace


def build_evidence_signals(
    observation: Dict[str, Any],
    weather: Optional[Dict[str, Any]],
    aod_res: Dict[str, Any],
    firms_res: Dict[str, Any],
    incident_name: Optional[str],
    openaq_sig: Dict[str, Any],
    hms_res: Optional[Dict[str, Any]] = None,
    wfigs_res: Optional[Dict[str, Any]] = None,
    place_res: Optional[Dict[str, Any]] = None,
    nws_res: Optional[Dict[str, Any]] = None,
    metar_res: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Build the shared evidence signals list for both /api/why paths.

    The non-streaming and streaming (SSE) endpoints must present identical
    evidence to scoring and the LLM. Incident names are passed through raw;
    scoring decides whether they are corroborated by hotspots (fire vote) or
    become unverified-news open questions. hms_res/wfigs_res/place_res/
    nws_res/metar_res default to None (treated as an unavailable feed) so
    callers that predate the new feeds keep working.
    """
    hms_res = hms_res or {"status": "unavailable", "details": "NOAA HMS smoke feed not queried"}
    wfigs_res = wfigs_res or {"status": "unavailable", "details": "WFIGS incident feed not queried"}
    place_res = place_res or {"status": "unavailable", "details": "Census place context not queried"}
    nws_res = nws_res or {"status": "unavailable", "details": "NWS dust alerts not queried"}
    metar_res = metar_res or {"status": "unavailable", "details": "METAR dust observations not queried"}
    p = get_params()
    aqi_val = observation.get("aqi", 0)
    primary_pollutant = observation.get("primary_pollutant", "").upper()
    pollutants = observation.get("pollutants", {})

    pm25_val = pollutants.get("PM2.5") or pollutants.get("PM25") or 0
    pm10_val = pollutants.get("PM10") or 0

    is_pm_primary = "PM" in primary_pollutant
    is_pm10_primary = "PM10" in primary_pollutant
    is_pm25_primary = is_pm_primary and not is_pm10_primary
    # None-safe: a missing AQI is unknown, never elevated.
    is_pm_elevated = aqi_val is not None and aqi_val > p.aqi_elevated and is_pm_primary

    wind_speed = weather.get("wind_speed_mph") if weather else None
    wind_dir = weather.get("wind_direction_deg") if weather else None
    temp_f = weather.get("temperature_f") if weather else None
    boundary_layer_height_m = weather.get("boundary_layer_height_m") if weather else None
    wind_gust_mph = weather.get("wind_gust_mph") if weather else None
    precip_30d_in = weather.get("precip_30d_in") if weather else None

    signals = []

    # Signal 1: Aerosol Optical Depth Plume
    signals.append({
        "id": "aerosol_plume",
        "label": "Modeled Atmospheric Column Particle Density (AOD)",
        "status": aod_res["status"],
        "density": aod_res.get("density"),
        "aod_value": aod_res.get("aod_value"),
        "details": aod_res.get("details", "")
    })

    # Signal 2: NOAA HMS Smoke Plume Analysis
    signals.append({
        "id": "hms_smoke",
        "label": "NOAA Smoke-Plume Analysis (HMS)",
        "status": hms_res.get("status", "unavailable"),
        "density": hms_res.get("density"),
        "details": hms_res.get("details", ""),
    })

    # Signal 3: FIRMS Upwind Hotspots
    firms_status = firms_res["status"]
    firms_details = firms_res.get("details", "")
    # AOD is modeled column loading, NOT fire evidence: it does not corroborate
    # a news incident. Only verified FIRMS hotspots do.
    firms_present = firms_status == "present"
    has_corroboration = firms_present

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

    # Signal 4: WFIGS Federal Incident Registry
    signals.append({
        "id": "wfigs_incident",
        "label": "Federal Wildfire Incident Registry (WFIGS)",
        "status": wfigs_res.get("status", "unavailable"),
        "incident": wfigs_res.get("incident"),
        "count": wfigs_res.get("count", 0),
        "alignment": wfigs_res.get("alignment"),
        "details": wfigs_res.get("details", ""),
    })

    # Signal: NWS Dust Warning/Advisory (one-sided dust confirmation)
    signals.append({
        "id": "nws_dust_alert",
        "label": "NWS Dust Warning/Advisory",
        "status": nws_res.get("status", "unavailable"),
        "event": nws_res.get("event"),
        "headline": nws_res.get("headline"),
        "severity": nws_res.get("severity"),
        "details": nws_res.get("details", ""),
    })

    # Signal: METAR Dust Report (one-sided dust confirmation)
    signals.append({
        "id": "metar_dust",
        "label": "METAR Dust Report",
        "status": metar_res.get("status", "unavailable"),
        "station": metar_res.get("station"),
        "phenomenon": metar_res.get("phenomenon"),
        "details": metar_res.get("details", ""),
    })

    # Signal 5: Wind Field & Boundary Layer Height
    if weather and wind_speed is not None and wind_dir is not None:
        # Wind direction is meteorological origin (upwind bearing)
        upwind_dir = wind_dir % 360

        # Dust-confirmed flag (gust >= dust_gust_mph_min over antecedent-dry
        # ground, on a PM10-primary elevated day) surfaces as a details note.
        gust_dry_dust = (
            is_pm10_primary and is_pm_elevated
            and wind_gust_mph is not None and precip_30d_in is not None
            and wind_gust_mph >= p.dust_gust_mph_min
            and precip_30d_in <= p.dust_precip_30d_max_in
        )
        details = f"{wind_speed} mph from {wind_dir}°"
        if gust_dry_dust:
            details += (
                f" — gusty wind ({wind_gust_mph:.0f} mph) over dry ground "
                f"(30-day precip {precip_30d_in:.1f} in) confirms dust lofting"
            )

        signals.append({
            "id": "wind",
            "label": "Surface Wind Vector",
            "status": "present",
            "speed_mph": wind_speed,
            "direction_deg": wind_dir,
            "upwind_deg": round(upwind_dir, 1),
            "boundary_layer_height_m": boundary_layer_height_m,
            "wind_gust_mph": wind_gust_mph,
            "precip_30d_in": precip_30d_in,
            "details": details
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
            "wind_gust_mph": None,
            "precip_30d_in": None,
            "details": "Wind vector data unavailable"
        })

    # Signal 4: Surface PM Level
    pm25_conc = openaq_sig.get("pm25")
    pm10_conc = openaq_sig.get("pm10")
    signals.append({
        "id": "surface_pm_level",
        "label": "Reported AirNow Index",
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
    is_hot = (temp_f is not None) and (temp_f >= p.ozone_hot_temp_f)
    
    signals.append({
        "id": "ozone_heat",
        "label": "Ozone & Atmospheric Heat",
        "status": "present" if (is_o3_primary or is_hot) else "absent",
        "primary": is_o3_primary,
        "hot_day": is_hot,
        "temperature_f": temp_f,
        "details": f"O3 Primary: {is_o3_primary}, Temperature: {f'{temp_f}°F' if temp_f else 'N/A'}"
    })

    # Signal 6: Local Population Context (Census)
    signals.append({
        "id": "place_context",
        "label": "Local Population Context (Census)",
        "status": place_res.get("status", "unavailable"),
        "population": place_res.get("population"),
        "rural": place_res.get("rural"),
        "details": place_res.get("details", ""),
    })

    signals.append(openaq_sig)

    return signals
