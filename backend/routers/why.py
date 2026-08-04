import time
import json
import asyncio
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from backend.services.openmeteo import fetch_openmeteo_weather, fetch_openmeteo_aqi
from backend.services.aod import fetch_aod_signal
from backend.services.firms import fetch_firms_hotspots
from backend.services.incident_search import search_fire_incident_name
from backend.engine.signals import (
    assemble_evidence_signals,
    TOOL_STEPS,
    create_trace_step,
    collect_openaq_signal,
)
from backend.engine.score import score_hypotheses
from backend.llm import generate_narrative_briefing, generate_narrative_briefing_stream, generate_fallback_narrative
from backend.llm_judge import judge_narrative
from backend.db import get_cached_narrative, set_cached_narrative

router = APIRouter(prefix="/api", tags=["Why Attribution"])

class WhyRequest(BaseModel):
    location: Dict[str, Any]
    observation: Dict[str, Any]

async def _async_judge_streamed_narrative(evidence_payload: Dict[str, Any], full_narrative: str, cache_key: str):
    """Post-hoc non-blocking evaluation of streamed narrative."""
    try:
        verdict = await judge_narrative(evidence_payload, full_narrative)
        if verdict.get("verdict") == "fail":
            print(f"[LLM Judge Stream Warning]: Streamed narrative failed judge check ({verdict.get('reasoning')}). Hallucinations: {verdict.get('hallucinations')}, Jargon: {verdict.get('leaked_jargon')}")
    except Exception as e:
        print(f"[LLM Judge Stream Error]: {e}")

@router.post("/why")
async def get_why_explanation(req: WhyRequest):
    """
    Main attribution explanation endpoint:
    1. Assembles live environmental evidence signals (AOD, FIRMS, Wind, PM level, Ozone).
    2. Deterministically scores & ranks the 5 attribution hypotheses (Evidence Matrix).
    3. Synthesizes a natural language briefing via DeepSeek.
    4. Evaluates narrative grounding via LLM Judge (Groq API).
    5. Returns signals, hypotheses, briefing, and map layers.
    """
    location = req.location
    observation = req.observation

    loc_key = location.get("zip_code") or f"{location.get('lat', 0):.2f}_{location.get('lon', 0):.2f}"
    hour_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H")
    cache_key = f"why_{loc_key}_{hour_stamp}"

    # Step 1-4: Assemble signals (records tool execution timing)
    signals, execution_trace = await assemble_evidence_signals(location, observation)
    
    # Step 5: Score hypotheses tool call
    t0 = time.perf_counter()
    hypotheses, open_questions = score_hypotheses(observation, signals)
    t1 = time.perf_counter()
    execution_trace.append(create_trace_step("score_hypotheses", (t1 - t0) * 1000, "done"))

    # DeepSeek Briefing Synthesis (Separate from tool trace)
    cached_narrative = get_cached_narrative(cache_key)
    if cached_narrative:
        narrative = cached_narrative
    else:
        narrative = await generate_narrative_briefing(
            location, observation, signals, hypotheses, open_questions
        )
        evidence_payload = {
            "location": location,
            "observation": observation,
            "signals": signals,
            "hypotheses": hypotheses,
            "open_questions": open_questions,
            "narrative": narrative
        }

        # Runtime Gatekeeper Evaluation
        verdict = await judge_narrative(evidence_payload, narrative)
        if verdict.get("verdict") == "fail":
            print(f"[LLM Judge Gatekeeper Fail]: Retrying narrative generation... Reason: {verdict.get('reasoning')}")
            narrative = await generate_narrative_briefing(
                location, observation, signals, hypotheses, open_questions
            )
            evidence_payload["narrative"] = narrative
            verdict = await judge_narrative(evidence_payload, narrative)
            if verdict.get("verdict") == "fail":
                print(f"[LLM Judge Gatekeeper Fail 2nd Time]: Falling back to deterministic narrative. Reason: {verdict.get('reasoning')}")
                narrative = generate_fallback_narrative(
                    location, observation, signals, hypotheses, open_questions
                )
                evidence_payload["narrative"] = narrative

        set_cached_narrative(cache_key, narrative, evidence_payload, verdict)

    # Prepare map overlays if FIRMS hotspots present
    firms_sig = next((s for s in signals if s["id"] == "firms_upwind"), None)
    map_layers = {}
    if firms_sig and firms_sig.get("hotspots"):
        map_layers["firms_hotspots"] = firms_sig["hotspots"]

    return {
        "location": location,
        "observation": observation,
        "signals": signals,
        "hypotheses": hypotheses,
        "open_questions": open_questions,
        "narrative": narrative,
        "execution_trace": execution_trace,
        "map_layers": map_layers
    }

@router.get("/why/stream")
async def stream_why_explanation(
    lat: float = Query(...),
    lon: float = Query(...),
    zip_code: Optional[str] = Query(None),
    city: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    aqi: Optional[int] = Query(None),
    primary_pollutant: Optional[str] = Query("PM2.5"),
    category: Optional[str] = Query("Moderate")
):
    """
    Real-time Server-Sent Events (SSE) endpoint:
    1. Streams tool_start / tool_done events for weather, AOD, FIRMS, web search, and scoring.
    2. Emits signals_ready payload once tool trace is complete.
    3. Streams DeepSeek LLM narrative briefing (stream=True) token-by-token directly into text box.
    4. Evaluates streamed narrative asynchronously (post-hoc non-blocking LLM judge).
    """
    location = {
        "lat": lat,
        "lon": lon,
        "zip_code": zip_code,
        "city": city,
        "state": state,
        "name": name or f"{city or 'Location'}, {state or ''}".strip(", ") or f"{lat:.2f}, {lon:.2f}"
    }

    # Fetch observation if not provided
    if aqi is None:
        obs_res = await fetch_openmeteo_aqi(lat, lon)
        observation = obs_res if obs_res else {
            "source": "Open-Meteo",
            "aqi": 50,
            "primary_pollutant": "PM2.5",
            "category": "Good",
            "pollutants": {"PM2.5": 12.0}
        }
    else:
        observation = {
            "source": "AirNow",
            "aqi": aqi,
            "primary_pollutant": primary_pollutant or "PM2.5",
            "category": category or "Moderate",
            "pollutants": {"PM2.5": float(aqi)}
        }

    async def sse_event_generator():
        execution_trace = []
        loc_key = zip_code or f"{lat:.2f}_{lon:.2f}"
        hour_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H")
        cache_key = f"why_{loc_key}_{hour_stamp}"

        # 1. Weather Vector Tool
        yield f"event: tool_start\ndata: {json.dumps({'step': 'weather_vector', 'label': TOOL_STEPS['weather_vector']})}\n\n"
        t0 = time.perf_counter()
        weather = await fetch_openmeteo_weather(lat, lon)
        t1 = time.perf_counter()
        wind_speed = weather.get("wind_speed_mph") if weather else None
        wind_dir = weather.get("wind_direction_deg") if weather else None
        temp_f = weather.get("temperature_f") if weather else None

        weather_trace = create_trace_step("weather_vector", (t1 - t0) * 1000, "done" if weather else "warning")
        execution_trace.append(weather_trace)
        yield f"event: tool_done\ndata: {json.dumps(weather_trace)}\n\n"

        # 2. AOD Density Tool
        yield f"event: tool_start\ndata: {json.dumps({'step': 'aod_density', 'label': TOOL_STEPS['aod_density']})}\n\n"
        t0 = time.perf_counter()
        aod_res = await fetch_aod_signal(lat, lon)
        t1 = time.perf_counter()
        aod_trace = create_trace_step("aod_density", (t1 - t0) * 1000, aod_res.get("status", "done"))
        execution_trace.append(aod_trace)
        yield f"event: tool_done\ndata: {json.dumps(aod_trace)}\n\n"

        # 3. FIRMS Scan Tool
        yield f"event: tool_start\ndata: {json.dumps({'step': 'firms_scan', 'label': TOOL_STEPS['firms_scan']})}\n\n"
        t0 = time.perf_counter()
        firms_res = await fetch_firms_hotspots(lat, lon, wind_dir, wind_speed)
        t1 = time.perf_counter()
        firms_trace = create_trace_step("firms_scan", (t1 - t0) * 1000, firms_res.get("status", "done"))
        execution_trace.append(firms_trace)
        yield f"event: tool_done\ndata: {json.dumps(firms_trace)}\n\n"

        # 4. Web Search Tool
        yield f"event: tool_start\ndata: {json.dumps({'step': 'web_search', 'label': TOOL_STEPS['web_search']})}\n\n"
        t0 = time.perf_counter()
        aqi_val = observation.get("aqi", 0)
        primary_pollutant = observation.get("primary_pollutant", "").upper()
        is_pm_primary = "PM" in primary_pollutant
        is_pm_elevated = aqi_val > 50 and is_pm_primary

        incident_name = None
        if is_pm_elevated or firms_res["status"] == "present" or aod_res["status"] == "present":
            incident_name = await search_fire_incident_name(state, city, lat, lon)
        t1 = time.perf_counter()

        web_trace = create_trace_step("web_search", (t1 - t0) * 1000, "done" if incident_name else "absent")
        execution_trace.append(web_trace)
        yield f"event: tool_done\ndata: {json.dumps(web_trace)}\n\n"

        # 5. OpenAQ Reference Monitor Concentrations Tool
        yield f"event: tool_start\ndata: {json.dumps({'step': 'openaq_monitors', 'label': TOOL_STEPS['openaq_monitors']})}\n\n"
        t0 = time.perf_counter()
        openaq_sig = await collect_openaq_signal(lat, lon)
        t1 = time.perf_counter()
        openaq_trace = create_trace_step(
            "openaq_monitors",
            (t1 - t0) * 1000,
            "done" if openaq_sig.get("status") == "present" else "warning"
        )
        execution_trace.append(openaq_trace)
        yield f"event: tool_done\ndata: {json.dumps(openaq_trace)}\n\n"

        boundary_layer_height_m = weather.get("boundary_layer_height_m") if weather else None
        is_pm10_primary = "PM10" in primary_pollutant
        is_pm25_primary = is_pm_primary and not is_pm10_primary

        # Assemble final signals
        signals = [
            {
                "id": "aerosol_plume",
                "label": "Atmospheric Column Particle Density (AOD)",
                "status": aod_res["status"],
                "density": aod_res.get("density"),
                "aod_value": aod_res.get("aod_value"),
                "details": aod_res.get("details", "")
            },
            {
                "id": "firms_upwind",
                "label": "Upwind Thermal Hotspots (NASA FIRMS)",
                "status": firms_res["status"],
                "count": firms_res.get("count", 0),
                "total_count": firms_res.get("total_count", 0),
                "nearest": firms_res.get("nearest"),
                "hotspots": firms_res.get("hotspots", []),
                "alignment": firms_res.get("alignment"),
                "incident_name": incident_name,
                "details": firms_res.get("details", "")
            },
            {
                "id": "wind",
                "label": "Surface Wind Vector",
                "status": "present" if weather else "unavailable",
                "speed_mph": wind_speed,
                "direction_deg": wind_dir,
                "upwind_deg": round((wind_dir + 180) % 360, 1) if wind_dir is not None else None,
                "boundary_layer_height_m": boundary_layer_height_m,
                "details": f"{wind_speed} mph from {wind_dir}°" if weather else "Unavailable"
            },
            {
                "id": "surface_pm_level",
                "label": "Surface Particulate Matter (PM2.5 / PM10)",
                "status": "present" if is_pm_primary else "absent",
                "primary": is_pm_primary,
                "pm10_primary": is_pm10_primary,
                "pm25_primary": is_pm25_primary,
                "elevated": is_pm_elevated,
                "pm25_conc": openaq_sig.get("pm25"),
                "pm10_conc": openaq_sig.get("pm10"),
                "details": f"Primary pollutant: {primary_pollutant} (AQI {aqi_val})"
            },
            {
                "id": "ozone_heat",
                "label": "Ozone & Atmospheric Heat",
                "status": "present" if ("O3" in primary_pollutant or (temp_f and temp_f >= 85)) else "absent",
                "primary": "O3" in primary_pollutant,
                "hot_day": bool(temp_f and temp_f >= 85),
                "temperature_f": temp_f,
                "details": f"Temperature: {temp_f}°F" if temp_f else "N/A"
            },
            openaq_sig
        ]

        # 5. Score Hypotheses Tool
        yield f"event: tool_start\ndata: {json.dumps({'step': 'score_hypotheses', 'label': TOOL_STEPS['score_hypotheses']})}\n\n"
        t0 = time.perf_counter()
        hypotheses, open_questions = score_hypotheses(observation, signals)
        t1 = time.perf_counter()
        score_trace = create_trace_step("score_hypotheses", (t1 - t0) * 1000, "done")
        execution_trace.append(score_trace)
        yield f"event: tool_done\ndata: {json.dumps(score_trace)}\n\n"

        firms_sig = next((s for s in signals if s["id"] == "firms_upwind"), None)
        map_layers = {}
        if firms_sig and firms_sig.get("hotspots"):
            map_layers["firms_hotspots"] = firms_sig["hotspots"]

        signals_payload = {
            "signals": signals,
            "hypotheses": hypotheses,
            "open_questions": open_questions,
            "map_layers": map_layers,
            "execution_trace": execution_trace
        }
        yield f"event: signals_ready\ndata: {json.dumps(signals_payload)}\n\n"

        # DeepSeek Briefing Token Streaming (Directly into narrative box)
        cached_narrative = get_cached_narrative(cache_key)
        full_narrative = ""

        if cached_narrative:
            full_narrative = cached_narrative
            words = cached_narrative.split(" ")
            for w in words:
                yield f"event: llm_token\ndata: {json.dumps({'token': w + ' '})}\n\n"
                await asyncio.sleep(0.015)
        else:
            async for token in generate_narrative_briefing_stream(location, observation, signals, hypotheses, open_questions):
                full_narrative += token
                yield f"event: llm_token\ndata: {json.dumps({'token': token})}\n\n"
            
            evidence_payload = {
                "location": location,
                "observation": observation,
                "signals": signals,
                "hypotheses": hypotheses,
                "open_questions": open_questions,
                "narrative": full_narrative
            }
            set_cached_narrative(cache_key, full_narrative, evidence_payload)
            # Post-hoc non-blocking judge check
            asyncio.create_task(_async_judge_streamed_narrative(evidence_payload, full_narrative, cache_key))

        yield f"event: complete\ndata: {json.dumps({'narrative': full_narrative, 'execution_trace': execution_trace})}\n\n"

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")
