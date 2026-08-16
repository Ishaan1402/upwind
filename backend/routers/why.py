import time
import json
import asyncio
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from backend.services.openmeteo import fetch_openmeteo_aqi
from backend.engine.signals import (
    assemble_evidence_signals,
    iter_evidence_signals,
    TOOL_STEPS,
    create_trace_step,
)
from backend.engine.score import score_hypotheses
from backend.llm import generate_narrative_briefing, generate_narrative_briefing_stream, generate_fallback_narrative
from backend.llm_judge import judge_narrative
from backend.db import get_cached_narrative, set_cached_narrative, update_cached_verdict

router = APIRouter(prefix="/api", tags=["Why Attribution"])

class WhyRequest(BaseModel):
    location: Dict[str, Any]
    observation: Dict[str, Any]

# strong reference to prevent garbage collection; keeps LLM judge task running in background
_background_tasks: set = set()


def _schedule_background(coro):
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task

async def _async_judge_streamed_narrative(evidence_payload: Dict[str, Any], full_narrative: str, cache_key: str):
    """Post-hoc non-blocking evaluation of streamed narrative."""
    try:
        verdict = await judge_narrative(evidence_payload, full_narrative)
        update_cached_verdict(cache_key, verdict)
        if verdict.get("verdict") == "fail":
            # Replace rejected narrative with safe fallback in cache
            fallback = generate_fallback_narrative(
                evidence_payload["location"],
                evidence_payload["observation"],
                evidence_payload["signals"],
                evidence_payload["hypotheses"],
                evidence_payload["open_questions"],
            )
            set_cached_narrative(cache_key, fallback, evidence_payload, verdict)
            print(f"[LLM Judge Stream Warning]: Streamed narrative failed judge check ({verdict.get('reasoning')}). Hallucinations: {verdict.get('hallucinations')}, Jargon: {verdict.get('leaked_jargon')}")
    except Exception as e:
        print(f"[LLM Judge Stream Error]: {e}")

@router.post("/why")
async def get_why_explanation(req: WhyRequest):
    """
    Main attribution explanation endpoint:
    1. Assembles live environmental evidence signals (AOD, FIRMS, Wind, PM level, Ozone).
    2. Deterministically scores & ranks the 5 attribution hypotheses (check score.py).
    3. Synthesizes a natural language briefing via LLM (most likely DeepSeek).
    4. Evaluates narrative grounding via LLM Judge (Groq API).
    5. Returns signals, hypotheses, briefing, and map layers.
    """
    location = req.location
    observation = req.observation

    loc_key = location.get("zip_code") or f"{location.get('lat', 0):.2f}_{location.get('lon', 0):.2f}"
    hour_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H")
    # Include pollutant and AQI value in cache key in case air quality shifts
    cache_key = (
        f"why_v2_{loc_key}_{observation.get('aqi', 0)}_"
        f"{observation.get('primary_pollutant', '')}_{hour_stamp}"
    )

    # Step 1-4: Assemble signals (records tool execution timing)
    t_start = time.perf_counter()
    signals, execution_trace = await assemble_evidence_signals(location, observation)
    
    # Step 5: Score hypotheses tool call
    t0 = time.perf_counter()
    hypotheses, open_questions = score_hypotheses(observation, signals)
    t1 = time.perf_counter()
    execution_trace.append(create_trace_step("score_hypotheses", (t1 - t0) * 1000, "done"))
    total_ms = (time.perf_counter() - t_start) * 1000

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
        "total_ms": total_ms,
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
            # Client reports AQI only without measured concentration
            "pollutants": {}
        }

    async def sse_event_generator():
        execution_trace = []
        t_start = time.perf_counter()
        loc_key = zip_code or f"{lat:.2f}_{lon:.2f}"
        hour_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H")
        cache_key = (
            f"why_v2_{loc_key}_{observation.get('aqi', 0)}_"
            f"{observation.get('primary_pollutant', '')}_{hour_stamp}"
        )

        # Stream evidence tool execution events
        async for kind, payload in iter_evidence_signals(location, observation):
            if kind == "tool_start":
                yield f"event: tool_start\ndata: {json.dumps(payload)}\n\n"
            elif kind == "tool_done":
                execution_trace.append(payload)
                yield f"event: tool_done\ndata: {json.dumps(payload)}\n\n"
            else:  # "signals"
                signals = payload

        # 5. Score Hypotheses Tool
        yield f"event: tool_start\ndata: {json.dumps({'step': 'score_hypotheses', 'label': TOOL_STEPS['score_hypotheses']})}\n\n"
        t0 = time.perf_counter()
        hypotheses, open_questions = score_hypotheses(observation, signals)
        t1 = time.perf_counter()
        score_trace = create_trace_step("score_hypotheses", (t1 - t0) * 1000, "done")
        execution_trace.append(score_trace)
        yield f"event: tool_done\ndata: {json.dumps(score_trace)}\n\n"
        total_ms = (time.perf_counter() - t_start) * 1000

        firms_sig = next((s for s in signals if s["id"] == "firms_upwind"), None)
        map_layers = {}
        if firms_sig and firms_sig.get("hotspots"):
            map_layers["firms_hotspots"] = firms_sig["hotspots"]

        signals_payload = {
            "signals": signals,
            "hypotheses": hypotheses,
            "open_questions": open_questions,
            "map_layers": map_layers,
            "execution_trace": execution_trace,
            "total_ms": total_ms
        }
        yield f"event: signals_ready\ndata: {json.dumps(signals_payload)}\n\n"

        # DeepSeek Briefing Token Streaming (Directly into narrative box)
        cached_narrative = get_cached_narrative(cache_key)
        full_narrative = ""

        if cached_narrative:
            full_narrative = cached_narrative
            # Return cached narrative immediately without word streaming
            yield f"event: llm_token\ndata: {json.dumps({'token': cached_narrative})}\n\n"
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
            # Run non-blocking narrative judge check in background
            _schedule_background(_async_judge_streamed_narrative(evidence_payload, full_narrative, cache_key))

        yield f"event: complete\ndata: {json.dumps({'narrative': full_narrative, 'execution_trace': execution_trace})}\n\n"

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")
