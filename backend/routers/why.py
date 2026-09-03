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
from backend.llm import generate_narrative_briefing, generate_narrative_briefing_stream, generate_fallback_narrative, StreamMeta
from backend.llm_judge import judge_narrative, JudgeResult
from backend.db import get_cached_narrative, set_cached_narrative, update_cached_verdict
from backend.config import (
    DEEPSEEK_API_KEY,
    ENFORCE_OBSERVATION_TOKENS,
    OBSERVATION_TOKEN_SECRET,
)
from backend.metrics import (
    estimate_llm_cost,
    record_why,
    record_signal_events,
    update_why_verdict,
)
from backend.services.coverage import coverage_for_location
from backend.observation_token import verify_observation_token

router = APIRouter(prefix="/api", tags=["Why Attribution"])

class WhyRequest(BaseModel):
    location: Dict[str, Any]
    observation: Dict[str, Any]
    observation_token: Optional[str] = None

# strong reference to prevent garbage collection; keeps LLM judge task running in background
_background_tasks: set = set()


def _schedule_background(coro):
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task

async def _async_judge_streamed_narrative(evidence_payload: Dict[str, Any], full_narrative: str, cache_key: str, why_event_id: Optional[int] = None):
    """Post-hoc non-blocking evaluation of streamed narrative."""
    try:
        verdict = await judge_narrative(evidence_payload, full_narrative)
        update_cached_verdict(cache_key, verdict.to_dict())
        # Write the verdict + judge tokens back to the exact telemetry row so
        # streamed traffic populates the live judge stats.
        update_why_verdict(
            why_event_id,
            verdict.verdict,
            verdict.judge_input_tokens or None,
            verdict.judge_output_tokens or None,
        )
        if verdict.verdict == "fail":
            # Replace rejected narrative with safe fallback in cache
            fallback = generate_fallback_narrative(
                evidence_payload["location"],
                evidence_payload["observation"],
                evidence_payload["signals"],
                evidence_payload["hypotheses"],
                evidence_payload["open_questions"],
            )
            set_cached_narrative(cache_key, fallback, evidence_payload, verdict.to_dict())
            print(f"[LLM Judge Stream Warning]: Streamed narrative failed judge check ({verdict.reasoning}). Hallucinations: {verdict.hallucinations}, Jargon: {verdict.leaked_jargon}")
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

    if ENFORCE_OBSERVATION_TOKENS and not verify_observation_token(
        req.observation_token,
        location,
        observation,
        OBSERVATION_TOKEN_SECRET,
    ):
        raise HTTPException(
            status_code=400,
            detail="observation_token is missing, expired, or does not match this observation.",
        )

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
    cache_hit = cached_narrative is not None
    final_verdict = None
    fallback_used = False
    gatekeeper_retries = 0
    llm_meta: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "fell_back": False}
    judge_meta: Dict[str, Any] = {"judge_input_tokens": 0, "judge_output_tokens": 0}

    async def _brief() -> str:
        """Generate a briefing and fold real token usage/fallback into meta."""
        result = await generate_narrative_briefing(
            location, observation, signals, hypotheses, open_questions
        )
        if result.fell_back:
            llm_meta["fell_back"] = True
        llm_meta["input_tokens"] += result.input_tokens
        llm_meta["output_tokens"] += result.output_tokens
        return result.narrative

    async def _judge(payload: Dict[str, Any], text: str) -> JudgeResult:
        """Judge a narrative and fold real judge token usage into meta."""
        verdict = await judge_narrative(payload, text)
        judge_meta["judge_input_tokens"] += verdict.judge_input_tokens
        judge_meta["judge_output_tokens"] += verdict.judge_output_tokens
        return verdict

    if cached_narrative:
        narrative = cached_narrative
    else:
        narrative = await _brief()
        evidence_payload = {
            "location": location,
            "observation": observation,
            "signals": signals,
            "hypotheses": hypotheses,
            "open_questions": open_questions,
            "narrative": narrative
        }

        # Runtime Gatekeeper Evaluation
        verdict = await _judge(evidence_payload, narrative)
        if verdict.verdict == "fail":
            print(f"[LLM Judge Gatekeeper Fail]: Retrying narrative generation... Reason: {verdict.reasoning}")
            gatekeeper_retries += 1
            narrative = await _brief()
            evidence_payload["narrative"] = narrative
            verdict = await _judge(evidence_payload, narrative)
            if verdict.verdict == "fail":
                print(f"[LLM Judge Gatekeeper Fail 2nd Time]: Falling back to deterministic narrative. Reason: {verdict.reasoning}")
                narrative = generate_fallback_narrative(
                    location, observation, signals, hypotheses, open_questions
                )
                evidence_payload["narrative"] = narrative
                fallback_used = True

        set_cached_narrative(cache_key, narrative, evidence_payload, verdict.to_dict())
        final_verdict = verdict.verdict if verdict else None

    fallback_used = fallback_used or bool(llm_meta.get("fell_back"))

    # Prepare map overlays if FIRMS hotspots present
    firms_sig = next((s for s in signals if s["id"] == "firms_upwind"), None)
    map_layers = {}
    if firms_sig and firms_sig.get("hotspots"):
        map_layers["firms_hotspots"] = firms_sig["hotspots"]

    llm_generated = (not cache_hit) and bool(DEEPSEEK_API_KEY)
    record_signal_events("/api/why", execution_trace)
    top_h = hypotheses[0] if hypotheses else None
    record_why(
        "/api/why",
        cache_hit=cache_hit,
        llm_generated=llm_generated,
        llm_cost_usd=estimate_llm_cost(narrative) if llm_generated else 0.0,
        judge_verdict=final_verdict,
        country_code=location.get("country_code"),
        loc_key=loc_key,
        total_ms=total_ms,
        top_hypothesis=top_h.get("id") if top_h else None,
        top_confidence=top_h.get("confidence") if top_h else None,
        llm_input_tokens=llm_meta.get("input_tokens") or None,
        llm_output_tokens=llm_meta.get("output_tokens") or None,
        judge_input_tokens=judge_meta.get("judge_input_tokens") or None,
        judge_output_tokens=judge_meta.get("judge_output_tokens") or None,
        fallback_used=fallback_used,
        gatekeeper_retries=gatekeeper_retries,
    )

    return {
        "location": location,
        "observation": observation,
        "signals": signals,
        "hypotheses": hypotheses,
        "open_questions": open_questions,
        "narrative": narrative,
        "execution_trace": execution_trace,
        "total_ms": total_ms,
        "map_layers": map_layers,
        "coverage": coverage_for_location(location, observation),
    }

@router.get("/why/stream")
async def stream_why_explanation(
    lat: float = Query(...),
    lon: float = Query(...),
    zip_code: Optional[str] = Query(None),
    city: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    country_code: Optional[str] = Query(None, description="ISO country code, when known"),
    observation_token: Optional[str] = Query(None, description="Signed token from /api/aqi"),
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
        "name": name or f"{city or 'Location'}, {state or ''}".strip(", ") or f"{lat:.2f}, {lon:.2f}",
        "country_code": country_code,
    }

    if ENFORCE_OBSERVATION_TOKENS:
        if aqi is None:
            raise HTTPException(
                status_code=400,
                detail="observation_token is missing, expired, or does not match this observation.",
            )
        client_observation = {
            "source": "AirNow",
            "aqi": aqi,
            "primary_pollutant": primary_pollutant or "PM2.5",
            "category": category or "Moderate",
            "pollutants": {},
        }
        if not verify_observation_token(
            observation_token,
            location,
            client_observation,
            OBSERVATION_TOKEN_SECRET,
        ):
            raise HTTPException(
                status_code=400,
                detail="observation_token is missing, expired, or does not match this observation.",
            )

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
        cache_hit = False
        llm_generated = False

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
            "total_ms": total_ms,
            "coverage": coverage_for_location(location, observation),
        }
        yield f"event: signals_ready\ndata: {json.dumps(signals_payload)}\n\n"

        # DeepSeek Briefing Token Streaming (Directly into narrative box)
        cached_narrative = get_cached_narrative(cache_key)
        full_narrative = ""
        stream_meta = StreamMeta()
        why_event_id: Optional[int] = None
        evidence_payload: Optional[Dict[str, Any]] = None

        def _record_stream() -> Optional[int]:
            record_signal_events("/api/why/stream", execution_trace)
            top_h = hypotheses[0] if hypotheses else None
            return record_why(
                "/api/why/stream",
                cache_hit=cache_hit,
                llm_generated=llm_generated,
                llm_cost_usd=estimate_llm_cost(full_narrative) if llm_generated else 0.0,
                judge_verdict=None,  # async judge writes this back via why_event_id
                country_code=location.get("country_code"),
                loc_key=loc_key,
                total_ms=total_ms,
                top_hypothesis=top_h.get("id") if top_h else None,
                top_confidence=top_h.get("confidence") if top_h else None,
                # Streamed responses don't surface usage; estimate both sides so
                # the prompt-token cost is not silently dropped (chars≈tokens/4).
                llm_input_tokens=max(1, len(json.dumps(evidence_payload)) // 4) if llm_generated else None,
                llm_output_tokens=max(1, len(full_narrative) // 4) if llm_generated else None,
                fallback_used=stream_meta.fell_back,
            )

        if cached_narrative:
            cache_hit = True
            full_narrative = cached_narrative
            # Return cached narrative immediately without word streaming
            yield f"event: llm_token\ndata: {json.dumps({'token': cached_narrative})}\n\n"
            _record_stream()
        else:
            llm_generated = bool(DEEPSEEK_API_KEY)
            async for token in generate_narrative_briefing_stream(location, observation, signals, hypotheses, open_questions, stream_meta=stream_meta):
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
            why_event_id = _record_stream()
            # Run non-blocking narrative judge check in background; it writes
            # the verdict back to the why_events row captured above.
            _schedule_background(_async_judge_streamed_narrative(evidence_payload, full_narrative, cache_key, why_event_id))

        yield f"event: complete\ndata: {json.dumps({'narrative': full_narrative, 'execution_trace': execution_trace, 'coverage': coverage_for_location(location, observation)})}\n\n"

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")
