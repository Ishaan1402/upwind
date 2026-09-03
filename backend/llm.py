import json
from dataclasses import dataclass
from typing import Dict, Any, List, AsyncGenerator, Optional
from backend.config import DEEPSEEK_API_KEY


@dataclass
class BriefingResult:
    """Structured result of a briefing generation.

    Carries the narrative plus real token usage from the API response and a
    ``fell_back`` flag set when the deterministic fallback was used.
    """
    narrative: str
    input_tokens: int = 0
    output_tokens: int = 0
    fell_back: bool = False


@dataclass
class StreamMeta:
    """Out-param for streamed briefings.

    A stream yields tokens, so it cannot return a value alongside them; this
    small typed out-param is unavoidable for recording the ``fell_back`` flag.
    """
    fell_back: bool = False

SYSTEM_PROMPT = """You are a friendly, engaging air quality expert explaining air pollution evidence to an everyday user.

Rules:
1. Tone: Light, clear, and human. Speak directly to the user.
2. Structure: 2 short, readable paragraphs. Total length MUST stay under ~110 words (roughly 2-4 sentences per paragraph) — a tight briefing, never an essay. Every sentence must earn its place.
3. NO HEADERS OR TITLES: Do NOT include titles, dates, or headers (no "**Briefing:**", no "# Titles").
4. NO JARGON OR INTERNAL SCORES: NEVER mention raw acronyms or backend numbers (no "hypothesis score of 90", no "AOD 0.68", no "FIRMS", no "photochemical ozone formation"). Translate technical data into plain English (e.g. "a thick layer of overhead smoke", "satellite fire trackers", "cool temperatures suppressing ozone").
5. Grounding: Ground every statement strictly in the provided signals[], hypotheses[], and open_questions[]. Never invent unlisted fires or industrial plants.
6. Actionable Takeaway: End with a warm, 1-sentence practical health tip tailored to the AQI category.
7. Good AQI Rule (AQI <= 50): When ground AQI is 50 or below, state clearly in your very first sentence that surface air is healthy and clean. Do NOT write as if there is elevated air pollution to explain. If satellite feeds or news mention distant fires or high-altitude smoke, frame them strictly as 'aloft or regional context' that has not impacted ground breathing air.
8. Fire Name Corroboration: Only state that a named wildfire is affecting ground air if support lists nearby thermal hotspots, a NOAA smoke-plume analysis, or a federal incident registry listing. News or haze alone is not enough; overhead haze without verified fire evidence should be framed as regional/urban particles, not a confirmed distant wildfire.
9. No Invented Transport: Do NOT invent transport mechanisms (e.g. trapping by light winds or shallow boundary layer) unless those appear in the top hypothesis's support list.
10. Unverified News Handling: If open_questions mention an unverified news fire, treat it as uncertainty or distant context, not the primary cause.
11. Reading Conflicts: If open_questions mention a mismatch between the reported AQI and a monitor reading, acknowledge it in one plain sentence that explains the difference (the AQI is a longer-term average; the monitor reading is current), and do not present it as a reason to doubt the overall AQI verdict.
12. No Invented Fire Size: Never assign a size, acreage, or rank to a fire whose name is sourced only from news headlines. If a fire's size is unknown, say the size is unknown - do not invent one, and do not describe it as a cluster of smaller fires.
"""

def generate_fallback_narrative(
    location: Dict[str, Any],
    observation: Dict[str, Any],
    signals: List[Dict[str, Any]],
    hypotheses: List[Dict[str, Any]],
    open_questions: List[str]
) -> str:
    loc_name = location.get("name", "this location")
    aqi_val = observation.get("aqi", 0)
    category = observation.get("category", "Good")
    pollutant = observation.get("primary_pollutant", "PM2.5")

    if aqi_val <= 50:
        text = f"Air quality in {loc_name} is currently Good (AQI {aqi_val}). Breathing air at the ground level is clean and healthy. "
        # Check all satellite and ground smoke feeds for clean ground aloft context
        aloft_feeds = ("aerosol_plume", "firms_upwind", "hms_smoke", "wfigs_incident")
        if any(s.get("status") == "present" for s in signals if s.get("id") in aloft_feeds):
            text += "Satellites detect some high-altitude smoke or regional fire activity, but it remains aloft without affecting ground air. "
        text += "\n\nIt's a great day to enjoy outdoor activities!"
        return text

    top_h = hypotheses[0] if hypotheses else None
    pollutant_desc = "coarse dust particulate (PM10)" if pollutant == "PM10" else ("fine particulate pollution (PM2.5)" if "PM" in pollutant else pollutant)
    text = f"Air quality in {loc_name} is currently {category} (AQI {aqi_val}), with {pollutant_desc} as the primary reporting pollutant. "

    if top_h:
        text += f"The primary cause is {top_h['title'].lower()} ({top_h['confidence']} confidence). "
        if top_h["support"]:
            clean_support = [
                s.replace("Dense atmospheric column particle plume detected", "Thick overhead smoke plume detected")
                 .replace("AOD", "particle density")
                 .replace("NASA FIRMS", "satellite fire tracking")
                 .replace("FIRMS", "satellite fire tracking")
                for s in top_h["support"]
            ]
            text += "Key observations: " + "; ".join(clean_support) + ". "
    else:
        text += "No strong attribution signals were identified. "

    if open_questions:
        text += "\n\n" + " ".join(open_questions)

    if aqi_val > 100:
        text += "\n\nIf you have sensitive lungs or asthma, consider keeping windows closed and taking it easy outdoors today."
    else:
        text += "\n\nIt is a good day to monitor local air trends if you are sensitive to air pollution."

    return text

async def generate_narrative_briefing(
    location: Dict[str, Any],
    observation: Dict[str, Any],
    signals: List[Dict[str, Any]],
    hypotheses: List[Dict[str, Any]],
    open_questions: List[str],
) -> BriefingResult:
    """Generate a briefing and return a ``BriefingResult`` carrying the
    narrative, real token usage from the API response, and a ``fell_back``
    flag when the deterministic fallback was used — lets the caller record
    honest cost without mutating an out-param."""
    if not DEEPSEEK_API_KEY:
        return BriefingResult(
            narrative=generate_fallback_narrative(location, observation, signals, hypotheses, open_questions),
            fell_back=True,
        )

    evidence_payload = {
        "location": location,
        "observation": observation,
        "signals": signals,
        "hypotheses": hypotheses,
        "open_questions": open_questions
    }

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url="https://api.deepseek.com", api_key=DEEPSEEK_API_KEY)
        
        response = await client.chat.completions.create(
            model="deepseek-v4-flash",
            extra_body={"thinking": {"type": "disabled"}},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Brief the user on this evidence package in a warm, light, conversational voice without headers or internal scores:\n{json.dumps(evidence_payload, indent=2)}"}
            ],
            max_tokens=280,
            temperature=0.4
        )
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
        narrative = response.choices[0].message.content.strip()
        if narrative.startswith("**Briefing") or narrative.startswith("#"):
            lines = narrative.split("\n")
            narrative = "\n".join([l for l in lines if not l.startswith("**Briefing") and not l.startswith("#")]).strip()
        if not narrative:
            return BriefingResult(
                narrative=generate_fallback_narrative(location, observation, signals, hypotheses, open_questions),
                fell_back=True,
            )
        return BriefingResult(
            narrative=narrative,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as e:
        print(f"[LLM Generation Warning]: {e}", flush=True)
        return BriefingResult(
            narrative=generate_fallback_narrative(location, observation, signals, hypotheses, open_questions),
            fell_back=True,
        )

async def generate_narrative_briefing_stream(
    location: Dict[str, Any],
    observation: Dict[str, Any],
    signals: List[Dict[str, Any]],
    hypotheses: List[Dict[str, Any]],
    open_questions: List[str],
    stream_meta: Optional[StreamMeta] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream narrative briefing tokens in real-time from DeepSeek API (stream=True).
    Yields string tokens directly. ``stream_meta`` (optional ``StreamMeta``
    out-param) has its ``fell_back`` flag set when the deterministic fallback
    was used; token counts for streams are estimated by the caller (chars/4)
    since usage is not reliably surfaced. Note: a stream cannot return a value
    alongside the yielded tokens, so the small typed out-param is unavoidable.
    """
    if not DEEPSEEK_API_KEY:
        if stream_meta is not None:
            stream_meta.fell_back = True
        fallback = generate_fallback_narrative(location, observation, signals, hypotheses, open_questions)
        words = fallback.split(" ")
        for w in words:
            yield w + " "
        return

    evidence_payload = {
        "location": location,
        "observation": observation,
        "signals": signals,
        "hypotheses": hypotheses,
        "open_questions": open_questions
    }

    emitted_any = False
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url="https://api.deepseek.com", api_key=DEEPSEEK_API_KEY)
        
        stream_response = await client.chat.completions.create(
            model="deepseek-v4-flash",
            extra_body={"thinking": {"type": "disabled"}},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Brief the user on this evidence package in a warm, light, conversational voice without headers or internal scores:\n{json.dumps(evidence_payload, indent=2)}"}
            ],
            max_tokens=280,
            temperature=0.4,
            stream=True
        )

        async for chunk in stream_response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                emitted_any = True
                yield chunk.choices[0].delta.content
    except Exception as e:
        print(f"[LLM Stream Warning]: {e}", flush=True)
        if emitted_any:
            # A fallback can't be appended to a partially-streamed narrative
            # without corrupting it; end the stream cleanly instead.
            return
        if stream_meta is not None:
            stream_meta.fell_back = True
        fallback = generate_fallback_narrative(location, observation, signals, hypotheses, open_questions)
        for w in fallback.split(" "):
            yield w + " "
