import json
from typing import Dict, Any, List, AsyncGenerator
from backend.config import DEEPSEEK_API_KEY

SYSTEM_PROMPT = """You are a friendly, engaging air quality expert explaining air pollution evidence to an everyday user.

Rules:
1. Tone: Light, clear, and human. Speak directly to the user.
2. Structure: 2 short, readable paragraphs.
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
    open_questions: List[str]
) -> str:
    if not DEEPSEEK_API_KEY:
        return generate_fallback_narrative(location, observation, signals, hypotheses, open_questions)

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
            max_tokens=600,
            temperature=0.4
        )
        narrative = response.choices[0].message.content.strip()
        if narrative.startswith("**Briefing") or narrative.startswith("#"):
            lines = narrative.split("\n")
            narrative = "\n".join([l for l in lines if not l.startswith("**Briefing") and not l.startswith("#")]).strip()
        return narrative if narrative else generate_fallback_narrative(location, observation, signals, hypotheses, open_questions)
    except Exception as e:
        print(f"[LLM Generation Warning]: {e}", flush=True)
        return generate_fallback_narrative(location, observation, signals, hypotheses, open_questions)

async def generate_narrative_briefing_stream(
    location: Dict[str, Any],
    observation: Dict[str, Any],
    signals: List[Dict[str, Any]],
    hypotheses: List[Dict[str, Any]],
    open_questions: List[str]
) -> AsyncGenerator[str, None]:
    """
    Stream narrative briefing tokens in real-time from DeepSeek API (stream=True).
    Yields string tokens directly.
    """
    if not DEEPSEEK_API_KEY:
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
            max_tokens=600,
            temperature=0.4,
            stream=True
        )

        async for chunk in stream_response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        print(f"[LLM Stream Warning]: {e}", flush=True)
        fallback = generate_fallback_narrative(location, observation, signals, hypotheses, open_questions)
        for w in fallback.split(" "):
            yield w + " "
