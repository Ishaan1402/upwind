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
6. Actionable Takeaway: End with a warm, 1-sentence practical health tip tailored to the AQI category (e.g. if AQI is elevated, suggest keeping windows closed or taking it easy outdoors for sensitive lungs).
"""

def generate_fallback_narrative(
    location: Dict[str, Any],
    observation: Dict[str, Any],
    signals: List[Dict[str, Any]],
    hypotheses: List[Dict[str, Any]],
    open_questions: List[str]
) -> str:
    top_h = hypotheses[0] if hypotheses else None
    loc_name = location.get("name", "this location")
    aqi_val = observation.get("aqi", 0)
    category = observation.get("category", "Moderate")
    pollutant = observation.get("primary_pollutant", "PM2.5")

    text = f"Air quality in {loc_name} is currently {category} (AQI {aqi_val}), driven primarily by fine smoke dust ({pollutant}). "

    if top_h:
        text += f"The primary cause is {top_h['title'].lower()} ({top_h['confidence']} confidence). "
        if top_h["support"]:
            text += "Key observations: " + "; ".join(top_h["support"]) + ". "
    else:
        text += "No strong attribution signals were identified. "

    if open_questions:
        text += "\n\n" + " ".join(open_questions)

    if aqi_val > 100:
        text += "\n\nIf you have sensitive lungs or asthma, consider keeping windows closed and taking it easy outdoors today."

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
            model="deepseek-chat",
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
        print(f"[LLM Generation Warning]: {e}")
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
            model="deepseek-chat",
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
        print(f"[LLM Stream Warning]: {e}")
        fallback = generate_fallback_narrative(location, observation, signals, hypotheses, open_questions)
        for w in fallback.split(" "):
            yield w + " "
