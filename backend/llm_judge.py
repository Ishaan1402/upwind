import os
import json
import re
from typing import Dict, Any
from backend.config import GROQ_API_KEY

DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
FALLBACK_JUDGE_MODEL = "llama-3.1-8b-instant"
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.groq.com/openai/v1")

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checker reviewing an AI-generated air quality narrative against the evidence it was supposed to be based on.

Check the narrative against these rules:
1. Grounding: every factual claim (fire names, locations, pollutant levels, causes) must be traceable to the provided signals/hypotheses/open_questions. Flag anything invented.
2. No leaked jargon: the narrative must not contain raw technical terms like "AOD", "FIRMS", "HMS", "OpenAQ", "hypothesis score", numeric confidence percentages, or raw units like "µg/m³", "ppb", "ppm".
3. No headers/titles: no "**Briefing:**", no markdown headers, no dates.
4. Structure: should read as 2 short paragraphs plus a brief actionable health tip at the end.
5. AQI/monitor conflicts: if the evidence includes an open question about a mismatch
   between the reported AQI and a monitor reading, the narrative must acknowledge it in
   one plain sentence (the AQI is a longer-term average; the monitor reading is current)
   and must not claim the AQI reading is wrong.
6. Fire size accuracy: if a fire name appears in the evidence only as a news mention
   (e.g. "Recent news mention of 'X'" or a news-tagged place pointer), the narrative
   must NOT assign that fire a size, acreage, containment, or rank, and must NOT
   describe it as a cluster of smaller fires. Flag any invented size or rank as a
   hallucination.

Respond ONLY with a JSON object with exactly these fields:
{
  "grounded": true or false,
  "hallucinations": ["list of specific unsupported claims found, empty if none"],
  "leaked_jargon": ["list of raw technical terms found, empty if none"],
  "has_disallowed_headers": true or false,
  "has_actionable_tip": true or false,
  "verdict": "pass" or "fail",
  "reasoning": "one or two sentence explanation of the verdict"
}
"""

async def judge_narrative(evidence_payload: Dict[str, Any], narrative: str) -> Dict[str, Any]:
    if not GROQ_API_KEY:
        return {"verdict": "skipped", "reasoning": "GROQ_API_KEY not configured", "hallucinations": [], "leaked_jargon": []}

    models_to_try = [DEFAULT_JUDGE_MODEL, FALLBACK_JUDGE_MODEL]
    
    # Avoid duplicating if DEFAULT_JUDGE_MODEL is already FALLBACK_JUDGE_MODEL
    if DEFAULT_JUDGE_MODEL == FALLBACK_JUDGE_MODEL:
        models_to_try = [DEFAULT_JUDGE_MODEL]

    last_error = None

    for model in models_to_try:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(base_url=JUDGE_BASE_URL, api_key=GROQ_API_KEY)

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Evidence package:\n{json.dumps(evidence_payload, indent=2)}\n\nNarrative to review:\n{narrative}"}
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.0
            )
            raw = response.choices[0].message.content.strip()
            
            # Strip reasoning <think>...</think> tags if present
            raw_clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

            # Parse JSON defensively
            try:
                verdict = json.loads(raw_clean)
            except json.JSONDecodeError:
                match = re.search(r"\{[^{}]*\"verdict\"[^{}]*\}", raw_clean, re.DOTALL) or re.search(r"\{.*\}", raw_clean, re.DOTALL)
                if match:
                    verdict = json.loads(match.group(0))
                else:
                    raise ValueError("Could not parse JSON response from judge")

            verdict.setdefault("verdict", "unknown")
            verdict["judge_model"] = model
            return verdict
        except Exception as e:
            last_error = e
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg.lower():
                print(f"[LLM Judge Warning]: Model {model} hit rate limit, trying fallback...")
                continue
            else:
                print(f"[LLM Judge Warning]: {e}")
                break

    return {"verdict": "unknown", "reasoning": f"Judge unavailable: {last_error}", "hallucinations": [], "leaked_jargon": []}
