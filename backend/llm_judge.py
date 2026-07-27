import json
import re
from typing import Dict, Any
from backend.config import GROQ_API_KEY

JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_BASE_URL = "https://api.groq.com/openai/v1"

JUDGE_SYSTEM_PROMPT = """Reasoning: medium
You are a strict fact-checker reviewing an AI-generated air quality narrative against the evidence it was supposed to be based on.

Check the narrative against these rules:
1. Grounding: every factual claim (fire names, locations, pollutant levels, causes) must be traceable to the provided signals/hypotheses/open_questions. Flag anything invented.
2. No leaked jargon: the narrative must not contain raw technical terms like "AOD", "FIRMS", "HMS", "hypothesis score", or numeric confidence percentages.
3. No headers/titles: no "**Briefing:**", no markdown headers, no dates.
4. Structure: should read as 2 short paragraphs plus a brief actionable health tip at the end.

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

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=JUDGE_BASE_URL, api_key=GROQ_API_KEY)

        response = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Evidence package:\n{json.dumps(evidence_payload, indent=2)}\n\nNarrative to review:\n{narrative}"}
            ],
            max_tokens=500,
            temperature=0.0
        )
        raw = response.choices[0].message.content.strip()
        
        # Parse JSON defensively
        try:
            verdict = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                verdict = json.loads(match.group(0))
            else:
                raise ValueError("Could not parse JSON response from judge")

        verdict.setdefault("verdict", "unknown")
        return verdict
    except Exception as e:
        print(f"[LLM Judge Warning]: {e}")
        return {"verdict": "unknown", "reasoning": f"Judge unavailable: {e}", "hallucinations": [], "leaked_jargon": []}
