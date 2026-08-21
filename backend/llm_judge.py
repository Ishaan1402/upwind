import os
import json
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from backend.config import GROQ_API_KEY

DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
FALLBACK_JUDGE_MODEL = "llama-3.1-8b-instant"
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.groq.com/openai/v1")


@dataclass
class JudgeResult:
    """Structured judge verdict with token usage metadata.

    ``to_dict()`` returns the JSON-serializable dict shape callers previously
    received directly from ``judge_narrative``.
    """
    verdict: str
    reasoning: str = ""
    hallucinations: List[str] = field(default_factory=list)
    leaked_jargon: List[str] = field(default_factory=list)
    judge_model: Optional[str] = None
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reasoning": self.reasoning,
            "hallucinations": self.hallucinations,
            "leaked_jargon": self.leaked_jargon,
            "judge_model": self.judge_model,
        }


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

async def judge_narrative(evidence_payload: Dict[str, Any], narrative: str) -> JudgeResult:
    """Judge a narrative against its evidence and return a ``JudgeResult``
    carrying the verdict plus real token usage from the API response so the
    dashboard can track honest judge cost."""
    if not GROQ_API_KEY:
        return JudgeResult(
            verdict="skipped",
            reasoning="GROQ_API_KEY not configured",
            hallucinations=[],
            leaked_jargon=[],
        )

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
            usage = getattr(response, "usage", None)
            judge_input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
            judge_output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
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
            return JudgeResult(
                verdict=verdict["verdict"],
                reasoning=verdict.get("reasoning") or "",
                hallucinations=verdict.get("hallucinations") or [],
                leaked_jargon=verdict.get("leaked_jargon") or [],
                judge_model=model,
                judge_input_tokens=judge_input_tokens,
                judge_output_tokens=judge_output_tokens,
            )
        except Exception as e:
            last_error = e
            err_msg = str(e)
            retryable = (
                isinstance(e, ValueError)
                or "429" in err_msg
                or "rate_limit" in err_msg.lower()
                or "500" in err_msg
                or "502" in err_msg
                or "503" in err_msg
                or "504" in err_msg
                or "timeout" in err_msg.lower()
                or "timed out" in err_msg.lower()
                or "connection" in err_msg.lower()
            )
            if retryable:
                print(f"[LLM Judge Warning]: Model {model} unavailable ({err_msg[:120]}), trying fallback...")
                continue
            else:
                print(f"[LLM Judge Warning]: {e}")
                break

    return JudgeResult(
        verdict="unknown",
        reasoning=f"Judge unavailable: {last_error}",
        hallucinations=[],
        leaked_jargon=[],
    )
