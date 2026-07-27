import pytest
import asyncio
from backend.engine.score import score_hypotheses
from backend.llm import generate_narrative_briefing
from backend.llm_judge import judge_narrative
from backend.config import GROQ_API_KEY
from tests.fixtures import (
    SMOKE_LOCATION, SMOKE_OBSERVATION, SMOKE_SIGNALS,
    OZONE_LOCATION, OZONE_OBSERVATION, OZONE_SIGNALS,
    DUST_LOCATION, DUST_OBSERVATION, DUST_SIGNALS,
    STAGNATION_LOCATION, STAGNATION_OBSERVATION, STAGNATION_SIGNALS,
    URBAN_LOCATION, URBAN_OBSERVATION, URBAN_SIGNALS
)

pytestmark = pytest.mark.skipif(not GROQ_API_KEY, reason="GROQ_API_KEY not configured")

def _run_scenario(location, observation, signals):
    hypotheses, open_questions = score_hypotheses(observation, signals)
    narrative = asyncio.run(generate_narrative_briefing(location, observation, signals, hypotheses, open_questions))
    verdict = asyncio.run(judge_narrative(
        {"location": location, "observation": observation, "signals": signals, "hypotheses": hypotheses, "open_questions": open_questions},
        narrative
    ))
    return narrative, verdict

def test_judge_smoke_scenario_passes():
    narrative, verdict = _run_scenario(SMOKE_LOCATION, SMOKE_OBSERVATION, SMOKE_SIGNALS)
    assert verdict["verdict"] != "fail", f"Judge flagged smoke narrative: {verdict.get('reasoning')} / hallucinations={verdict.get('hallucinations')}"

def test_judge_ozone_scenario_passes():
    narrative, verdict = _run_scenario(OZONE_LOCATION, OZONE_OBSERVATION, OZONE_SIGNALS)
    assert verdict["verdict"] != "fail", f"Judge flagged ozone narrative: {verdict.get('reasoning')} / hallucinations={verdict.get('hallucinations')}"

def test_judge_dust_scenario_passes():
    narrative, verdict = _run_scenario(DUST_LOCATION, DUST_OBSERVATION, DUST_SIGNALS)
    assert verdict["verdict"] != "fail", f"Judge flagged dust narrative: {verdict.get('reasoning')} / hallucinations={verdict.get('hallucinations')}"

def test_judge_stagnation_scenario_passes():
    narrative, verdict = _run_scenario(STAGNATION_LOCATION, STAGNATION_OBSERVATION, STAGNATION_SIGNALS)
    assert verdict["verdict"] != "fail", f"Judge flagged stagnation narrative: {verdict.get('reasoning')} / hallucinations={verdict.get('hallucinations')}"

def test_judge_urban_scenario_passes():
    narrative, verdict = _run_scenario(URBAN_LOCATION, URBAN_OBSERVATION, URBAN_SIGNALS)
    assert verdict["verdict"] != "fail", f"Judge flagged urban narrative: {verdict.get('reasoning')} / hallucinations={verdict.get('hallucinations')}"
