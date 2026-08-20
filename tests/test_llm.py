"""Deterministic fallback narrative regression tests."""

import pytest

from backend.llm import generate_fallback_narrative, SYSTEM_PROMPT
from backend.llm_judge import JUDGE_SYSTEM_PROMPT

LOC = {"name": "Test Town"}
GOOD_OBS = {"aqi": 35, "primary_pollutant": "PM2.5", "category": "Good"}


def _signals(*present_ids):
    signals = []
    for sid in ("aerosol_plume", "firms_upwind", "hms_smoke", "wfigs_incident"):
        signals.append({"id": sid, "status": "present" if sid in present_ids else "absent"})
    return signals


def test_good_aqi_fallback_leads_with_clean_air():
    text = generate_fallback_narrative(LOC, GOOD_OBS, _signals(), [], [])
    assert text.startswith("Air quality in Test Town is currently Good")
    assert "aloft" not in text


def test_good_aqi_hms_plume_gets_aloft_context():
    text = generate_fallback_narrative(LOC, GOOD_OBS, _signals("hms_smoke"), [], [])
    assert "clean and healthy" in text
    assert "aloft" in text


def test_good_aqi_wfigs_incident_gets_aloft_context():
    text = generate_fallback_narrative(LOC, GOOD_OBS, _signals("wfigs_incident"), [], [])
    assert "clean and healthy" in text
    assert "aloft" in text


@pytest.mark.honesty
def test_system_prompt_has_no_invented_fire_size_rule():
    """Track C Part 1: the narrative system prompt forbids inventing a size,
    acreage, or rank for news-sourced fire names and forbids the 'cluster of
    smaller fires' hallucination."""
    assert "Never assign a size, acreage, or rank" in SYSTEM_PROMPT
    assert "size is unknown" in SYSTEM_PROMPT
    assert "cluster of smaller fires" in SYSTEM_PROMPT


@pytest.mark.honesty
def test_judge_prompt_flags_invented_fire_size():
    """Track C Part 1: the judge prompt must flag narratives that attribute an
    invented size/acreage/rank to a news-only fire name."""
    assert "cluster of smaller fires" in JUDGE_SYSTEM_PROMPT
    assert "invented size" in JUDGE_SYSTEM_PROMPT.lower()
    assert "news mention" in JUDGE_SYSTEM_PROMPT
