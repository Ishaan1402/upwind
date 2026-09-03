"""Deterministic fallback narrative regression tests."""

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend import llm_judge
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


def _fake_openai(fake_create):
    class FakeCompletions:
        async def create(self, **kwargs):
            return await fake_create(**kwargs)

    class FakeClient:
        def __init__(self, *a, **k):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    return FakeClient


def _judge_response(content):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


def test_judge_falls_back_on_5xx():
    """Track Part 2 #10: a 5xx/timeout on the primary judge model must try the
    fallback model instead of aborting."""
    calls = {"n": 0}

    async def fake_create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Exception("HTTP 503 Service Unavailable")
        return _judge_response('{"verdict": "pass", "hallucinations": [], "leaked_jargon": []}')

    with patch.object(llm_judge, "GROQ_API_KEY", "test-key"), \
         patch("openai.AsyncOpenAI", _fake_openai(fake_create)):
        verdict = asyncio.run(llm_judge.judge_narrative({"signals": []}, "A narrative."))

    assert verdict.verdict == "pass"
    assert verdict.judge_model == llm_judge.FALLBACK_JUDGE_MODEL
    assert calls["n"] == 2


def test_judge_falls_back_on_404_decommissioned_model():
    """Groq returns 404 when a model ID is retired; that must not abort the
    chain (this is what broke nightly Narrative Eval after 2026-08-16)."""
    calls = {"models": []}

    async def fake_create(**kwargs):
        calls["models"].append(kwargs["model"])
        if kwargs["model"] == llm_judge.DEFAULT_JUDGE_MODEL:
            raise Exception(
                "Error code: 404 - {'error': {'message': "
                "'The model `llama-3.3-70b-versatile` does not exist or you do not have access to it.', "
                "'type': 'invalid_request_error', 'code': 'model_not_found'}}"
            )
        return _judge_response('{"verdict": "pass", "hallucinations": [], "leaked_jargon": []}')

    with patch.object(llm_judge, "GROQ_API_KEY", "test-key"), \
         patch("openai.AsyncOpenAI", _fake_openai(fake_create)):
        verdict = asyncio.run(llm_judge.judge_narrative({"signals": []}, "A narrative."))

    assert verdict.verdict == "pass"
    assert verdict.judge_model == llm_judge.FALLBACK_JUDGE_MODEL
    assert calls["models"] == [llm_judge.DEFAULT_JUDGE_MODEL, llm_judge.FALLBACK_JUDGE_MODEL]


def test_shipped_judge_defaults_are_current_groq_ids():
    src = inspect.getsource(llm_judge)
    assert 'os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")' in src
    assert 'os.getenv("JUDGE_FALLBACK_MODEL", "openai/gpt-oss-20b")' in src
    kwargs = llm_judge._completion_kwargs("openai/gpt-oss-120b")
    assert kwargs["max_tokens"] >= 2048
    assert kwargs["extra_body"]["reasoning_effort"] == "low"


def test_retryable_judge_error_treats_404_as_fallback():
    class NotFound(Exception):
        status_code = 404

    assert llm_judge._is_retryable_judge_error(NotFound("model gone"))
    assert llm_judge._is_retryable_judge_error(Exception("Error code: 404 - model_not_found"))
    assert not llm_judge._is_retryable_judge_error(Exception("Error code: 400 - bad_request"))
