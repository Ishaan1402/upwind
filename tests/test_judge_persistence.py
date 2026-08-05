"""The streamed judge must persist its verdict, not just log it."""

import asyncio
from unittest.mock import AsyncMock, patch

from backend.routers.why import _async_judge_streamed_narrative


def test_streamed_judge_persists_verdict():
    verdict = {"verdict": "fail", "reasoning": "x", "hallucinations": ["y"], "leaked_jargon": []}
    with patch("backend.routers.why.judge_narrative", new_callable=AsyncMock, return_value=verdict) as judge, \
         patch("backend.routers.why.update_cached_verdict", return_value=None) as persist:
        asyncio.run(_async_judge_streamed_narrative({"signals": []}, "narrative", "why_k_20260804_19"))

    judge.assert_awaited_once()
    persist.assert_called_once_with("why_k_20260804_19", verdict)


def test_streamed_judge_fail_heals_cache():
    verdict = {"verdict": "fail", "reasoning": "x", "hallucinations": ["y"], "leaked_jargon": []}
    evidence = {
        "location": {},
        "observation": {"aqi": 85},
        "signals": [],
        "hypotheses": [],
        "open_questions": [],
    }
    with patch("backend.routers.why.judge_narrative", new_callable=AsyncMock, return_value=verdict), \
         patch("backend.routers.why.generate_fallback_narrative", return_value="fallback text") as fallback, \
         patch("backend.routers.why.set_cached_narrative", return_value=None) as persist, \
         patch("backend.routers.why.update_cached_verdict", return_value=None):
        asyncio.run(_async_judge_streamed_narrative(evidence, "bad narrative", "why_k_20260804_19"))

    fallback.assert_called_once()
    persist.assert_called_once_with("why_k_20260804_19", "fallback text", evidence, verdict)


def test_streamed_judge_pass_does_not_heal_cache():
    verdict = {"verdict": "pass", "reasoning": "ok", "hallucinations": [], "leaked_jargon": []}
    evidence = {
        "location": {},
        "observation": {"aqi": 85},
        "signals": [],
        "hypotheses": [],
        "open_questions": [],
    }
    with patch("backend.routers.why.judge_narrative", new_callable=AsyncMock, return_value=verdict), \
         patch("backend.routers.why.generate_fallback_narrative", return_value="fallback text") as fallback, \
         patch("backend.routers.why.set_cached_narrative", return_value=None) as persist, \
         patch("backend.routers.why.update_cached_verdict", return_value=None) as update:
        asyncio.run(_async_judge_streamed_narrative(evidence, "good narrative", "why_k_20260804_19"))

    fallback.assert_not_called()
    persist.assert_not_called()
    update.assert_called_once_with("why_k_20260804_19", verdict)
