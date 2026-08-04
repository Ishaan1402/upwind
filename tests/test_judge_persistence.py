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
