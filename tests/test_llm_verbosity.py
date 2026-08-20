"""Static regression guards for briefing verbosity (no API key required).

Briefings drifted to 150-200+ words against a "2 short paragraphs" prompt
with a 600-token budget. These guards pin the two levers that keep them
tight: an explicit word cap in SYSTEM_PROMPT and a hard max_tokens at both
the plain and streaming call sites.
"""

import inspect

from backend import llm


def test_system_prompt_has_word_cap():
    assert "110 words" in llm.SYSTEM_PROMPT
    assert "never an essay" in llm.SYSTEM_PROMPT


def test_briefing_call_sites_use_tight_token_budget():
    plain = inspect.getsource(llm.generate_narrative_briefing)
    stream = inspect.getsource(llm.generate_narrative_briefing_stream)
    for source in (plain, stream):
        assert "max_tokens=280" in source
        assert "max_tokens=600" not in source


def test_fallback_narrative_stays_short():
    narrative = llm.generate_fallback_narrative(
        {"name": "Testville"},
        {"aqi": 155, "category": "Unhealthy", "primary_pollutant": "PM2.5"},
        [{"id": "aerosol_plume", "status": "present"}],
        [{"title": "Wildfire smoke", "confidence": "high", "support": ["Hotspots upwind"]}],
        ["Unverified news fire nearby"],
    )
    assert len(narrative.split()) <= 120
