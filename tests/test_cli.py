import pytest
import asyncio
from backend.cli import parse_zip_inputs, fetch_zip_briefing

def test_parse_zip_inputs():
    res1 = parse_zip_inputs(["90210", "94103"])
    assert res1 == ["90210", "94103"]

    res2 = parse_zip_inputs(["90210, 94103,10001"])
    assert res2 == ["90210", "94103", "10001"]

    res3 = parse_zip_inputs(["90210 94103"])
    assert res3 == ["90210", "94103"]

def test_fetch_zip_briefing():
    result = asyncio.run(fetch_zip_briefing("90210", use_cache=True))
    assert result["zip"] == "90210"
    assert "Beverly Hills" in result["location"]
    assert "observation" in result
    assert result["observation"]["aqi"] >= 0
    assert "narrative" in result
    assert isinstance(result["narrative"], str)

def test_invalid_zip_briefing():
    result = asyncio.run(fetch_zip_briefing("invalid_zip_xyz", use_cache=True))
    assert result["zip"] == "invalid_zip_xyz"
    assert "error" in result
