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

from unittest.mock import patch, AsyncMock

def test_fetch_zip_briefing():
    mock_obs = {
        "aqi": 45,
        "primary_pollutant": "PM2.5",
        "category": "Good",
        "category_color": "#00e400",
        "category_text_color": "#000000",
        "timestamp": "2026-07-29T12:00:00Z",
        "sources": {"airnow": False, "open_meteo": True}
    }
    with patch("backend.cli.fetch_airnow_observation", new_callable=AsyncMock) as mock_airnow, \
         patch("backend.cli.fetch_openmeteo_aqi", new_callable=AsyncMock) as mock_openmeteo, \
         patch("backend.cli.generate_narrative_briefing", new_callable=AsyncMock) as mock_llm:
        mock_airnow.return_value = None
        mock_openmeteo.return_value = mock_obs
        mock_llm.return_value = "Beverly Hills has good air quality overall."

        result = asyncio.run(fetch_zip_briefing("90210", use_cache=False))
        assert result["zip"] == "90210"
        assert "Beverly Hills" in result["location"]
        assert "observation" in result
        assert result["observation"]["aqi"] == 45
        assert "narrative" in result
        assert isinstance(result["narrative"], str)

def test_invalid_zip_briefing():
    result = asyncio.run(fetch_zip_briefing("invalid_zip_xyz", use_cache=True))
    assert result["zip"] == "invalid_zip_xyz"
    assert "error" in result
