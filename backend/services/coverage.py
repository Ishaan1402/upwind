"""Location coverage metadata shared by the AQI and Why APIs."""

from typing import Any, Dict, Optional


def coverage_for_location(
    location: Dict[str, Any],
    observation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return an explicit coverage object so clients know what mode they are in."""
    country_code = (location.get("country_code") or "").strip().lower()
    source = (observation or {}).get("source", "")

    if country_code == "us":
        mode = "us"
        aqi_index = "US EPA AQI"
        sources = ["AirNow", "Open-Meteo", "OpenAQ", "FIRMS"]
        disclaimer = "US EPA AQI from AirNow monitors, supplemented by model and satellite feeds."
    elif country_code:
        mode = "international"
        aqi_index = "US-style AQI (Open-Meteo)"
        sources = ["Open-Meteo", "OpenAQ", "FIRMS"]
        disclaimer = (
            "Outside the US, AQI is the US-style index from Open-Meteo, not a local "
            "regulatory index. Attribution remains an evidence-based hypothesis."
        )
    elif source == "AirNow":
        mode = "us"
        country_code = "us"
        aqi_index = "US EPA AQI"
        sources = ["AirNow", "Open-Meteo", "OpenAQ", "FIRMS"]
        disclaimer = "US EPA AQI from AirNow monitors, supplemented by model and satellite feeds."
    else:
        mode = "unknown"
        aqi_index = "US-style AQI (Open-Meteo)"
        sources = ["Open-Meteo", "OpenAQ", "FIRMS"]
        disclaimer = (
            "Location country could not be confirmed. Data is a best-effort mix of "
            "global model, satellite, and monitor feeds."
        )

    return {
        "country_code": country_code or None,
        "mode": mode,
        "aqi_index": aqi_index,
        "sources": sources,
        "disclaimer": disclaimer,
    }
