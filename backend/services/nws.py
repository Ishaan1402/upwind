"""
NWS active-alerts dust confirmation.

The National Weather Service publishes an active-alerts GeoJSON feed filterable
by point. A "Dust Storm Warning", "Blowing Dust Warning", "Dust Advisory", or
"Blowing Dust Advisory" overlapping the target location is a high-precision,
low-recall confirmation of windblown dust: when present it confirms dust, and
its absence proves nothing.
"""

from typing import Any, Dict, List, Optional

import httpx

# NWS REQUIRES a descriptive User-Agent on every request.
NWS_USER_AGENT = "upwind-app/1.0 (contact@getupwind.me)"
NWS_ALERTS_URL = "https://api.weather.gov/alerts/active"
NWS_TIMEOUT_S = 4.0

# Case-insensitive event-name substrings (the ``event`` field of an alert
# feature) that confirm dust.
DUST_EVENT_NAMES = (
    "Dust Storm Warning",
    "Blowing Dust Warning",
    "Dust Advisory",
    "Blowing Dust Advisory",
)


def _unavailable(reason: str) -> Dict[str, Any]:
    return {
        "status": "unavailable",
        "event": None,
        "headline": None,
        "severity": None,
        "details": reason,
    }


def _find_dust_alert(features: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the properties of the first alert whose event names dust, else None."""
    for feature in features or []:
        props = feature.get("properties") or {}
        event = props.get("event")
        if not event:
            continue
        for name in DUST_EVENT_NAMES:
            if name.lower() in str(event).lower():
                return props
    return None


def _headline(props: Dict[str, Any]) -> Optional[str]:
    """The alert headline: properties.headline, falling back to the first
    parameters.NWSheadline entry (some product types carry it there)."""
    headline = props.get("headline")
    if headline:
        return headline
    params = props.get("parameters") or {}
    for key in ("NWSheadline", "Headline"):
        values = params.get(key) or []
        if values:
            return values[0]
    return None


async def fetch_dust_alert(lat: float, lon: float) -> Dict[str, Any]:
    """
    Query NWS active alerts for a point and return the dust confirmation.

    Never raises: any transport/parse/HTTP failure returns status
    "unavailable". A healthy response with no dust-named event returns
    "absent" (these are one-sided confirmations - absence is not evidence).
    """
    headers = {
        "User-Agent": NWS_USER_AGENT,
        "Accept": "application/geo+json, application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=NWS_TIMEOUT_S, headers=headers) as client:
            resp = await client.get(NWS_ALERTS_URL, params={"point": f"{lat},{lon}"})
            if resp.status_code != 200:
                return _unavailable(f"NWS alerts feed HTTP {resp.status_code}")
            data = resp.json()
            if not isinstance(data, dict):
                return _unavailable("Unexpected NWS alerts payload")
            props = _find_dust_alert(data.get("features") or [])
            if props is not None:
                return {
                    "status": "present",
                    "event": props.get("event"),
                    "headline": _headline(props),
                    "severity": props.get("severity"),
                }
            return {"status": "absent", "event": None, "headline": None, "severity": None}
    except Exception as e:
        return _unavailable(str(e) or "NWS alerts feed unreachable")
