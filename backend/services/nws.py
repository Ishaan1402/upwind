"""
NWS active-alerts dust confirmation.

The National Weather Service publishes an active-alerts GeoJSON feed filterable
by point. A "Dust Storm Warning", "Blowing Dust Warning", "Dust Advisory", or
"Blowing Dust Advisory" overlapping the target location is a high-precision,
low-recall confirmation of windblown dust: when present it confirms dust, and
its absence proves nothing.
"""

from datetime import datetime, timezone
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


def _find_dust_alerts(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Properties of every alert whose event names dust, in feed order."""
    matched: List[Dict[str, Any]] = []
    for feature in features or []:
        props = feature.get("properties") or {}
        event = props.get("event")
        if not event:
            continue
        for name in DUST_EVENT_NAMES:
            if name.lower() in str(event).lower():
                matched.append(props)
                break
    return matched


def _find_dust_alert(features: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the properties of the first alert whose event names dust, else None."""
    matched = _find_dust_alerts(features)
    return matched[0] if matched else None


def _parse_nws_time(value: Any) -> Optional[datetime]:
    """Parse an NWS alert timestamp (ISO-8601 with offset, or 'Z') into an aware
    UTC datetime; None when missing or malformed."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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
                # as_of is the most recent effective/sent time across ALL
                # matched dust alerts, not just the first (display) one.
                matched = _find_dust_alerts(data.get("features") or [])
                alert_times = [
                    t
                    for p in matched
                    for t in (_parse_nws_time(p.get("effective")), _parse_nws_time(p.get("sent")))
                    if t is not None
                ]
                result = {
                    "status": "present",
                    "event": props.get("event"),
                    "headline": _headline(props),
                    "severity": props.get("severity"),
                }
                if alert_times:
                    result["as_of"] = max(alert_times).isoformat()
                return result
            return {"status": "absent", "event": None, "headline": None, "severity": None}
    except Exception as e:
        return _unavailable(str(e) or "NWS alerts feed unreachable")
