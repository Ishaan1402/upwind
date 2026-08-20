"""
Nearby-airport METAR dust present-weather confirmation.

ASOS/AWOS stations report present weather in their METAR raw messages. When the
latest nearby METARs carry a blowing-dust phenomenon code (BLDU, DU, DS, SS,
VCDS, PO, TSDS), dust is confirmed at a nearby airport. This is a
high-precision, low-recall confirmation: presence confirms dust, absence
proves nothing.

Implementation note: we use the aviationweather.gov METAR feed with a small
bbox around the target rather than the NWS grid -> stations -> observations
hop. The grid/points hop is flaky under load and requires up to ~6+ sequential
lookups per request; the bbox query is a single call that returns the latest
observation for every station in the neighborhood.
"""

import re
from typing import Any, Dict, List, Optional

import httpx

METAR_USER_AGENT = "upwind-app/1.0 (contact@getupwind.me)"
METAR_URL = "https://aviationweather.gov/api/data/metar"
METAR_TIMEOUT_S = 4.0
# Half-width of the search box in degrees (~35 mi N-S; the E-W extent shrinks
# with latitude but stays a valid "nearby" neighborhood).
METAR_BBOX_DEG = 0.5

# Standard METAR present-weather codes that confirm dust. `DU` (widespread
# dust in suspension) is a substring of `BLDU`, but the standalone-token match
# below means each code only matches itself.
DUST_PHENOMENA = ("BLDU", "DU", "DS", "SS", "VCDS", "PO", "TSDS")


def _unavailable(reason: str) -> Dict[str, Any]:
    return {
        "status": "unavailable",
        "station": None,
        "raw": None,
        "phenomenon": None,
        "details": reason,
    }


def _absent() -> Dict[str, Any]:
    return {"status": "absent", "station": None, "raw": None, "phenomenon": None}


def match_dust_phenomenon(raw_metar: str, wx_string: Optional[str]) -> Optional[str]:
    """First dust phenomenon code present in the METAR/weather string, else None.

    Codes are matched as standalone tokens (word boundaries) so a substring
    like "DU" inside "DULUTH" or "BLDU" cannot false-positive.
    """
    haystacks = [text for text in (raw_metar, wx_string) if text]
    if not haystacks:
        return None
    for code in DUST_PHENOMENA:
        pattern = re.compile(rf"\b{code}\b")
        if any(pattern.search(h.upper()) for h in haystacks):
            return code
    return None


async def fetch_metar_dust(lat: float, lon: float) -> Dict[str, Any]:
    """
    Check nearby airport METARs for blowing-dust present weather.

    Uses the aviationweather.gov METAR feed with a small bbox around the target
    (single request, no per-station fan-out). Never raises: failures return
    "unavailable"; a healthy response with no dust code returns "absent".
    """
    bbox = (
        f"{lat - METAR_BBOX_DEG},{lon - METAR_BBOX_DEG},"
        f"{lat + METAR_BBOX_DEG},{lon + METAR_BBOX_DEG}"
    )
    params = {"bbox": bbox, "format": "json"}
    headers = {"User-Agent": METAR_USER_AGENT, "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=METAR_TIMEOUT_S, headers=headers) as client:
            resp = await client.get(METAR_URL, params=params)
            if resp.status_code != 200:
                return _unavailable(f"METAR feed HTTP {resp.status_code}")
            data = resp.json()
            if not isinstance(data, list):
                return _unavailable("Unexpected METAR feed payload")
            for obs in data:
                raw = obs.get("rawOb") or ""
                code = match_dust_phenomenon(raw, obs.get("wxString"))
                if code is not None:
                    return {
                        "status": "present",
                        "station": obs.get("icaoId") or obs.get("stationId"),
                        "raw": raw,
                        "phenomenon": code,
                    }
            return _absent()
    except Exception as e:
        return _unavailable(str(e) or "METAR feed unreachable")
