from typing import Dict, Any, Optional
import httpx

from backend.config import CENSUS_API_KEY
from backend.db import get_cached_place, set_cached_place

# 2020 Decennial Census DHC P1 total population by ZCTA.
# NOTE: the PL 94-171 endpoint (dec/pl) does NOT support "zip code tabulation
# area" geography (400 "unknown/unsupported geography hierarchy"); ZCTA is only
# available via dec/dhc (or acs5, which uses a different variable name).
CENSUS_URL = "https://api.census.gov/data/2020/dec/dhc"

RURAL_POPULATION_CAP = 5000
PLACE_CACHE_TTL_DAYS = 365


def _parse_population(payload: Any) -> Optional[int]:
    """
    The decennial API returns JSON array-of-arrays:
      [["NAME","P1_001N","zip code tabulation area"], ["ZCTA5 97028","230","97028"]]
    Return the P1_001N population of the first data row, or None if malformed.
    """
    if not isinstance(payload, list) or len(payload) < 2:
        return None
    for row in payload[1:]:
        if isinstance(row, list) and len(row) >= 2:
            try:
                return int(float(row[1]))
            except (TypeError, ValueError):
                continue
    return None


async def fetch_place_context(zip_code: Optional[str]) -> Dict[str, Any]:
    """
    Fetch ZCTA population for a ZIP code (cached in SQLite under pop:{zip},
    365-day TTL). Any failure such as a missing key, network error, or no ZIP returns
    status "unavailable" without raising, so scoring is unchanged.
    """
    if not zip_code:
        return {
            "status": "unavailable",
            "population": None,
            "rural": None,
            "details": "No ZIP code available for population context",
        }

    # Strip ZIP+4 suffix to 5-digit ZCTA
    zip_code = zip_code.strip()[:5]
    if not zip_code:
        return {
            "status": "unavailable",
            "population": None,
            "rural": None,
            "details": "No ZIP code available for population context",
        }

    cache_key = f"pop:{zip_code}"
    cached = get_cached_place(cache_key, max_age_days=PLACE_CACHE_TTL_DAYS)
    if cached:
        return cached

    if not CENSUS_API_KEY:
        return {
            "status": "unavailable",
            "population": None,
            "rural": None,
            "details": "Census API key not configured",
        }

    params = {
        "get": "NAME,P1_001N",
        "for": f"zip code tabulation area:{zip_code}",
        "key": CENSUS_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            resp = await client.get(CENSUS_URL, params=params)
            if resp.status_code == 200:
                population = _parse_population(resp.json())
                if population is not None:
                    result = {
                        "status": "present",
                        "population": population,
                        "rural": population < RURAL_POPULATION_CAP,
                        "details": f"ZCTA population {population}",
                    }
                    set_cached_place(cache_key, result)
                    return result
    except Exception:
        pass

    return {
        "status": "unavailable",
        "population": None,
        "rural": None,
        "details": "Census population data unavailable",
    }
