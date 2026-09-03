import re
import email.utils
from datetime import datetime, timezone, timedelta
from urllib.parse import quote
from typing import Optional

import httpx

# Explicit wildland / incident-reporting language 
WILDLAND_CONTEXT_RE = re.compile(
    r"\b("
    r"wildfire|wildfires|wildland|wild[- ]?land|"
    r"forest\s+fire|brush\s+fire|grass\s+fire|"
    r"acres?\b|hectare|containment|evacuat|"
    r"burn\s+scar|smoke\s+plume|fire\s+complex|inciweb|nifc"
    r")\b",
    re.IGNORECASE,
)

# Urban / person / structure / vehicle fires — domain class, not place-specific names.
# Blocks "Man Starts Fire…", "House Fire…", "Car Fire…" everywhere in the US.
URBAN_OR_PERSON_FIRE_RE = re.compile(
    r"(?i)"
    r"(\b(man|woman|men|person|people|teen|teenager|resident|homeowner|neighbor)\b.{0,48}\bfires?\b)"
    r"|"
    r"(\b(house|home|apartment|condo|garage|warehouse|barn|vehicle|car|truck|bus|dumpster)\s+fires?\b)"
)

# Named-incident extractor: "Creek Fire", "Hay Creek Fire", "Dixie Complex"
INCIDENT_NAME_RE = re.compile(
    r"\b([A-Z][a-zA-Z0-9'-]+(?:\s+[A-Z][a-zA-Z0-9'-]+)?)\s+(?:Fire|Wildfire|Complex)\b"
)

# Fire *categories* (not named incidents). Closed domain ontology.
FIRE_TYPE_GENERICS = {
    "Brush", "Grass", "Forest", "Structure", "House", "Home", "Building",
    "Vehicle", "Car", "Truck", "Dumpster", "Trash", "Rubbish", "Campfire",
    "Bonfire", "Wildfire", "Wildland",
}

ADMIN_NOISE = {
    "Google News", "The", "State", "County", "Department", "Local", "City", "Area",
    "News", "Breaking", "Live", "Daily", "Press", "Herald", "Times", "Tribune",
    "Update", "Updates", "Active", "Ongoing", "Latest",
}


def parse_pubdate(pubdate_str: Optional[str]) -> Optional[datetime]:
    """Parse RSS RFC 822 pubDate string to timezone-aware UTC datetime."""
    if not pubdate_str:
        return None
    try:
        dt = email.utils.parsedate_to_datetime(pubdate_str)
        if dt is not None:
            return dt.astimezone(timezone.utc)
    except Exception:
        pass
    return None


def is_stale_article(pubdate_str: Optional[str], max_days: int = 7) -> bool:
    """Return True if article pubDate is missing, unparseable, or older than max_days."""
    dt = parse_pubdate(pubdate_str)
    if dt is None:
        return True
    now = datetime.now(timezone.utc)
    return (now - dt) > timedelta(days=max_days)


def looks_like_verb_token(token: str) -> bool:
    """
    Morphology check for headline verb fragments (Starts, Caused, Igniting).
    Avoids a person/verb name blocklist. Rare place-name collisions are preferable
    to accepting Subject+Verb fragments as incident names.
    """
    t = token.strip()
    if len(t) < 4:
        return False
    if re.fullmatch(r"[A-Z][a-z]+(?:ed|ing)", t):
        return True
    # Present-tense headline verbs: Starts, Sparks, Spreads — not Oaks/Hills/Falls
    if re.fullmatch(r"[A-Z][a-z]+s", t) and t.endswith(("ts", "rs", "ks", "ds", "ns", "ps")):
        if t.lower() in {"oaks", "hills", "falls", "woods", "springs", "acres"}:
            return False
        return True
    return False


def is_valid_incident_candidate(candidate: str) -> bool:
    """
    Structural validation for US named wildfire incidents (place-like names).
    Rejects fire-type generics and Subject+Verb headline fragments.
    """
    candidate = (candidate or "").strip()
    if len(candidate) < 3:
        return False
    if candidate in ADMIN_NOISE or candidate in FIRE_TYPE_GENERICS:
        return False
    if not re.fullmatch(r"[A-Z][a-zA-Z0-9'-]+(?:\s+[A-Z][a-zA-Z0-9'-]+)?", candidate):
        return False

    tokens = candidate.split()
    if len(tokens) >= 2 and any(looks_like_verb_token(t) for t in tokens):
        return False
    return True


def title_allows_incident_extraction(title: str) -> bool:
    """
    General US gate:
    1) Never extract from urban/person/structure/vehicle fire headlines.
    2) Prefer explicit wildland language.
    3) Otherwise allow titles that already look like named-incident reporting
       (e.g. "Creek Fire grows overnight") without requiring 'acres'/'evac' every time.
    """
    if not title:
        return False
    if URBAN_OR_PERSON_FIRE_RE.search(title):
        return False
    if WILDLAND_CONTEXT_RE.search(title):
        return True
    return any(is_valid_incident_candidate(m) for m in INCIDENT_NAME_RE.findall(title))


def parse_rss_items_for_incident(rss_text: str, max_days: int = 7) -> Optional[str]:
    """
    Parse Google News RSS for recent wildland named incidents (within max_days).
    Uses structural title gates — not location-specific name blocklists.
    """
    items = re.findall(r"<item>(.*?)</item>", rss_text, re.DOTALL)
    for item in items:
        title_match = re.search(r"<title>(.*?)</title>", item)
        pubdate_match = re.search(r"<pubDate>(.*?)</pubDate>", item)
        if not title_match:
            continue

        title = title_match.group(1)
        pubdate_str = pubdate_match.group(1) if pubdate_match else None

        if is_stale_article(pubdate_str, max_days=max_days):
            continue
        if not title_allows_incident_extraction(title):
            continue

        for m in INCIDENT_NAME_RE.findall(title):
            base_name = re.sub(r"\s+(Fire|Wildfire|Complex)$", "", m).strip()
            if base_name and is_valid_incident_candidate(base_name):
                return f"{base_name} Fire"

    return None


async def search_fire_incident_name(
    state: Optional[str],
    city: Optional[str],
    lat: float,
    lon: float,
    country_code: Optional[str] = None,
) -> Optional[str]:
    """
    Search Google News RSS for recent wildland fire incident mentions (7-day window).
    Queries bias toward wildfire/wildland language so urban 'fire' crime headlines are rare.
    Naming is decorative — scoring only treats news as a fire vote when FIRMS corroborates.
    """
    location_parts = [city, state]
    if country_code and country_code.lower() != "us":
        location_parts.append(country_code.upper())
    location_str = " ".join(part for part in location_parts if part).strip() or f"{lat:.2f}, {lon:.2f}"

    wildland_terms = '(wildfire OR wildland OR "acres burned" OR evacuation)'
    queries = []
    if city and state:
        queries.append(f"{city} {state} {wildland_terms} when:7d")
    elif location_str:
        queries.append(f"{location_str} {wildland_terms} when:7d")
    if state:
        queries.append(f"{state} {wildland_terms} when:7d")
    # A city-less state query duplicates the location_str fallback.
    queries = list(dict.fromkeys(queries))

    async with httpx.AsyncClient(timeout=4.5) as client:
        for q in queries:
            url = f"https://news.google.com/rss/search?q={quote(q)}&hl=en-US&gl=US&ceid=US:en"
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    incident = parse_rss_items_for_incident(resp.text, max_days=7)
                    if incident:
                        return incident
            except Exception as e:
                print(f"[Incident Web Search Warning]: {e}")
                continue

    return None
