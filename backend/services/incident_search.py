import re
from urllib.parse import quote
from typing import Optional
import httpx

IGNORED_FIRE_WORDS = {
    "Google News", "The", "State", "County", "Forest", "Wildfire", "Department",
    "Local", "City", "Area", "Oregon", "Central Oregon", "Pacific Northwest",
    "California", "Washington", "Idaho", "Nevada", "Montana", "Wyoming", "Utah",
    "Arizona", "New Mexico", "Texas", "Colorado", "Northwest", "Southwest", "East",
    "West", "North", "South", "Central", "Mountain", "Western", "Eastern", "Northern", "Southern"
}

async def search_fire_incident_name(state: Optional[str], city: Optional[str], lat: float, lon: float) -> Optional[str]:
    """
    Search public web news RSS feeds for official wildfire incident names.
    Executed whenever PM is elevated, HMS smoke is present, or FIRMS hotspots are detected.
    Returns confirmed fire name (e.g., 'Brewer Fire', 'Falls Fire') or None.
    """
    location_str = f"{city} {state}".strip() if (city or state) else f"{state}".strip() if state else f"{lat:.2f}, {lon:.2f}"
    
    queries = [
        f"{location_str} wildfire fire",
        f"{state} active wildfire fire" if state else f"{location_str} fire"
    ]

    async with httpx.AsyncClient(timeout=4.5) as client:
        for q in queries:
            url = f"https://news.google.com/rss/search?q={quote(q)}&hl=en-US&gl=US&ceid=US:en"
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    titles = re.findall(r'<title>(.*?)</title>', resp.text)
                    for title in titles:
                        matches = re.findall(r'\b([A-Z][a-zA-Z0-9\'-]+(?:\s+[A-Z][a-zA-Z0-9\'-]+)?)\s+(?:Fire|Wildfire|Fires|Complex)\b', title)
                        for m in matches:
                            candidate = m.strip()
                            if candidate not in IGNORED_FIRE_WORDS and len(candidate) > 2:
                                return f"{candidate} Fire"
            except Exception as e:
                print(f"[Incident Web Search Warning]: {e}")
                continue

    return None
