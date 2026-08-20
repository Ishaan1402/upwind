import os
from dotenv import load_dotenv

load_dotenv()

def _clean_key(key: str) -> str:
    cleaned = key.strip().strip("'").strip('"')
    if not cleaned or "your_" in cleaned.lower() or "here" in cleaned.lower() or "placeholder" in cleaned.lower():
        return ""
    return cleaned

AIRNOW_KEY = _clean_key(os.getenv("AIRNOW_KEY", ""))
OPENAQ_API_KEY = _clean_key(os.getenv("OPENAQ_API_KEY", ""))
FIRMS_MAP_KEY = _clean_key(os.getenv("FIRMS_MAP_KEY", ""))
CENSUS_API_KEY = _clean_key(os.getenv("CENSUS_API_KEY", ""))
DEEPSEEK_API_KEY = _clean_key(os.getenv("DEEPSEEK_API_KEY", ""))
GROQ_API_KEY = _clean_key(os.getenv("GROQ_API_KEY", ""))
PORT = int(os.getenv("PORT", "8000"))

CORS_ORIGINS = [
    o.strip() for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000"
    ).split(",") if o.strip()
]

RATE_LIMIT_AQI_PER_MIN = int(os.getenv("RATE_LIMIT_AQI_PER_MIN", "60"))
RATE_LIMIT_WHY_PER_HOUR = int(os.getenv("RATE_LIMIT_WHY_PER_HOUR", "8"))
TRUST_PROXY = os.getenv("TRUST_PROXY", "0").strip() in ("1", "true", "True")

# Cost estimation for the metrics dashboard. Update these when provider
# pricing changes; the dashboard labels the result as an estimate.
LLM_INPUT_PRICE_PER_1M = float(os.getenv("LLM_INPUT_PRICE_PER_1M", "0.27"))
LLM_OUTPUT_PRICE_PER_1M = float(os.getenv("LLM_OUTPUT_PRICE_PER_1M", "1.10"))

# Observation tokens bind /api/aqi responses to /api/why requests so clients
# cannot fabricate observations. Set a strong secret and enable enforcement in
# production.
OBSERVATION_TOKEN_SECRET = os.getenv("OBSERVATION_TOKEN_SECRET", "")
OBSERVATION_TOKEN_MAX_AGE_SECONDS = int(os.getenv("OBSERVATION_TOKEN_MAX_AGE_SECONDS", "600"))
ENFORCE_OBSERVATION_TOKENS = os.getenv("ENFORCE_OBSERVATION_TOKENS", "0").strip() in ("1", "true", "True")

# AQI standard category colors & labels according to EPA
AQI_CATEGORIES = [
    {"max": 50, "label": "Good", "color": "#00e400", "textColor": "#000000", "description": "Air quality is satisfactory, and air pollution poses little or no risk."},
    {"max": 100, "label": "Moderate", "color": "#ffff00", "textColor": "#000000", "description": "Air quality is acceptable; however, there may be a risk for some sensitive individuals."},
    {"max": 150, "label": "Unhealthy for Sensitive Groups", "color": "#ff7e00", "textColor": "#ffffff", "description": "Members of sensitive groups may experience health effects."},
    {"max": 200, "label": "Unhealthy", "color": "#ff0000", "textColor": "#ffffff", "description": "Some members of the general public may experience health effects."},
    {"max": 300, "label": "Very Unhealthy", "color": "#8f3f97", "textColor": "#ffffff", "description": "Health alert: The risk of health effects is increased for everyone."},
    {"max": 9999, "label": "Hazardous", "color": "#7e0023", "textColor": "#ffffff", "description": "Health warning of emergency conditions: everyone is more likely to be affected."}
]

def get_aqi_category(aqi_val: int):
    for cat in AQI_CATEGORIES:
        if aqi_val <= cat["max"]:
            return cat
    return AQI_CATEGORIES[-1]
