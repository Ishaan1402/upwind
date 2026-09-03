"""Short-lived signed observation tokens for the Why APIs.

The browser receives an ``observation_token`` from ``/api/aqi`` and echoes it
back to ``/api/why`` or ``/api/why/stream``. This stops clients from fabricating
AQI observations (and therefore LLM narratives/cost) while keeping the flow
fast: no second server-side observation fetch is needed.

The token is only secure when ``OBSERVATION_TOKEN_SECRET`` is set in
production. ``ENFORCE_OBSERVATION_TOKENS=1`` makes missing/invalid tokens a
400 instead of a soft skip.
"""

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional


def _payload(location: Dict[str, Any], observation: Dict[str, Any], expires_at: int) -> Dict[str, Any]:
    return {
        "v": 1,
        "lat": round(float(location.get("lat", 0)), 4),
        "lon": round(float(location.get("lon", 0)), 4),
        "aqi": int(observation.get("aqi", 0)),
        "primary_pollutant": str(observation.get("primary_pollutant", "") or ""),
        "category": str(observation.get("category", "") or ""),
        "exp": expires_at,
    }


def sign_observation(
    location: Dict[str, Any],
    observation: Dict[str, Any],
    secret: str,
    max_age_seconds: int = 600,
) -> Optional[str]:
    """Return a signed token, or None when no secret is configured."""
    if not secret:
        return None
    body = _payload(location, observation, int(time.time()) + max_age_seconds)
    message = json.dumps(body, sort_keys=True, separators=(",", ":"))
    signature = hmac.new(
        secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    raw = f"{message}.{signature}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def verify_observation_token(
    token: Optional[str],
    location: Dict[str, Any],
    observation: Dict[str, Any],
    secret: str,
) -> bool:
    """Verify token signature, expiry, and that it matches the request body."""
    if not token or not secret:
        return False
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
        message, signature = raw.rsplit(".", 1)
        expected = hmac.new(
            secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return False
        body = json.loads(message)
        if int(body.get("exp", 0)) < int(time.time()):
            return False
        expected_payload = _payload(location, observation, int(body.get("exp", 0)))
        return body == expected_payload
    except Exception:
        return False
