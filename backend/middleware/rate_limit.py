import time
import logging
from collections import defaultdict
from typing import Dict, List
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from backend.config import RATE_LIMIT_AQI_PER_MIN, RATE_LIMIT_WHY_PER_HOUR, RATE_LIMIT_EVENTS_PER_MIN, TRUST_PROXY
from backend.metrics import record_request

logger = logging.getLogger("upwind.http")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class RateLimitAndLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        # Store timestamps of requests: dict[str, dict[str, List[float]]]
        # e.g., self.history[ip]['aqi'] = [t1, t2, ...]
        self.history: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._last_sweep = time.time()

    def _sweep(self, now: float) -> None:
        """Evict IPs with no activity within a 1h horizon.

        Called periodically so the per-IP history does not grow monotonically
        with every unique visitor/crawler that ever hit the API.
        """
        horizon = 3600.0
        for ip in list(self.history.keys()):
            categories = self.history[ip]
            for category in list(categories.keys()):
                if not any(t > now - horizon for t in categories[category]):
                    categories.pop(category, None)
            if not categories:
                self.history.pop(ip, None)
        self._last_sweep = now

    def _get_client_ip(self, request: Request) -> str:
        if TRUST_PROXY:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "127.0.0.1"

    def _is_rate_limited(self, ip: str, category: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        now = time.time()
        timestamps = self.history[ip][category]

        # Filter out expired timestamps
        cutoff = now - window_seconds
        valid_timestamps = [t for t in timestamps if t > cutoff]

        if len(valid_timestamps) >= limit:
            self.history[ip][category] = valid_timestamps
            oldest = valid_timestamps[0]
            retry_after = max(1, int(window_seconds - (now - oldest)))
            return True, retry_after

        valid_timestamps.append(now)
        # Always store back a non-empty list so the entry stays bounded and the
        # limiter accumulates across requests (a stale ip key is dropped by the
        # expiry filter on its next request).
        self.history[ip][category] = valid_timestamps
        return False, 0

    def _limited(self, ip: str, label: str, method: str, path: str, retry_after: int) -> JSONResponse:
        logger.warning(f"Rate limit exceeded for {label} from {ip}")
        # Rate-limited requests must still be recorded, or the dashboard's
        # 429_count / 429_rate are permanently 0 in production.
        record_request(method, path, 429, 0.0)
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
            headers={"Retry-After": str(retry_after)}
        )

    async def dispatch(self, request: Request, call_next):
        # Periodically evict stale IP entries (every ~10 minutes).
        if time.time() - self._last_sweep > 600:
            self._sweep(time.time())

        ip = self._get_client_ip(request)
        path = request.url.path
        method = request.method

        # Check rate limits for specific endpoints
        if path == "/api/aqi" and method == "GET":
            is_limited, retry_after = self._is_rate_limited(ip, "aqi", RATE_LIMIT_AQI_PER_MIN, 60)
            if is_limited:
                return self._limited(ip, "AQI", method, path, retry_after)
        elif (path in ("/api/why", "/api/why/stream")) and method in ("GET", "POST"):
            is_limited, retry_after = self._is_rate_limited(ip, "why", RATE_LIMIT_WHY_PER_HOUR, 3600)
            if is_limited:
                return self._limited(ip, "Show Why", method, path, retry_after)
        elif path == "/api/events" and method == "POST":
            is_limited, retry_after = self._is_rate_limited(ip, "events", RATE_LIMIT_EVENTS_PER_MIN, 60)
            if is_limited:
                return self._limited(ip, "Events", method, path, retry_after)

        t0 = time.perf_counter()
        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(f"{method} {path} {response.status_code} {duration_ms}ms {ip}")
            record_request(method, path, response.status_code, duration_ms)
            return response
        except Exception as e:
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.error(f"{method} {path} 500 {duration_ms}ms {ip} - Error: {e}")
            record_request(method, path, 500, duration_ms)
            raise e
