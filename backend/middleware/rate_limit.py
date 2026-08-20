import time
import logging
from collections import defaultdict
from typing import Dict, List
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from backend.config import RATE_LIMIT_AQI_PER_MIN, RATE_LIMIT_WHY_PER_HOUR, TRUST_PROXY
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
        self.history[ip][category] = valid_timestamps

        if len(valid_timestamps) >= limit:
            oldest = valid_timestamps[0]
            retry_after = max(1, int(window_seconds - (now - oldest)))
            return True, retry_after

        valid_timestamps.append(now)
        return False, 0

    async def dispatch(self, request: Request, call_next):
        ip = self._get_client_ip(request)
        path = request.url.path
        method = request.method

        # Check rate limits for specific endpoints
        if path == "/api/aqi" and method == "GET":
            is_limited, retry_after = self._is_rate_limited(ip, "aqi", RATE_LIMIT_AQI_PER_MIN, 60)
            if is_limited:
                logger.warning(f"Rate limit exceeded for AQI from {ip}")
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."},
                    headers={"Retry-After": str(retry_after)}
                )
        elif (path in ("/api/why", "/api/why/stream")) and method in ("GET", "POST"):
            is_limited, retry_after = self._is_rate_limited(ip, "why", RATE_LIMIT_WHY_PER_HOUR, 3600)
            if is_limited:
                logger.warning(f"Rate limit exceeded for Show Why from {ip}")
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."},
                    headers={"Retry-After": str(retry_after)}
                )

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
