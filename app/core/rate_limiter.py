import threading
import time

from fastapi import Request
from fastapi.responses import JSONResponse


class RateLimiter:
    """
    Simple thread-safe in-memory fixed-window rate limiter.

    Limits are tracked independently for each client IP
    and endpoint.

    NOTE:
    This implementation is appropriate for a single-process
    deployment/development environment.

    For a multi-worker or multi-instance production deployment,
    this should eventually be replaced with Redis-backed
    distributed rate limiting.
    """

    DEFAULT_WINDOW_SECONDS = 60

    LIMITS: dict[str, int] = {
        "/api/v1/auth/login": 5,
        "/api/v1/auth/register": 5,

        "/api/v1/face/login": 5,
        "/api/v1/face/session/start": 10,
        "/api/v1/face/liveness/start": 10,
        "/api/v1/face/liveness/frame": 30,
        "/api/v1/face/anti-spoof/frame": 30,
        "/api/v1/face/enroll": 5,
    }

    def __init__(
        self,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ):
        self.window_seconds = window_seconds

        self._lock = threading.RLock()

        # key:
        #     (client_ip, endpoint)
        #
        # value:
        #     (window_start_timestamp, request_count)

        self._requests: dict[
            tuple[str, str],
            tuple[float, int],
        ] = {}

    def _get_client_ip(
        self,
        request: Request,
    ) -> str:
        """
        Get the direct client IP.

        We intentionally do not trust X-Forwarded-For here.
        Trusting proxy headers requires explicit trusted-proxy
        configuration.
        """

        if request.client is None:
            return "unknown"

        return request.client.host

    def is_rate_limited(
        self,
        client_ip: str,
        endpoint: str,
    ) -> tuple[bool, int, int]:
        """
        Check and record a request.

        Returns:

            (
                limited,
                limit,
                retry_after_seconds,
            )
        """

        limit = self.LIMITS.get(endpoint)

        # Endpoint is not rate limited.
        if limit is None:
            return False, 0, 0

        now = time.monotonic()

        key = (
            client_ip,
            endpoint,
        )

        with self._lock:

            entry = self._requests.get(key)

            # First request in this window.
            if entry is None:
                self._requests[key] = (
                    now,
                    1,
                )

                return False, limit, 0

            window_start, request_count = entry

            elapsed = now - window_start

            # Existing window expired.
            if elapsed >= self.window_seconds:

                self._requests[key] = (
                    now,
                    1,
                )

                return False, limit, 0

            # Limit already reached.
            if request_count >= limit:

                retry_after = max(
                    1,
                    int(
                        self.window_seconds
                        - elapsed
                    ),
                )

                return (
                    True,
                    limit,
                    retry_after,
                )

            # Count this request.
            self._requests[key] = (
                window_start,
                request_count + 1,
            )

            return False, limit, 0

    def reset(self) -> None:
        """
        Clear all tracked requests.

        Primarily useful for tests and controlled
        application resets.
        """

        with self._lock:
            self._requests.clear()

    def cleanup(self) -> None:
        """
        Remove expired entries to prevent unlimited
        in-memory dictionary growth.
        """

        now = time.monotonic()

        with self._lock:

            expired_keys = []

            for key, (window_start, _) in self._requests.items():

                if (
                    now - window_start
                    >= self.window_seconds
                ):
                    expired_keys.append(key)

            for key in expired_keys:
                self._requests.pop(
                    key,
                    None,
                )

    async def middleware(
        self,
        request: Request,
        call_next,
    ):
        """
        FastAPI/Starlette middleware entry point.
        """

        endpoint = request.url.path

        # Only rate-limit configured endpoints.
        if endpoint not in self.LIMITS:
            return await call_next(request)

        # Only rate-limit HTTP requests.
        if request.method != "POST":
            return await call_next(request)

        client_ip = self._get_client_ip(request)

        limited, limit, retry_after = (
            self.is_rate_limited(
                client_ip=client_ip,
                endpoint=endpoint,
            )
        )

        if limited:

            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        "Too many requests. "
                        "Please try again later."
                    ),
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)

        remaining = max(
            0,
            limit
            - self._requests[
                (client_ip, endpoint)
            ][1],
        )

        response.headers[
            "X-RateLimit-Limit"
        ] = str(limit)

        response.headers[
            "X-RateLimit-Remaining"
        ] = str(remaining)

        return response


rate_limiter = RateLimiter()