from fastapi.testclient import TestClient

from app.core.rate_limiter import rate_limiter
from app.main import app


client = TestClient(app)


def test_configured_login_rate_limit_is_enforced():
    endpoint = "/api/v1/auth/login"
    original_limit = rate_limiter.LIMITS[endpoint]

    rate_limiter.reset()
    rate_limiter.LIMITS[endpoint] = 2

    try:
        first = client.post(endpoint, json={})
        second = client.post(endpoint, json={})
        limited = client.post(endpoint, json={})

        assert first.status_code == 422
        assert second.status_code == 422
        assert limited.status_code == 429
        assert limited.headers["Retry-After"]
        assert limited.headers["X-RateLimit-Limit"] == "2"
        assert limited.headers["X-RateLimit-Remaining"] == "0"

    finally:
        rate_limiter.LIMITS[endpoint] = original_limit
        rate_limiter.reset()
