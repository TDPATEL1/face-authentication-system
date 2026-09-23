import pytest

from app.core.rate_limiter import rate_limiter


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    rate_limiter.reset()
    yield
    rate_limiter.reset()