import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes.door import door_service
from app.services.unified_auth_session import (
    UnifiedAuthSessionManager,
)


client = TestClient(app)


# ============================================================
# HELPERS
# ============================================================

def create_test_manager():
    return UnifiedAuthSessionManager()


# ============================================================
# JWT-ONLY DOOR ACCESS
# ============================================================

def test_door_unlock_requires_door_grant(monkeypatch):
    """
    A valid JWT alone must NOT be sufficient to unlock the door.
    """

    class FakeUser:
        id = 1
        name = "Test User"
        email = "test@example.com"

    def fake_current_user():
        return FakeUser()

    monkeypatch.setattr(
        "app.api.routes.door.get_current_user",
        fake_current_user,
    )

    unlocked = False

    def fake_unlock():
        nonlocal unlocked
        unlocked = True
        return True

    monkeypatch.setattr(
        door_service,
        "unlock",
        fake_unlock,
    )

    # The dependency is overridden so the request represents
    # an authenticated user with a valid JWT.
    from app.api.dependencies import get_current_user

    app.dependency_overrides[
        get_current_user
    ] = fake_current_user

    try:
        response = client.post(
            "/api/v1/door/unlock",
        )

        assert response.status_code == 401
        assert unlocked is False

        detail = response.json()["detail"].lower()

        assert (
            "authorization" in detail
            or "door" in detail
        )

    finally:
        app.dependency_overrides.pop(
            get_current_user,
            None,
        )

# ============================================================
# GRANT CREATION
# ============================================================

def test_door_grant_is_created():
    manager = create_test_manager()

    grant = manager.create_door_grant(
        user_id=1,
    )

    assert grant
    assert isinstance(grant, str)

    assert grant in manager.door_grants


# ============================================================
# VALID GRANT
# ============================================================

def test_valid_door_grant_can_be_consumed():
    manager = create_test_manager()

    grant = manager.create_door_grant(
        user_id=1,
    )

    valid, reason = manager.consume_door_grant(
        grant=grant,
        user_id=1,
    )

    assert valid is True
    assert reason == "Door authorization accepted"

    assert grant not in manager.door_grants


# ============================================================
# ONE-TIME USE
# ============================================================

def test_door_grant_cannot_be_reused():
    manager = create_test_manager()

    grant = manager.create_door_grant(
        user_id=1,
    )

    first_valid, _ = manager.consume_door_grant(
        grant=grant,
        user_id=1,
    )

    second_valid, second_reason = (
        manager.consume_door_grant(
            grant=grant,
            user_id=1,
        )
    )

    assert first_valid is True
    assert second_valid is False

    assert (
        "invalid" in second_reason.lower()
        or "expired" in second_reason.lower()
    )


# ============================================================
# USER BINDING
# ============================================================

def test_door_grant_cannot_be_used_by_another_user():
    manager = create_test_manager()

    grant = manager.create_door_grant(
        user_id=1,
    )

    valid, reason = manager.consume_door_grant(
        grant=grant,
        user_id=2,
    )

    assert valid is False
    assert "invalid" in reason.lower()

    # The grant remains available for the correct user.
    valid_correct_user, _ = (
        manager.consume_door_grant(
            grant=grant,
            user_id=1,
        )
    )

    assert valid_correct_user is True


# ============================================================
# INVALID GRANT
# ============================================================

def test_invalid_door_grant_is_rejected():
    manager = create_test_manager()

    valid, reason = manager.consume_door_grant(
        grant="invalid-grant",
        user_id=1,
    )

    assert valid is False
    assert (
        "invalid" in reason.lower()
        or "expired" in reason.lower()
    )


# ============================================================
# MISSING GRANT
# ============================================================

def test_empty_door_grant_is_rejected():
    manager = create_test_manager()

    valid, reason = manager.consume_door_grant(
        grant="",
        user_id=1,
    )

    assert valid is False
    assert "required" in reason.lower()


# ============================================================
# INVALID USER
# ============================================================

@pytest.mark.parametrize(
    "user_id",
    [
        0,
        -1,
        None,
    ],
)
def test_invalid_user_cannot_consume_door_grant(
    user_id,
):
    manager = create_test_manager()

    grant = manager.create_door_grant(
        user_id=1,
    )

    valid, _ = manager.consume_door_grant(
        grant=grant,
        user_id=user_id,
    )

    assert valid is False


# ============================================================
# EXPIRATION
# ============================================================

def test_expired_door_grant_is_rejected():
    manager = create_test_manager()

    manager.DOOR_GRANT_TTL_SECONDS = 0.01

    grant = manager.create_door_grant(
        user_id=1,
    )

    time.sleep(0.05)

    valid, reason = manager.consume_door_grant(
        grant=grant,
        user_id=1,
    )

    assert valid is False
    assert "expired" in reason.lower()


# ============================================================
# CONCURRENT REUSE
# ============================================================

def test_concurrent_door_grant_use_only_succeeds_once():
    manager = create_test_manager()

    grant = manager.create_door_grant(
        user_id=1,
    )

    results = []

    def consume():
        valid, _ = manager.consume_door_grant(
            grant=grant,
            user_id=1,
        )
        results.append(valid)

    threads = [
        threading.Thread(
            target=consume
        )
        for _ in range(10)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert results.count(True) == 1
    assert results.count(False) == 9


# ============================================================
# GRANT CLEANUP
# ============================================================

def test_expired_door_grants_are_cleaned_up():
    manager = create_test_manager()

    manager.DOOR_GRANT_TTL_SECONDS = 0.01

    grant = manager.create_door_grant(
        user_id=1,
    )

    assert grant in manager.door_grants

    time.sleep(0.05)

    manager.cleanup_expired_door_grants()

    assert grant not in manager.door_grants