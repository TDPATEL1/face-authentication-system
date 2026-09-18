import threading

from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies import get_current_user
from app.api.routes.door import door_service
from app.services.unified_auth_session import (
    unified_auth_session_manager,
)


client = TestClient(app)


def test_concurrent_door_unlock_same_grant_only_unlocks_once(
    monkeypatch,
):
    """
    A single door grant must be usable only once.

    Ten concurrent requests using the same grant must result
    in exactly one successful door unlock.
    """

    class FakeUser:
        id = 1
        name = "Test User"
        email = "test@example.com"

    def fake_current_user():
        return FakeUser()

    app.dependency_overrides[
        get_current_user
    ] = fake_current_user

    # IMPORTANT:
    # The door route uses the application's singleton manager.
    grant = unified_auth_session_manager.create_door_grant(
        user_id=FakeUser.id,
    )

    unlock_count = 0
    unlock_lock = threading.Lock()

    def fake_unlock():
        nonlocal unlock_count

        with unlock_lock:
            unlock_count += 1

        return True

    monkeypatch.setattr(
        door_service,
        "unlock",
        fake_unlock,
    )

    results = []
    result_lock = threading.Lock()

    def unlock_request():
        try:
            response = client.post(
                "/api/v1/door/unlock",
                headers={
                    "X-Door-Authorization": grant,
                },
            )

            with result_lock:
                results.append(response.status_code)

        except Exception as exc:
            with result_lock:
                results.append(exc)

    threads = [
        threading.Thread(target=unlock_request)
        for _ in range(10)
    ]

    try:
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    finally:
        app.dependency_overrides.pop(
            get_current_user,
            None,
        )

        # Defensive cleanup in case the test fails.
        unified_auth_session_manager.door_grants.pop(
            grant,
            None,
        )

    exceptions = [
        result
        for result in results
        if isinstance(result, Exception)
    ]

    successful_requests = [
        result
        for result in results
        if result == 200
    ]

    rejected_requests = [
        result
        for result in results
        if result == 401
    ]

    assert exceptions == []

    assert len(results) == 10

    assert len(successful_requests) == 1

    assert len(rejected_requests) == 9

    assert unlock_count == 1


def test_concurrent_door_unlock_wrong_user_never_succeeds(
    monkeypatch,
):
    """
    A door grant belonging to User 1 must never authorize
    User 2, even when requests happen concurrently.
    """

    class OwnerUser:
        id = 1
        name = "Owner"
        email = "owner@example.com"

    class AttackerUser:
        id = 2
        name = "Attacker"
        email = "attacker@example.com"

    grant = unified_auth_session_manager.create_door_grant(
        user_id=OwnerUser.id,
    )

    unlock_count = 0
    unlock_lock = threading.Lock()

    def fake_unlock():
        nonlocal unlock_count

        with unlock_lock:
            unlock_count += 1

        return True

    monkeypatch.setattr(
        door_service,
        "unlock",
        fake_unlock,
    )

    def fake_current_user():
        return AttackerUser()

    app.dependency_overrides[
        get_current_user
    ] = fake_current_user

    results = []
    result_lock = threading.Lock()

    def unlock_request():
        try:
            response = client.post(
                "/api/v1/door/unlock",
                headers={
                    "X-Door-Authorization": grant,
                },
            )

            with result_lock:
                results.append(response.status_code)

        except Exception as exc:
            with result_lock:
                results.append(exc)

    threads = [
        threading.Thread(target=unlock_request)
        for _ in range(10)
    ]

    try:
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    finally:
        app.dependency_overrides.pop(
            get_current_user,
            None,
        )

        unified_auth_session_manager.door_grants.pop(
            grant,
            None,
        )

    assert len(results) == 10

    assert all(
        result == 401
        for result in results
    )

    assert unlock_count == 0