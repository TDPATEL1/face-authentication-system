import threading
import time

import pytest

from app.services.unified_auth_session import (
    UnifiedAuthSessionManager,
)


# ============================================================
# Door Grant Concurrency Tests
# ============================================================


def test_door_grant_only_one_concurrent_request_can_consume():
    """
    A door authorization grant is one-time-use.

    When multiple concurrent requests attempt to consume
    the same grant, exactly one request must succeed.
    """

    manager = UnifiedAuthSessionManager()

    user_id = 1
    grant = manager.create_door_grant(user_id)

    results = []
    lock = threading.Lock()

    def consume_grant():
        result = manager.consume_door_grant(
            grant=grant,
            user_id=user_id,
        )

        with lock:
            results.append(result)

    threads = [
        threading.Thread(target=consume_grant)
        for _ in range(10)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    successful = [
        result
        for result in results
        if result[0] is True
    ]

    failed = [
        result
        for result in results
        if result[0] is False
    ]

    assert len(results) == 10
    assert len(successful) == 1
    assert len(failed) == 9


def test_door_grant_cannot_cross_user_boundary_concurrently():
    """
    A grant issued to one user must never be accepted
    for another user, even under concurrent access.
    """

    manager = UnifiedAuthSessionManager()

    owner_user_id = 1
    attacker_user_id = 2

    grant = manager.create_door_grant(owner_user_id)

    results = []
    lock = threading.Lock()

    def consume_as_owner():
        result = manager.consume_door_grant(
            grant=grant,
            user_id=owner_user_id,
        )

        with lock:
            results.append(("owner", result))

    def consume_as_attacker():
        result = manager.consume_door_grant(
            grant=grant,
            user_id=attacker_user_id,
        )

        with lock:
            results.append(("attacker", result))

    threads = []

    for _ in range(5):
        threads.append(
            threading.Thread(target=consume_as_owner)
        )

    for _ in range(5):
        threads.append(
            threading.Thread(target=consume_as_attacker)
        )

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    owner_successes = [
        result
        for user, result in results
        if user == "owner" and result[0] is True
    ]

    attacker_successes = [
        result
        for user, result in results
        if user == "attacker" and result[0] is True
    ]

    assert len(owner_successes) == 1
    assert len(attacker_successes) == 0


def test_door_grant_replay_after_concurrent_consumption():
    """
    After a grant has been consumed, replay attempts must
    remain rejected.
    """

    manager = UnifiedAuthSessionManager()

    user_id = 1
    grant = manager.create_door_grant(user_id)

    first_result = manager.consume_door_grant(
        grant=grant,
        user_id=user_id,
    )

    assert first_result[0] is True

    replay_results = []
    lock = threading.Lock()

    def replay():
        result = manager.consume_door_grant(
            grant=grant,
            user_id=user_id,
        )

        with lock:
            replay_results.append(result)

    threads = [
        threading.Thread(target=replay)
        for _ in range(20)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert len(replay_results) == 20
    assert all(
        result[0] is False
        for result in replay_results
    )


# ============================================================
# Door Grant Expiration Concurrency
# ============================================================


def test_expired_door_grant_rejected_concurrently():
    """
    An expired door grant must never become valid again
    during concurrent access.
    """

    manager = UnifiedAuthSessionManager()

    user_id = 1

    original_ttl = manager.DOOR_GRANT_TTL_SECONDS

    try:
        manager.DOOR_GRANT_TTL_SECONDS = 0.05

        grant = manager.create_door_grant(user_id)

        time.sleep(0.10)

        results = []
        lock = threading.Lock()

        def consume():
            result = manager.consume_door_grant(
                grant=grant,
                user_id=user_id,
            )

            with lock:
                results.append(result)

        threads = [
            threading.Thread(target=consume)
            for _ in range(10)
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 10
        assert all(
            result[0] is False
            for result in results
        )

    finally:
        manager.DOOR_GRANT_TTL_SECONDS = original_ttl


# ============================================================
# Session Concurrency Tests
# ============================================================


def test_multiple_sessions_are_isolated():
    """
    Concurrently created authentication sessions must have
    unique IDs and remain independently addressable.
    """

    manager = UnifiedAuthSessionManager()

    session_ids = []
    lock = threading.Lock()

    def create_session():
        session_id = manager.create_session()

        with lock:
            session_ids.append(session_id)

    threads = [
        threading.Thread(target=create_session)
        for _ in range(20)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert len(session_ids) == 20
    assert len(set(session_ids)) == 20

    for session_id in session_ids:
        assert manager.get_session(session_id) is not None


def test_concurrent_session_invalidation_is_safe():
    """
    Multiple concurrent invalidation attempts against the same
    session must not cause exceptions or corrupt manager state.

    The session is expected to be removed after invalidation.
    """

    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    results = []
    errors = []
    lock = threading.Lock()

    def invalidate():
        try:
            result = manager.invalidate_session(session_id)

            with lock:
                results.append(result)

        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=invalidate)
        for _ in range(10)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # No thread should have crashed.
    assert errors == []

    # Every concurrent operation returned normally.
    assert len(results) == 10

    # The session must no longer be available after invalidation.
    assert manager.get_session(session_id) is None

    # The session ID must not remain in the manager.
    assert session_id not in manager.sessions


# ============================================================
# Manager Cleanup Concurrency
# ============================================================


def test_concurrent_cleanup_does_not_corrupt_door_grants():
    """
    Concurrent cleanup and grant creation must not corrupt
    the in-memory grant store.
    """

    manager = UnifiedAuthSessionManager()

    errors = []
    lock = threading.Lock()

    def create_grants():
        try:
            for _ in range(20):
                manager.create_door_grant(user_id=1)
        except Exception as exc:
            with lock:
                errors.append(exc)

    def cleanup():
        try:
            for _ in range(20):
                manager.cleanup_expired_door_grants()
        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=create_grants)
        for _ in range(5)
    ]

    threads.extend(
        threading.Thread(target=cleanup)
        for _ in range(5)
    )

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert errors == []
    assert len(manager.door_grants) > 0


# ============================================================
# Invalid Input Concurrency
# ============================================================


@pytest.mark.parametrize(
    "invalid_user_id",
    [
        None,
        0,
        -1,
        "1",
        "",
    ],
)
def test_invalid_door_grant_user_ids_are_rejected_concurrently(
    invalid_user_id,
):
    """
    Invalid user IDs must never create a valid authorization
    grant, regardless of concurrent execution.
    """

    manager = UnifiedAuthSessionManager()

    errors = []
    lock = threading.Lock()

    def create_grant():
        try:
            manager.create_door_grant(
                user_id=invalid_user_id,
            )
        except ValueError:
            return
        except Exception as exc:
            with lock:
                errors.append(exc)
        else:
            with lock:
                errors.append(
                    AssertionError(
                        "Invalid user ID created a door grant"
                    )
                )

    threads = [
        threading.Thread(target=create_grant)
        for _ in range(10)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert errors == []