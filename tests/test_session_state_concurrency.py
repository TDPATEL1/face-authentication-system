import threading

from app.services.unified_auth_session import (
    AuthState,
    UnifiedAuthSessionManager,
)


def prepare_session_at_state(
    manager: UnifiedAuthSessionManager,
    session_id: str,
    state: AuthState,
) -> None:
    """
    Prepare a test session at a specific state without
    exercising the preceding ML pipeline.

    This is test setup only. Production state transitions
    remain unchanged.
    """

    with manager._lock:
        session = manager.get_session(session_id)

        assert session is not None

        session.state = state


def test_concurrent_begin_face_match_only_one_succeeds():
    """
    Multiple concurrent face-match requests must not be able
    to reserve the same session more than once.
    """

    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    prepare_session_at_state(
        manager,
        session_id,
        AuthState.ANTI_SPOOF_PASSED,
    )

    results = []
    errors = []
    lock = threading.Lock()

    def begin_match():
        try:
            result = manager.begin_face_match(
                session_id=session_id,
            )

            with lock:
                results.append(result)

        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=begin_match)
        for _ in range(10)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 10

    successful = [
        result
        for result in results
        if result[0] is True
    ]

    rejected = [
        result
        for result in results
        if result[0] is False
    ]

    assert len(successful) == 1
    assert len(rejected) == 9

    # Later invalid-state attempts invalidate the session.
    assert manager.get_session(session_id) is None


def test_concurrent_validate_and_consume_only_one_succeeds():
    """
    validate_and_consume_for_login() is an alias for the
    atomic face-match reservation.

    Concurrent calls must therefore allow only one success.
    """

    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    prepare_session_at_state(
        manager,
        session_id,
        AuthState.ANTI_SPOOF_PASSED,
    )

    results = []
    errors = []
    lock = threading.Lock()

    def validate():
        try:
            result = manager.validate_and_consume_for_login(
                session_id=session_id,
            )

            with lock:
                results.append(result)

        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=validate)
        for _ in range(10)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 10

    successful = [
        result
        for result in results
        if result[0] is True
    ]

    rejected = [
        result
        for result in results
        if result[0] is False
    ]

    assert len(successful) == 1
    assert len(rejected) == 9

    # Later invalid-state attempts invalidate the session.
    assert manager.get_session(session_id) is None


def test_concurrent_completion_only_one_succeeds():
    """
    Once a session reaches FACE_MATCHED, concurrent completion
    attempts must consume the session exactly once.
    """

    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    prepare_session_at_state(
        manager,
        session_id,
        AuthState.FACE_MATCHED,
    )

    results = []
    errors = []
    lock = threading.Lock()

    def complete():
        try:
            result = manager.complete_session(
                session_id=session_id,
                user_id=1,
                similarity=0.90,
            )

            with lock:
                results.append(result)

        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=complete)
        for _ in range(10)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 10

    assert sum(
        result is True
        for result in results
    ) == 1

    assert sum(
        result is False
        for result in results
    ) == 9

    # Completion consumes the session.
    assert manager.get_session(session_id) is None


def test_out_of_order_face_match_is_rejected():
    """
    Face matching must not be possible directly from CREATED.
    """

    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    result = manager.begin_face_match(
        session_id=session_id,
    )

    assert result[0] is False

    # begin_face_match invalidates the session on an invalid state.
    assert manager.get_session(session_id) is None


def test_out_of_order_completion_is_rejected():
    """
    A session that has not reached FACE_MATCHED must not
    be completed.
    """

    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    result = manager.complete_session(
        session_id=session_id,
        user_id=1,
        similarity=0.90,
    )

    assert result is False

    # Invalid completion must not silently authenticate the session.
    session = manager.get_session(session_id)

    assert session is not None
    assert session.state == AuthState.CREATED


def test_concurrent_out_of_order_face_match_attempts_cannot_bypass_state():
    """
    Concurrent attempts to skip directly from CREATED to
    FACE_MATCHED must all fail.

    Because begin_face_match invalidates the session when the
    state is invalid, the session must not reach FACE_MATCHED.
    """

    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    results = []
    errors = []
    lock = threading.Lock()

    def invalid_match():
        try:
            result = manager.begin_face_match(
                session_id=session_id,
            )

            with lock:
                results.append(result)

        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=invalid_match)
        for _ in range(20)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 20

    assert all(
        result[0] is False
        for result in results
    )

    # No request may have bypassed the state machine.
    assert manager.get_session(session_id) is None
