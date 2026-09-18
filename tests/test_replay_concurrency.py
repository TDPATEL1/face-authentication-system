import threading

from app.services.unified_auth_session import (
    UnifiedAuthSessionManager,
)


def test_concurrent_nonce_reservation_only_one_succeeds():
    """
    A single nonce must not be reservable by multiple
    concurrent requests.

    Exactly one request must reserve the nonce.
    All other concurrent requests must be rejected.
    """

    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    session = manager.get_session(session_id)

    assert session is not None
    assert session.current_nonce is not None

    nonce = session.current_nonce

    results = []
    errors = []
    lock = threading.Lock()

    def reserve_nonce():
        try:
            result = manager.reserve_nonce(
                session_id=session_id,
                nonce=nonce,
            )

            with lock:
                results.append(result)

        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=reserve_nonce)
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

    # The nonce must remain reserved until the request
    # either commits or rolls back.
    assert session.nonce_reserved is True


def test_nonce_cannot_be_reused_after_commit():
    """
    Once a nonce reservation has been committed, the nonce
    is consumed and rotated. Replay attempts using the old
    nonce must be rejected.
    """

    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    session = manager.get_session(session_id)

    assert session is not None
    assert session.current_nonce is not None

    original_nonce = session.current_nonce

    reserved, message = manager.reserve_nonce(
        session_id=session_id,
        nonce=original_nonce,
    )

    assert reserved is True
    assert message == "Request nonce reserved"
    assert session.nonce_reserved is True

    committed, commit_message, new_nonce = (
        manager.commit_nonce_reservation(
            session_id=session_id,
            nonce=original_nonce,
        )
    )

    assert committed is True
    assert commit_message == "Request nonce accepted"
    assert new_nonce is not None
    assert new_nonce != original_nonce

    # The reservation has been consumed.
    assert session.nonce_reserved is False

    # The session now expects a new nonce.
    assert session.current_nonce == new_nonce

    results = []
    errors = []
    lock = threading.Lock()

    def replay_nonce():
        try:
            result = manager.reserve_nonce(
                session_id=session_id,
                nonce=original_nonce,
            )

            with lock:
                results.append(result)

        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=replay_nonce)
        for _ in range(20)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 20

    # Every replay attempt using the consumed nonce
    # must be rejected.
    assert all(
        result[0] is False
        for result in results
    )


def test_concurrent_wrong_nonce_attempts_are_rejected():
    """
    Concurrent requests using invalid nonces must never
    reserve the session's legitimate nonce.
    """

    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    session = manager.get_session(session_id)

    assert session is not None
    assert session.current_nonce is not None

    real_nonce = session.current_nonce

    invalid_nonces = [
        f"invalid-nonce-{index}"
        for index in range(10)
    ]

    results = []
    errors = []
    lock = threading.Lock()

    def try_invalid_nonce(invalid_nonce):
        try:
            result = manager.reserve_nonce(
                session_id=session_id,
                nonce=invalid_nonce,
            )

            with lock:
                results.append(result)

        except Exception as exc:
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(
            target=try_invalid_nonce,
            args=(invalid_nonce,),
        )
        for invalid_nonce in invalid_nonces
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 10

    assert all(
        result[0] is False
        for result in results
    )

    # The legitimate nonce remains unchanged.
    assert session.current_nonce == real_nonce

    # Invalid requests must not reserve the legitimate nonce.
    assert session.nonce_reserved is False
