import logging
import threading
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.services.unified_auth_session import (
    AuthState,
    UnifiedAuthSessionManager,
)
from app.api.routes import face as face_routes


client = TestClient(app)


def test_session_creation():
    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    assert session_id is not None
    assert len(session_id) > 10

    session = manager.get_session(session_id)

    assert session is not None
    assert session.state == AuthState.CREATED


def test_current_nonce_can_be_reserved():
    manager = UnifiedAuthSessionManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)

    assert session is not None
    nonce = session.current_nonce

    valid, reason = manager.reserve_nonce(session_id, nonce)

    assert valid is True
    assert "reserved" in reason.lower()
    assert session.nonce_reserved is True
    assert manager.rollback_nonce_reservation(session_id, nonce)


def test_nonce_replay_is_rejected_after_commit():
    manager = UnifiedAuthSessionManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)

    assert session is not None
    nonce = session.current_nonce

    assert manager.reserve_nonce(session_id, nonce)[0] is True
    committed, _, next_nonce = manager.commit_nonce_reservation(
        session_id,
        nonce,
    )

    assert committed is True
    assert next_nonce is not None

    replay_valid, _ = manager.reserve_nonce(session_id, nonce)

    assert replay_valid is False


def test_invalid_nonce_does_not_rotate_current_nonce():
    manager = UnifiedAuthSessionManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)

    assert session is not None
    current_nonce = session.current_nonce

    valid, _ = manager.reserve_nonce(session_id, "invalid-nonce")

    assert valid is False
    assert session.current_nonce == current_nonce
    assert session.nonce_reserved is False
    assert manager.reserve_nonce(session_id, current_nonce)[0] is True
    assert manager.rollback_nonce_reservation(session_id, current_nonce)


def test_empty_nonce_is_rejected_without_rotation():
    manager = UnifiedAuthSessionManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)

    assert session is not None
    current_nonce = session.current_nonce

    valid, reason = manager.reserve_nonce(session_id, "")

    assert valid is False
    assert "required" in reason.lower()
    assert session.current_nonce == current_nonce
    assert session.nonce_reserved is False


def test_missing_nonce_is_rejected_without_state_mutation(monkeypatch):
    manager = UnifiedAuthSessionManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)

    assert session is not None
    current_nonce = session.current_nonce
    monkeypatch.setattr(
        face_routes,
        "unified_auth_session_manager",
        manager,
    )

    response = client.post(
        f"/api/v1/face/liveness/frame?session_id={session_id}",
    )

    assert response.status_code == 422
    assert session.current_nonce == current_nonce
    assert session.nonce_reserved is False


def test_wrong_session_cannot_use_nonce():
    manager = UnifiedAuthSessionManager()
    first_session_id = manager.create_session()
    second_session_id = manager.create_session()
    first_session = manager.get_session(first_session_id)
    second_session = manager.get_session(second_session_id)

    assert first_session is not None
    assert second_session is not None
    first_nonce = first_session.current_nonce
    second_nonce = second_session.current_nonce

    valid, _ = manager.reserve_nonce(second_session_id, first_nonce)

    assert valid is False
    assert second_session.current_nonce == second_nonce
    assert manager.reserve_nonce(first_session_id, first_nonce)[0] is True
    assert manager.rollback_nonce_reservation(
        first_session_id,
        first_nonce,
    )


def test_expired_session_rejects_nonce_reservation():
    manager = UnifiedAuthSessionManager()
    manager.SESSION_TIMEOUT_SECONDS = 0.1
    session_id = manager.create_session()
    session = manager.get_session(session_id)

    assert session is not None
    nonce = session.current_nonce
    session.last_activity = time.monotonic() - 1

    valid, reason = manager.reserve_nonce(session_id, nonce)

    assert valid is False
    assert "expired" in reason.lower()
    assert manager.get_session(session_id) is None


def test_nonce_rotation_rejects_old_nonce_and_accepts_new_nonce():
    manager = UnifiedAuthSessionManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)

    assert session is not None
    old_nonce = session.current_nonce

    assert manager.reserve_nonce(session_id, old_nonce)[0] is True
    committed, _, new_nonce = manager.commit_nonce_reservation(
        session_id,
        old_nonce,
    )

    assert committed is True
    assert new_nonce is not None
    assert new_nonce != old_nonce
    assert manager.reserve_nonce(session_id, old_nonce)[0] is False
    assert manager.reserve_nonce(session_id, new_nonce)[0] is True
    assert manager.rollback_nonce_reservation(session_id, new_nonce)


def test_invalid_image_releases_nonce_reservation(monkeypatch):
    manager = UnifiedAuthSessionManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)

    assert session is not None
    nonce = session.current_nonce
    monkeypatch.setattr(
        face_routes,
        "unified_auth_session_manager",
        manager,
    )

    response = client.post(
        f"/api/v1/face/liveness/frame?session_id={session_id}&nonce={nonce}",
        files={
            "image": (
                "corrupt.jpg",
                b"not-a-decodable-image",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400
    assert session.current_nonce == nonce
    assert session.nonce_reserved is False
    assert manager.reserve_nonce(session_id, nonce)[0] is True
    assert manager.rollback_nonce_reservation(session_id, nonce)


def test_concurrent_same_nonce_reserves_exactly_once():
    manager = UnifiedAuthSessionManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)

    assert session is not None
    nonce = session.current_nonce
    barrier = threading.Barrier(2)
    results = []

    def reserve_same_nonce():
        barrier.wait()
        results.append(manager.reserve_nonce(session_id, nonce))

    first = threading.Thread(target=reserve_same_nonce)
    second = threading.Thread(target=reserve_same_nonce)
    first.start()
    second.start()
    first.join()
    second.join()

    successful = [result for result in results if result[0]]
    rejected = [result for result in results if not result[0]]

    assert len(successful) == 1
    assert len(rejected) == 1
    assert manager.rollback_nonce_reservation(session_id, nonce)


def test_concurrent_liveness_requests_same_nonce_allow_one_success(
    monkeypatch,
):
    manager = UnifiedAuthSessionManager()
    session_id = manager.create_session()
    session = manager.get_session(session_id)

    assert session is not None
    nonce = session.current_nonce
    monkeypatch.setattr(
        face_routes,
        "unified_auth_session_manager",
        manager,
    )

    detector_started = threading.Event()
    release_detector = threading.Event()
    rejected_response_received = threading.Event()
    start_barrier = threading.Barrier(3)
    responses = []

    def blocking_no_face_detector(_frame):
        detector_started.set()
        assert release_detector.wait(timeout=5)
        return []

    monkeypatch.setattr(
        face_routes.face_matcher.detector,
        "detect",
        blocking_no_face_detector,
    )

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode(".jpg", image)
    image_bytes = encoded_image.tobytes()

    def submit_frame():
        start_barrier.wait()
        response = client.post(
            (
                "/api/v1/face/liveness/frame?"
                f"session_id={session_id}&nonce={nonce}"
            ),
            files={
                "image": (
                    "frame.jpg",
                    image_bytes,
                    "image/jpeg",
                )
            },
        )
        responses.append(response)

        if response.status_code == 401:
            rejected_response_received.set()

    first = threading.Thread(target=submit_frame)
    second = threading.Thread(target=submit_frame)
    first.start()
    second.start()
    start_barrier.wait()

    assert detector_started.wait(timeout=5)
    assert rejected_response_received.wait(timeout=5)
    release_detector.set()
    first.join()
    second.join()

    assert sorted(response.status_code for response in responses) == [
        200,
        401,
    ]


def test_nonce_values_are_not_logged(caplog):
    manager = UnifiedAuthSessionManager()

    with caplog.at_level(
        logging.INFO,
        logger="app.services.unified_auth_session",
    ):
        session_id = manager.create_session()
        session = manager.get_session(session_id)

        assert session is not None
        nonce = session.current_nonce
        assert manager.reserve_nonce(session_id, nonce)[0] is True
        committed, _, next_nonce = manager.commit_nonce_reservation(
            session_id,
            nonce,
        )

    assert committed is True
    assert next_nonce is not None
    assert nonce not in caplog.text
    assert next_nonce not in caplog.text


def test_valid_state_transitions():
    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()
    session = manager.get_session(session_id)

    assert session is not None

    # ---------------------------------------------------------
    # 1. Process liveness movement until passed
    # ---------------------------------------------------------

    session.liveness_detector.previous_position = (
        100.0,
        100.0,
    )

    session.liveness_detector.challenge = "TURN_RIGHT"

    # Frame 1 - movement right
    res1 = manager.process_liveness_frame(
        session_id,
        [110.0, 100.0, 100.0, 100.0],
    )

    assert res1["valid"] is True

    # Frame 2 - movement right
    res2 = manager.process_liveness_frame(
        session_id,
        [120.0, 100.0, 100.0, 100.0],
    )

    assert res2["valid"] is True
    assert session.state == AuthState.LIVENESS_PASSED

    # ---------------------------------------------------------
    # 2. Process 5 anti-spoof frames
    # ---------------------------------------------------------

    for _ in range(5):
        prediction = {
            "class_id": 2,
            "confidence": 0.95,
        }

        res_as = manager.add_anti_spoof_prediction(
            session_id,
            prediction,
        )

        assert res_as["valid"] is True

    assert session.state == AuthState.ANTI_SPOOF_PASSED

    # ---------------------------------------------------------
    # 3. Reserve session for face matching
    # ---------------------------------------------------------

    valid, message = manager.begin_face_match(session_id)

    assert valid is True
    assert "reserved" in message.lower()

    session = manager.get_session(session_id)

    assert session is not None
    assert session.state == AuthState.FACE_MATCHED

    # ---------------------------------------------------------
    # 4. Complete authentication
    # ---------------------------------------------------------

    completed = manager.complete_session(
        session_id,
        user_id=1,
        similarity=0.85,
    )

    assert completed is True

    # Session must be consumed
    assert manager.get_session(session_id) is None


def test_liveness_before_anti_spoof_requirement():
    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    # Attempt anti-spoof directly without liveness
    prediction = {
        "class_id": 2,
        "confidence": 0.95,
    }

    res = manager.add_anti_spoof_prediction(
        session_id,
        prediction,
    )

    assert res["valid"] is False
    assert res["state"] == AuthState.FAILED.value

    # Invalid session must be removed
    assert manager.get_session(session_id) is None


def test_anti_spoof_before_face_login_requirement():
    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    # Attempt face login before liveness/anti-spoof
    valid, message = (
        manager.validate_and_consume_for_login(
            session_id
        )
    )

    assert valid is False

    assert (
        "liveness and anti-spoofing" in message.lower()
        or "authentication requires" in message.lower()
    )

    # Invalid session must be removed
    assert manager.get_session(session_id) is None


def test_session_expiration():
    manager = UnifiedAuthSessionManager()

    # Very short timeout for testing
    manager.SESSION_TIMEOUT_SECONDS = 0.1

    session_id = manager.create_session()

    time.sleep(0.15)

    session = manager.get_session(session_id)

    assert session is None


def test_session_reuse():
    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    session = manager.get_session(session_id)

    assert session is not None

    # Put session into the correct state
    # before attempting face-match reservation.
    session.state = AuthState.ANTI_SPOOF_PASSED

    # ---------------------------------------------------------
    # First authentication attempt
    # ---------------------------------------------------------

    valid, message = manager.begin_face_match(
        session_id
    )

    assert valid is True
    assert "reserved" in message.lower()

    # Session should now be FACE_MATCHED
    session = manager.get_session(session_id)

    assert session is not None
    assert session.state == AuthState.FACE_MATCHED

    # Complete authentication
    result = manager.complete_session(
        session_id,
        user_id=1,
        similarity=0.90,
    )

    assert result is True

    # Session must be consumed
    assert manager.get_session(session_id) is None

    # ---------------------------------------------------------
    # Second attempt using same session
    # ---------------------------------------------------------

    valid, message = manager.begin_face_match(
        session_id
    )

    assert valid is False

    result = manager.complete_session(
        session_id,
        user_id=1,
        similarity=0.90,
    )

    assert result is False


def test_invalid_session_id():
    manager = UnifiedAuthSessionManager()

    session = manager.get_session(
        "invalid_session_12345"
    )

    assert session is None

    valid, message = (
        manager.validate_and_consume_for_login(
            "invalid_session_12345"
        )
    )

    assert valid is False


def test_concurrent_session_access():
    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    session = manager.get_session(session_id)

    assert session is not None

    # Prepare session for face-match reservation
    session.state = AuthState.ANTI_SPOOF_PASSED

    results = []

    def attempt_face_match():
        result = manager.begin_face_match(
            session_id
        )

        results.append(result)

    # Ten simultaneous authentication attempts
    threads = [
        threading.Thread(
            target=attempt_face_match
        )
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

    # ---------------------------------------------------------
    # Security requirement:
    # Only ONE request may reserve the session.
    # ---------------------------------------------------------

    assert len(successful) == 1
    assert len(failed) == 9

    # The session may still exist in FACE_MATCHED state.
    # It may also have been removed depending on manager
    # cleanup/consumption behavior.
    session = manager.get_session(session_id)

    if session is not None:
        assert session.state == AuthState.FACE_MATCHED


def test_only_one_request_can_reserve_face_match():
    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    session = manager.get_session(session_id)

    assert session is not None

    session.state = AuthState.ANTI_SPOOF_PASSED

    results = []

    def attempt_reservation():
        result = manager.begin_face_match(
            session_id
        )

        results.append(result)

    threads = [
        threading.Thread(
            target=attempt_reservation
        )
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

    assert len(successful) == 1
    assert len(failed) == 9

    session = manager.get_session(session_id)

    if session is not None:
        assert session.state == AuthState.FACE_MATCHED


def test_completion_requires_face_matched_state():
    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    session = manager.get_session(session_id)

    assert session is not None

    # Deliberately try to complete too early.
    session.state = AuthState.ANTI_SPOOF_PASSED

    completed = manager.complete_session(
        session_id,
        user_id=1,
        similarity=0.95,
    )

    assert completed is False

    session = manager.get_session(session_id)

    assert session is not None
    assert session.state == AuthState.ANTI_SPOOF_PASSED


def test_face_match_reservation_then_completion():
    manager = UnifiedAuthSessionManager()

    session_id = manager.create_session()

    session = manager.get_session(session_id)

    assert session is not None

    session.state = AuthState.ANTI_SPOOF_PASSED

    # Reserve
    valid, message = manager.begin_face_match(
        session_id
    )

    assert valid is True
    assert "reserved" in message.lower()

    session = manager.get_session(session_id)

    assert session is not None
    assert session.state == AuthState.FACE_MATCHED

    # Complete
    completed = manager.complete_session(
        session_id,
        user_id=1,
        similarity=0.91,
    )

    assert completed is True

    # Session is one-time-use
    assert manager.get_session(session_id) is None


def test_session_expires_after_inactivity():
    manager = UnifiedAuthSessionManager()

    manager.SESSION_TIMEOUT_SECONDS = 0.1

    session_id = manager.create_session()

    time.sleep(0.15)

    session = manager.get_session(session_id)

    assert session is None


def test_api_session_security_flow():
    # ---------------------------------------------------------
    # 1. Start unified authentication session
    # ---------------------------------------------------------

    res_start = client.post(
        "/api/v1/face/session/start"
    )

    assert res_start.status_code == 200

    data_start = res_start.json()

    session_id = data_start["session_id"]

    assert session_id

    # ---------------------------------------------------------
    # 2. Try face login directly
    #    This skips liveness and anti-spoofing.
    # ---------------------------------------------------------

    res_direct_login = client.post(
        f"/api/v1/face/login?session_id={session_id}"
    )

    assert res_direct_login.status_code == 401

    detail = (
        res_direct_login
        .json()["detail"]
        .lower()
    )

    assert (
        "liveness and anti-spoofing" in detail
        or "authentication requires" in detail
        or "invalid" in detail
        or "expired" in detail
    )

    # ---------------------------------------------------------
    # 3. Session must not be reusable
    # ---------------------------------------------------------

    res_reuse = client.post(
        f"/api/v1/face/login?session_id={session_id}"
    )

    assert res_reuse.status_code == 401
