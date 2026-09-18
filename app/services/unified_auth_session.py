import logging
import secrets
import threading
import time
from enum import Enum
from typing import Any, Dict, Optional

from face_service.liveness import LivenessDetector


logger = logging.getLogger(__name__)


class AuthState(str, Enum):
    CREATED = "CREATED"
    LIVENESS_PASSED = "LIVENESS_PASSED"
    ANTI_SPOOF_PASSED = "ANTI_SPOOF_PASSED"
    FACE_MATCHED = "FACE_MATCHED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class DoorAuthorizationGrant:
    """
    Short-lived, one-time authorization for a physical door unlock.

    The grant is:
    - generated server-side
    - bound to a specific authenticated user
    - short-lived
    - consumed atomically
    """

    def __init__(
        self,
        user_id: int,
        ttl_seconds: float,
    ):
        self.user_id = user_id
        self.expires_at = time.monotonic() + ttl_seconds

    def is_expired(self) -> bool:
        return time.monotonic() >= self.expires_at


class UnifiedAuthSession:
    def __init__(self, session_id: str):
        self.session_id: str = session_id

        # Server-generated one-time request nonce.
        self.current_nonce: str = secrets.token_urlsafe(32)

        # A frame request may reserve the current nonce while its
        # image is decoded and processed outside the manager lock.
        self.nonce_reserved: bool = False

        self.state: AuthState = AuthState.CREATED

        now = time.monotonic()
        self.created_at: float = now
        self.last_activity: float = now

        # Each unified session owns its own liveness detector.
        self.liveness_detector: LivenessDetector = LivenessDetector()

        # Multi-frame anti-spoof predictions.
        self.anti_spoof_predictions: list[Dict[str, Any]] = []

        # Authentication results.
        self.matched_user_id: Optional[int] = None
        self.matched_similarity: Optional[float] = None
        self.anti_spoof_summary: Optional[Dict[str, Any]] = None

    def touch(self) -> None:
        """Update the inactivity timer."""
        self.last_activity = time.monotonic()

    def is_expired(self, timeout_seconds: float) -> bool:
        """
        Expire the session after a period of inactivity.

        Terminal states are considered unavailable immediately.
        """
        if self.state in (
            AuthState.COMPLETED,
            AuthState.FAILED,
            AuthState.EXPIRED,
        ):
            return True

        return (
            time.monotonic() - self.last_activity
        ) > timeout_seconds


class UnifiedAuthSessionManager:
    """
    Thread-safe unified authentication state machine.

    Authoritative flow:

        CREATED
            ↓
        LIVENESS_PASSED
            ↓
        ANTI_SPOOF_PASSED
            ↓
        FACE_MATCHED
            ↓
        COMPLETED

    Security properties:

    - One authoritative session per authentication attempt.
    - Out-of-order operations invalidate the session.
    - Face matching is atomically reserved by changing
      ANTI_SPOOF_PASSED -> FACE_MATCHED.
    - Only FACE_MATCHED sessions can be completed.
    - Completion consumes the session.
    - Expired sessions are removed.
    - All state mutations are protected by an RLock.
    - Frame nonces are reserved and committed atomically.
    - Door authorization grants are server-generated,
      user-bound, short-lived, and one-time use.

    This manager is in-process only. A multi-worker deployment needs
    shared session/grant storage and atomic operations to preserve
    these guarantees across workers.
    """

    SESSION_TIMEOUT_SECONDS: float = 90.0

    # Door authorization is intentionally much shorter than the
    # authentication-session lifetime.
    DOOR_GRANT_TTL_SECONDS: float = 15.0

    REQUIRED_ANTI_SPOOF_FRAMES: int = 5
    MIN_REAL_ANTI_SPOOF_FRAMES: int = 4
    MIN_AVERAGE_ANTI_SPOOF_CONFIDENCE: float = 0.80

    # MiniFASNetV2 REAL class.
    REAL_CLASS_ID: int = 2

    def __init__(self):
        self.sessions: Dict[str, UnifiedAuthSession] = {}

        # grant -> DoorAuthorizationGrant
        self.door_grants: Dict[str, DoorAuthorizationGrant] = {}

        self._lock = threading.RLock()

    # ============================================================
    # SESSION CREATION
    # ============================================================

    def create_session(self) -> str:
        with self._lock:
            self.cleanup_expired_sessions()

            session_id = secrets.token_urlsafe(32)

            self.sessions[session_id] = UnifiedAuthSession(
                session_id=session_id
            )

            logger.info(
                "Unified auth session created: session_id=%s",
                session_id,
            )

            return session_id

    # ============================================================
    # SESSION LOOKUP
    # ============================================================

    def get_session(
        self,
        session_id: str,
    ) -> Optional[UnifiedAuthSession]:

        with self._lock:
            session = self.sessions.get(session_id)

            if session is None:
                return None

            if session.is_expired(
                self.SESSION_TIMEOUT_SECONDS
            ):
                logger.warning(
                    "Unified auth session expired: session_id=%s",
                    session_id,
                )

                session.state = AuthState.EXPIRED
                self.sessions.pop(session_id, None)

                return None

            return session

    # ============================================================
    # REPLAY PROTECTION
    # ============================================================

    def reserve_nonce(
        self,
        session_id: str,
        nonce: str,
    ) -> tuple[bool, str]:

        with self._lock:

            session = self.get_session(session_id)

            if session is None:
                return False, "Session is invalid or expired"

            if not nonce:
                logger.warning(
                    "Replay protection rejected empty nonce: "
                    "session_id=%s",
                    session_id,
                )

                return False, "Request nonce is required"

            if not secrets.compare_digest(
                session.current_nonce,
                nonce,
            ):
                logger.warning(
                    "Replay protection rejected invalid/replayed "
                    "nonce: session_id=%s",
                    session_id,
                )

                return False, "Invalid or replayed request nonce"

            if session.nonce_reserved:
                logger.warning(
                    "Replay protection rejected nonce already "
                    "reserved by an in-flight request: session_id=%s",
                    session_id,
                )

                return (
                    False,
                    "Request nonce is already being processed",
                )

            session.nonce_reserved = True
            session.touch()

            logger.info(
                "Authentication request nonce reserved: "
                "session_id=%s",
                session_id,
            )

            return True, "Request nonce reserved"

    def commit_nonce_reservation(
        self,
        session_id: str,
        nonce: str,
    ) -> tuple[bool, str, Optional[str]]:

        with self._lock:

            session = self.get_session(session_id)

            if session is None:
                return (
                    False,
                    "Session is invalid or expired",
                    None,
                )

            if (
                not nonce
                or not session.nonce_reserved
                or not secrets.compare_digest(
                    session.current_nonce,
                    nonce,
                )
            ):
                logger.warning(
                    "Nonce reservation commit rejected: "
                    "session_id=%s",
                    session_id,
                )

                return (
                    False,
                    "Invalid or replayed request nonce",
                    None,
                )

            session.current_nonce = secrets.token_urlsafe(32)
            session.nonce_reserved = False
            session.touch()

            logger.info(
                "Authentication request nonce consumed "
                "and rotated: session_id=%s",
                session_id,
            )

            return (
                True,
                "Request nonce accepted",
                session.current_nonce,
            )

    def rollback_nonce_reservation(
        self,
        session_id: str,
        nonce: str,
    ) -> bool:

        with self._lock:

            session = self.get_session(session_id)

            if (
                session is None
                or not nonce
                or not session.nonce_reserved
                or not secrets.compare_digest(
                    session.current_nonce,
                    nonce,
                )
            ):
                return False

            session.nonce_reserved = False
            session.touch()

            logger.info(
                "Authentication request nonce reservation released: "
                "session_id=%s",
                session_id,
            )

            return True

    # ============================================================
    # LIVENESS
    # ============================================================

    def process_liveness_frame(
        self,
        session_id: str,
        face: Any,
    ) -> Dict[str, Any]:

        with self._lock:

            session = self.get_session(session_id)

            if session is None:
                return {
                    "valid": False,
                    "reason": "Session invalid or expired",
                    "state": AuthState.EXPIRED.value,
                }

            if session.state not in (
                AuthState.CREATED,
                AuthState.LIVENESS_PASSED,
            ):
                logger.warning(
                    "Invalid liveness state: "
                    "session_id=%s state=%s",
                    session_id,
                    session.state,
                )

                self.invalidate_session(session_id)

                return {
                    "valid": False,
                    "reason": (
                        "Liveness cannot be run in "
                        f"state {session.state.value}"
                    ),
                    "state": AuthState.FAILED.value,
                }

            session.touch()

            liveness_passed = (
                session.liveness_detector.check(face)
            )

            if liveness_passed:
                session.state = AuthState.LIVENESS_PASSED

                logger.info(
                    "Liveness passed: session_id=%s",
                    session_id,
                )

            return {
                "valid": True,
                "liveness": (
                    session.state
                    == AuthState.LIVENESS_PASSED
                ),
                "status": (
                    "passed"
                    if session.state
                    == AuthState.LIVENESS_PASSED
                    else "checking"
                ),
                "challenge": (
                    session.liveness_detector.challenge
                ),
                "challenge_message": (
                    session.liveness_detector.challenge_message()
                ),
                "movement_count": (
                    session.liveness_detector.movement_count
                ),
                "state": session.state.value,
            }

    # ============================================================
    # ANTI-SPOOFING
    # ============================================================

    def add_anti_spoof_prediction(
        self,
        session_id: str,
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:

        with self._lock:

            session = self.get_session(session_id)

            if session is None:
                return {
                    "valid": False,
                    "reason": "Session invalid or expired",
                    "state": AuthState.EXPIRED.value,
                }

            if session.state != AuthState.LIVENESS_PASSED:

                logger.warning(
                    "Invalid anti-spoof state: "
                    "session_id=%s state=%s",
                    session_id,
                    session.state,
                )

                self.invalidate_session(session_id)

                return {
                    "valid": False,
                    "reason": (
                        "Liveness verification must pass "
                        "before anti-spoofing"
                    ),
                    "state": AuthState.FAILED.value,
                }

            session.touch()

            session.anti_spoof_predictions.append(
                prediction
            )

            frame_count = len(
                session.anti_spoof_predictions
            )

            if frame_count < self.REQUIRED_ANTI_SPOOF_FRAMES:
                return {
                    "valid": True,
                    "status": "checking",
                    "frame_number": frame_count,
                    "required_frames": (
                        self.REQUIRED_ANTI_SPOOF_FRAMES
                    ),
                    "state": session.state.value,
                }

            real_predictions = [
                prediction
                for prediction
                in session.anti_spoof_predictions
                if prediction.get("class_id")
                == self.REAL_CLASS_ID
            ]

            real_count = len(real_predictions)

            confidences = [
                float(
                    prediction.get(
                        "confidence",
                        0.0,
                    )
                )
                for prediction in real_predictions
            ]

            average_confidence = (
                sum(confidences) / len(confidences)
                if confidences
                else 0.0
            )

            is_real = (
                real_count
                >= self.MIN_REAL_ANTI_SPOOF_FRAMES
                and average_confidence
                >= self.MIN_AVERAGE_ANTI_SPOOF_CONFIDENCE
            )

            summary = {
                "is_real": is_real,
                "frames_analyzed": frame_count,
                "real_frames": real_count,
                "spoof_frames": (
                    frame_count - real_count
                ),
                "average_real_confidence": round(
                    average_confidence,
                    4,
                ),
            }

            session.anti_spoof_summary = summary

            if is_real:

                session.state = (
                    AuthState.ANTI_SPOOF_PASSED
                )

                session.touch()

                logger.info(
                    "Anti-spoofing passed: "
                    "session_id=%s",
                    session_id,
                )

                return {
                    "valid": True,
                    "status": "passed",
                    "state": session.state.value,
                    **summary,
                }

            logger.warning(
                "Anti-spoofing failed: session_id=%s",
                session_id,
            )

            self.invalidate_session(session_id)

            return {
                "valid": False,
                "status": "failed",
                "reason": (
                    "Anti-spoofing verification failed"
                ),
                "state": AuthState.FAILED.value,
                **summary,
            }

    # ============================================================
    # ATOMIC FACE-MATCH RESERVATION
    # ============================================================

    def begin_face_match(
        self,
        session_id: str,
    ) -> tuple[bool, str]:

        with self._lock:

            session = self.get_session(session_id)

            if session is None:
                return False, "Session is invalid or expired"

            if session.state != AuthState.ANTI_SPOOF_PASSED:

                current_state = session.state.value

                logger.warning(
                    "Face-match attempted in invalid state: "
                    "session_id=%s state=%s",
                    session_id,
                    current_state,
                )

                self.invalidate_session(session_id)

                return (
                    False,
                    (
                        "Authentication requires liveness "
                        "and anti-spoofing. "
                        f"Current state: {current_state}"
                    ),
                )

            session.state = AuthState.FACE_MATCHED
            session.touch()

            logger.info(
                "Face matching atomically reserved: "
                "session_id=%s",
                session_id,
            )

            return True, "Session reserved for face matching"

    # ============================================================
    # BACKWARD-COMPATIBLE METHOD NAME
    # ============================================================

    def validate_and_consume_for_login(
        self,
        session_id: str,
    ) -> tuple[bool, str]:

        return self.begin_face_match(session_id)

    # ============================================================
    # COMPLETE AUTHENTICATION
    # ============================================================

    def complete_session(
        self,
        session_id: str,
        user_id: int,
        similarity: float,
    ) -> bool:

        with self._lock:

            session = self.sessions.get(session_id)

            if session is None:
                return False

            if session.state != AuthState.FACE_MATCHED:

                logger.warning(
                    "Cannot complete session in state %s: "
                    "session_id=%s",
                    session.state,
                    session_id,
                )

                return False

            session.matched_user_id = user_id
            session.matched_similarity = similarity
            session.state = AuthState.COMPLETED

            self.sessions.pop(
                session_id,
                None,
            )

            logger.info(
                "Unified auth session completed and consumed: "
                "session_id=%s user_id=%s",
                session_id,
                user_id,
            )

            return True

    # ============================================================
    # DOOR AUTHORIZATION GRANTS
    # ============================================================

    def create_door_grant(
        self,
        user_id: int,
    ) -> str:
        """
        Create a short-lived, one-time door authorization grant.

        The caller must only invoke this after successful biometric
        authentication.
        """

        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("Invalid user_id")

        with self._lock:
            self.cleanup_expired_door_grants()

            grant = secrets.token_urlsafe(32)

            self.door_grants[grant] = DoorAuthorizationGrant(
                user_id=user_id,
                ttl_seconds=self.DOOR_GRANT_TTL_SECONDS,
            )

            logger.info(
                "One-time door authorization grant created "
                "for user_id=%s",
                user_id,
            )

            return grant

    def consume_door_grant(
        self,
        grant: str,
        user_id: int,
    ) -> tuple[bool, str]:
        """
        Atomically validate and consume a door authorization grant.

        A successful call permanently consumes the grant.
        """

        with self._lock:

            if not grant or not isinstance(grant, str):
                return False, "Door authorization is required"

            if not isinstance(user_id, int) or user_id <= 0:
                return False, "Invalid authenticated user"

            stored_grant = self.door_grants.get(grant)

            if stored_grant is None:
                logger.warning(
                    "Door authorization rejected: "
                    "unknown or already consumed grant"
                )

                return False, "Invalid or expired door authorization"

            if stored_grant.is_expired():
                self.door_grants.pop(grant, None)

                logger.warning(
                    "Door authorization rejected: expired grant"
                )

                return False, "Invalid or expired door authorization"

            if stored_grant.user_id != user_id:
                logger.warning(
                    "Door authorization rejected: user mismatch "
                    "authenticated_user_id=%s grant_user_id=%s",
                    user_id,
                    stored_grant.user_id,
                )

                return False, "Invalid door authorization"

            # Atomic one-time consumption.
            self.door_grants.pop(grant, None)

            logger.info(
                "One-time door authorization consumed "
                "for user_id=%s",
                user_id,
            )

            return True, "Door authorization accepted"

    def cleanup_expired_door_grants(self) -> None:
        """Remove expired door grants."""

        expired_grants = [
            grant
            for grant, authorization
            in self.door_grants.items()
            if authorization.is_expired()
        ]

        for grant in expired_grants:
            self.door_grants.pop(grant, None)

    # ============================================================
    # INVALIDATE
    # ============================================================

    def invalidate_session(
        self,
        session_id: str,
    ) -> None:

        with self._lock:

            session = self.sessions.pop(
                session_id,
                None,
            )

            if session is not None:

                session.state = AuthState.FAILED

                logger.info(
                    "Unified auth session invalidated: "
                    "session_id=%s",
                    session_id,
                )

    # ============================================================
    # CLEANUP
    # ============================================================

    def cleanup_expired_sessions(self) -> None:

        with self._lock:

            expired = [
                session_id
                for session_id, session
                in self.sessions.items()
                if session.is_expired(
                    self.SESSION_TIMEOUT_SECONDS
                )
            ]

            for session_id in expired:

                self.sessions.pop(
                    session_id,
                    None,
                )

                logger.info(
                    "Expired unified auth session cleaned: "
                    "session_id=%s",
                    session_id,
                )


unified_auth_session_manager = (
    UnifiedAuthSessionManager()
)