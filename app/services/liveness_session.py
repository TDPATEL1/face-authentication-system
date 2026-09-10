import secrets
import time
from dataclasses import dataclass

from face_service.liveness import LivenessDetector


@dataclass
class LivenessSession:
    detector: LivenessDetector
    created_at: float


class LivenessSessionManager:
    """
    Manages short-lived liveness verification sessions.

    Each session owns its own LivenessDetector instance.
    """

    SESSION_TIMEOUT_SECONDS = 60.0

    def __init__(self):
        self.sessions: dict[
            str,
            LivenessSession
        ] = {}

    # ---------------------------------------------------------
    # Create session
    # ---------------------------------------------------------

    def create_session(self) -> str:
        """
        Create a new liveness session.

        Returns:
            Secure random session ID.
        """

        self.cleanup_expired_sessions()

        session_id = (
            secrets.token_urlsafe(32)
        )

        detector = LivenessDetector()

        self.sessions[session_id] = (
            LivenessSession(
                detector=detector,
                created_at=time.monotonic(),
            )
        )

        return session_id

    # ---------------------------------------------------------
    # Get session
    # ---------------------------------------------------------

    def get_session(
        self,
        session_id: str,
    ) -> LivenessSession | None:
        """
        Retrieve a valid liveness session.
        """

        session = self.sessions.get(
            session_id
        )

        if session is None:
            return None

        elapsed = (
            time.monotonic()
            - session.created_at
        )

        if (
            elapsed
            > self.SESSION_TIMEOUT_SECONDS
        ):

            self.sessions.pop(
                session_id,
                None,
            )

            return None

        return session

    # ---------------------------------------------------------
    # Remove session
    # ---------------------------------------------------------

    def remove_session(
        self,
        session_id: str,
    ) -> None:
        """
        Consume/remove a liveness session.
        """

        self.sessions.pop(
            session_id,
            None,
        )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def cleanup_expired_sessions(
        self,
    ) -> None:
        """
        Remove expired sessions.
        """

        now = time.monotonic()

        expired_sessions = [
            session_id
            for session_id, session
            in self.sessions.items()
            if (
                now
                - session.created_at
                > self.SESSION_TIMEOUT_SECONDS
            )
        ]

        for session_id in expired_sessions:

            self.sessions.pop(
                session_id,
                None,
            )


# =========================================================
# Global Session Manager
# =========================================================

liveness_session_manager = (
    LivenessSessionManager()
)