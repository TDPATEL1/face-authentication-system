import time
import secrets
from dataclasses import dataclass, field


@dataclass
class AntiSpoofingSession:
    session_id: str
    created_at: float
    predictions: list[dict] = field(default_factory=list)


class AntiSpoofingSessionManager:
    """
    Manages short-lived multi-frame anti-spoofing sessions.

    A session collects several MiniFASNetV2 predictions
    before producing the final anti-spoofing decision.
    """

    SESSION_TIMEOUT_SECONDS = 10.0

    REQUIRED_FRAMES = 5

    MIN_REAL_FRAMES = 4

    MIN_AVERAGE_CONFIDENCE = 0.80

    def __init__(self):
        self.sessions: dict[str, AntiSpoofingSession] = {}

    def create_session(self) -> str:
        self.cleanup_expired_sessions()

        session_id = secrets.token_urlsafe(32)

        self.sessions[session_id] = AntiSpoofingSession(
            session_id=session_id,
            created_at=time.monotonic(),
        )

        return session_id

    def get_session(
        self,
        session_id: str,
    ) -> AntiSpoofingSession | None:

        session = self.sessions.get(session_id)

        if session is None:
            return None

        if (
            time.monotonic() - session.created_at
            > self.SESSION_TIMEOUT_SECONDS
        ):
            self.sessions.pop(session_id, None)
            return None

        return session

    def add_prediction(
        self,
        session_id: str,
        prediction: dict,
    ) -> AntiSpoofingSession | None:

        session = self.get_session(session_id)

        if session is None:
            return None

        session.predictions.append(prediction)

        return session

    def is_ready(
        self,
        session_id: str,
    ) -> bool:

        session = self.get_session(session_id)

        if session is None:
            return False

        return (
            len(session.predictions)
            >= self.REQUIRED_FRAMES
        )

    def get_result(
        self,
        session_id: str,
    ) -> dict | None:

        session = self.get_session(session_id)

        if session is None:
            return None

        predictions = session.predictions

        if len(predictions) < self.REQUIRED_FRAMES:
            return None

        real_predictions = [
            prediction
            for prediction in predictions
            if prediction.get("class_id") == 2
        ]

        real_count = len(real_predictions)

        real_confidences = [
            float(prediction.get("confidence", 0.0))
            for prediction in real_predictions
        ]

        if real_confidences:
            average_confidence = (
                sum(real_confidences)
                / len(real_confidences)
            )
        else:
            average_confidence = 0.0

        is_real = (
            real_count >= self.MIN_REAL_FRAMES
            and average_confidence
            >= self.MIN_AVERAGE_CONFIDENCE
        )

        return {
            "is_real": is_real,
            "frames_analyzed": len(predictions),
            "real_frames": real_count,
            "spoof_frames": (
                len(predictions) - real_count
            ),
            "average_real_confidence": round(
                average_confidence,
                4,
            ),
            "required_frames": self.REQUIRED_FRAMES,
            "required_real_frames": self.MIN_REAL_FRAMES,
        }

    def remove_session(
        self,
        session_id: str,
    ) -> None:

        self.sessions.pop(
            session_id,
            None,
        )

    def cleanup_expired_sessions(self) -> None:

        now = time.monotonic()

        expired_sessions = [
            session_id
            for session_id, session
            in self.sessions.items()
            if (
                now - session.created_at
                > self.SESSION_TIMEOUT_SECONDS
            )
        ]

        for session_id in expired_sessions:
            self.sessions.pop(
                session_id,
                None,
            )


anti_spoofing_session_manager = (
    AntiSpoofingSessionManager()
)