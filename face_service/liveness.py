import time
from typing import Optional


class LivenessDetector:
    """
    Basic motion-based liveness detector.

    The detector observes the movement of a detected face across
    multiple camera frames.

    Liveness is considered successful when sufficient face movement
    is detected across consecutive frames.

    NOTE:
        This is a basic anti-spoofing mechanism. It can help detect
        static photos, but it is NOT a production-grade liveness
        detection system against sophisticated replay/deepfake attacks.
    """

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    MOVEMENT_THRESHOLD = 5.0
    REQUIRED_MOVEMENTS = 2

    # Once liveness passes, keep it valid for this many seconds.
    LIVENESS_VALID_SECONDS = 3.0

    # Maximum time allowed for a liveness attempt.
    SESSION_TIMEOUT_SECONDS = 10.0

    def __init__(self):
        self.previous_position: Optional[tuple[float, float]] = None

        self.movement_count = 0

        self.live_until = 0.0

        self.started_at = time.monotonic()

        self.completed = False

    # ---------------------------------------------------------
    # Reset detector
    # ---------------------------------------------------------

    def reset(self) -> None:
        """
        Reset the detector so a new liveness attempt can begin.
        """

        self.previous_position = None
        self.movement_count = 0
        self.live_until = 0.0
        self.started_at = time.monotonic()
        self.completed = False

    # ---------------------------------------------------------
    # Check whether session has timed out
    # ---------------------------------------------------------

    def is_expired(self) -> bool:
        """
        Return True if the liveness session exceeded its
        maximum allowed duration.
        """

        elapsed = time.monotonic() - self.started_at

        return elapsed > self.SESSION_TIMEOUT_SECONDS

    # ---------------------------------------------------------
    # Check whether liveness has already passed
    # ---------------------------------------------------------

    def is_live(self) -> bool:
        """
        Return True while the successful liveness result
        remains valid.
        """

        if not self.completed:
            return False

        return time.monotonic() < self.live_until

    # ---------------------------------------------------------
    # Process one face frame
    # ---------------------------------------------------------

    def check(self, face) -> bool:
        """
        Process one detected face.

        Args:
            face:
                YuNet face detection result:
                [x, y, width, height, ...]

        Returns:
            True  -> liveness passed
            False -> liveness not yet passed
        """

        # -----------------------------------------------------
        # Validate session
        # -----------------------------------------------------

        if self.is_expired():
            return False

        # -----------------------------------------------------
        # If already live, keep returning True until expiration
        # -----------------------------------------------------

        if self.is_live():
            return True

        # -----------------------------------------------------
        # Validate face data
        # -----------------------------------------------------

        if face is None:
            return False

        try:
            x = float(face[0])
            y = float(face[1])
            w = float(face[2])
            h = float(face[3])
        except (TypeError, ValueError, IndexError):
            return False

        if w <= 0 or h <= 0:
            return False

        # -----------------------------------------------------
        # Calculate face center
        # -----------------------------------------------------

        current_x = x + (w / 2.0)
        current_y = y + (h / 2.0)

        # -----------------------------------------------------
        # First frame
        # -----------------------------------------------------

        if self.previous_position is None:
            self.previous_position = (
                current_x,
                current_y,
            )

            return False

        # -----------------------------------------------------
        # Calculate movement
        # -----------------------------------------------------

        previous_x, previous_y = self.previous_position

        movement_x = abs(
            current_x - previous_x
        )

        movement_y = abs(
            current_y - previous_y
        )

        total_movement = (
            movement_x + movement_y
        )

        # Update previous position
        self.previous_position = (
            current_x,
            current_y,
        )

        # -----------------------------------------------------
        # Detect meaningful movement
        # -----------------------------------------------------

        if total_movement >= self.MOVEMENT_THRESHOLD:
            self.movement_count += 1

        else:
            # Slowly reduce the movement score instead of
            # immediately resetting it.
            self.movement_count = max(
                0,
                self.movement_count - 1,
            )

        # -----------------------------------------------------
        # Liveness passed
        # -----------------------------------------------------

        if self.movement_count >= self.REQUIRED_MOVEMENTS:

            self.completed = True

            self.live_until = (
                time.monotonic()
                + self.LIVENESS_VALID_SECONDS
            )

            self.movement_count = 0

            return True

        return False

    # ---------------------------------------------------------
    # Status information
    # ---------------------------------------------------------

    def status(self) -> dict:
        """
        Return the current liveness state.

        Useful for API responses and debugging.
        """

        return {
            "live": self.is_live(),
            "completed": self.completed,
            "movement_count": self.movement_count,
            "expired": self.is_expired(),
            "remaining_seconds": max(
                0.0,
                self.live_until - time.monotonic(),
            )
            if self.completed
            else 0.0,
        }
